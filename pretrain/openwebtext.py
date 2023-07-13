import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Union

import lightning as L
import numpy as np
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor, measure_flops, estimate_flops
from lit_gpt.utils import step_csv_logger, chunked_cross_entropy

model_name = "pythia-70m"
name = "openwebtext"
out_dir = Path("out") / name
data_dir = Path("data") / name
save_interval = 10
eval_interval = 1000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_interval)


def setup(
    devices: int = 1, precision: Optional[str] = None, tpu: bool = False, resume: Union[bool, Path] = False
) -> None:
    if precision is None:
        precision = "32-true" if tpu else "bf16-mixed"
    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                activation_checkpointing=Block,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, resume=resume)


def main(fabric, resume) -> None:
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = load_datasets(data_dir)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.time()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(model._init_weights)
    fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.")

    num_total_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"Total parameters {num_total_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    model, optimizer = fabric.setup(model, optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.time()
    train(fabric, state, train_data, val_data, speed_monitor)
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")


def train(fabric, state, train_data, val_data, speed_monitor):
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_data)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # estimated is too much of an optimistic estimate, left just for reference
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.time()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    for state["iter_num"] in range(state["iter_num"], max_iters):
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data, model.config.block_size)

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        t1 = time.time()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (state["iter_num"] + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if state["iter_num"] % log_interval == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and state["step_count"] % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, val_data)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, model.config.block_size)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    out = losses.mean()

    model.train()
    return out


def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (micro_batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def load_datasets(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
