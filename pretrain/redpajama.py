import glob
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import Block, GPT, Config
from lit_gpt.packed_dataset import PackedDataset, CombinedDataset
from lit_gpt.utils import step_csv_logger, chunked_cross_entropy
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor, estimate_flops, measure_flops

model_name = "pythia-70m"
name = "redpajama"
out_dir = Path("out") / name
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


# Data proportions from https://arxiv.org/pdf/2302.13971.pdf Table 1
data_config = [
    ("arxiv", 2.5),
    ("book", 4.5),
    ("c4", 15.0),
    ("cc", 67.0),
    ("github", 4.5),
    ("stackexchange", 2.0),
    ("wikipedia", 4.5),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_interval)


def setup(
    devices: int = 4,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
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
    fabric.launch(main, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1337,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

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
    train(fabric, state, train_dataloader, val_dataloader, speed_monitor)
    fabric.print(f"Training time: {(time.time()-train_time):.2f}s")


def train(fabric, state, train_dataloader, val_dataloader, speed_monitor):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

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
    for state["iter_num"], train_data in enumerate(train_dataloader, state["iter_num"]):
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.time()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

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

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_interval == 0:
            t0 = time.time()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.time() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        losses[k] = loss.item()
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = glob.glob(str(data_dir / f"{prefix}*"))
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


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
