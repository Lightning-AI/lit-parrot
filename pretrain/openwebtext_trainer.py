import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional, Any

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy, XLAStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.speed_monitor import measure_flops, estimate_flops, SpeedMonitorCallback
from lit_gpt.utils import step_csv_logger, chunked_cross_entropy

model_name = "pythia-70m"
name = "openwebtext"
out_dir = Path("out") / name
data_dir = Path("data") / name
save_interval = 1000
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
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


class LightningGPTModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.measured_flops: Optional[int] = None

    def configure_model(self) -> None:
        self.module = GPT(self.config)
        self.module.apply(self.module._init_weights)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.module.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
        )

    def on_fit_start(self) -> None:
        trainer = self.trainer
        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            # estimated is too much of an optimistic estimate, left just for reference
            estimated_flops = estimate_flops(meta_model) * micro_batch_size
            self.print(f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
            self.measured_flops = measure_flops(meta_model, x)
            self.print(f"Measured TFLOPs: {self.measured_flops * trainer.world_size / 1e12:.2f}")

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(self.trainer.fit_loop.total_batch_idx)
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits = self.module(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


def main(devices: int = 1, precision: Optional[str] = None, tpu: bool = False) -> None:
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
                # the argument is not available in the Trainer strategy, but it's the default anyways
                # state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    logger = step_csv_logger("out", name, cls=CSVLogger, flush_logs_every_n_steps=log_interval)
    speed_monitor = SpeedMonitorCallback(
        length_fn=lambda batch: batch[0].size(1), batch_size=micro_batch_size, window_size=50, time_unit="seconds"
    )
    model_checkpoint = ModelCheckpoint(dirpath=out_dir, every_n_train_steps=save_interval, save_last=True, verbose=True)
    trainer = L.Trainer(
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=[speed_monitor, model_checkpoint],
        max_steps=max_iters,
        max_epochs=1,
        limit_val_batches=eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_interval,
        val_check_interval=eval_interval,
    )

    L.seed_everything(1337 + trainer.global_rank, workers=True)

    trainer.print(hparams)

    if trainer.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)
    trainer.print(f"Loading model with {config.__dict__}")
    t0 = time.time()
    model = LightningGPTModule(config)
    trainer.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.")

    train_data = Dataset(str(data_dir / "train.bin"), config.block_size)
    val_data = Dataset(str(data_dir / "val.bin"), config.block_size)

    t0 = time.time()
    trainer.fit(model, train_data, val_data, ckpt_path="last")
    trainer.print(f"Training time: {(time.time()-t0):.2f}s")


class Dataset:
    def __init__(self, bin: str, block_size: int) -> None:
        self.data = np.memmap(bin, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __iter__(self):
        while True:
            ix = torch.randint(len(self.data) - self.block_size, (micro_batch_size,))
            x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack(
                [torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix]
            )
            yield x, y


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

    CLI(main)
