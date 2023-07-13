import re
import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Iterator

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, Block
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization
from scripts.prepare_alpaca import generate_prompt


lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


def prompt_config(
    checkpoint_dir: Path, tokenizer: Tokenizer
) -> Tuple[str, Tuple[List[int], ...]]:
    checkpoint_name = str(checkpoint_dir)
    # First line could be modified. AFAIK Falcon doesn't impose a specific system prompt
    # The instruction to not prefix its replies doesn't work always, but better than nothing
    system_prompt = "Do not prefix your replies with 'Bot: '\nUser: {prompt}\n"
    # I've also tried just "{prompt}\n" but the model seems to ramble more often
    stop_tokens = (
        [tokenizer.eos_id],
        # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
        # to stop or else things like code generation wouldn't work
        [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
        [193, tokenizer.token_to_id("User")],  # 193: '\n'
    )
    return system_prompt, stop_tokens


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    stop_tokens: Tuple[List[int], ...] = tuple(),
):
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as possible.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        stop_tokens: If specified, stop generating any more token once one of this list is generated.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device = idx.device
    stop_tokens = [torch.tensor(tokens, device=device) for tokens in stop_tokens]
    input_pos = torch.arange(0, T, device=device)

    # buffer holds the tokens that haven't been yield yet
    buffer_length = max((len(tokens) for tokens in stop_tokens), default=1)
    buffer = torch.full(
        (buffer_length,), -999, device=device
    )  # fill with non-existing token

    yield_i = -1
    # generate up to a fixed number of tokens
    for t in range(max_returned_tokens - T):
        # forward
        logits = model(idx.view(1, -1), max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        buffer[min(t, buffer_length - 1)] = idx

        # check the stop condition
        for tokens in stop_tokens:
            l = len(tokens)
            if torch.equal(buffer[-l:], tokens):
                # stop token hit, yield any leftovers that aren't part of it
                if buffer_length > l:  # avoid an empty yield
                    yield buffer[:-l]
                return
        # if the buffer is full
        if t - yield_i >= buffer_length:
            # we know this idx is not part of stop tokens, safe to yield
            yield buffer[0]
            # roll once to the left, as next generation will be put at the end
            buffer = torch.roll(buffer, -1, 0)
            yield_i += 1


def token_decode(
    fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]
) -> int:
    tokens_generated = 0
    valid_token = {"History", "Examination", "Plan", "Diagnosis"}
    if tokenizer.backend == "huggingface":
        for token in token_stream:
            tokens_generated += 1
            decoded_token = tokenizer.decode(token)
            if decoded_token in valid_token:
                print(decoded_token)
                fabric.print("\n\n" + decoded_token, end="", flush=True)
            else:
                fabric.print(tokenizer.decode(token), end="", flush=True)
    else:
        raise NotImplementedError(tokenizer.backend)
    return tokens_generated


def main(
    input: str = "",
    instruction: str = "",
    lora_path: Path = Path(""),
    checkpoint_dir: Path = Path(f"lit-gpt/checkpoints/tiiuae/falcon-7b"),
    quantize: Literal["llm.int8", "gptq.int4"] = None,
    max_new_tokens: int = 400,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: str = "bf16-true",
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    if strategy == "fsdp":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={Block}
        )
        strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()
    check_valid_checkpoint_dir(checkpoint_dir)

    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(
            r=lora_r, alpha=lora_alpha, dropout=lora_dropout, **json.load(fp)
        )

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.time()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr
    )
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(
        lora_path
    ) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)
    fabric.print(
        f"Time to load the model weights: {time.time() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    model.eval()
    model = fabric.setup_module(model)
    tokenizer = Tokenizer(checkpoint_dir)
    system_prompt, stop_tokens = prompt_config(checkpoint_dir, tokenizer)

    while True:
        try:
            instruction = input(">> Prompt: ")
            input = input(">> Input: ")
        except KeyboardInterrupt:
            break
        if not input:
            sample = {"instruction": instruction}
        else:
            sample = {"instruction": instruction, "input": input}

        t0 = time.time()
        prompt = generate_prompt(sample)
        encoded_prompt = tokenizer.encode(prompt, device=model.device)
        prompt_length = encoded_prompt.size(0)
        max_returned_tokens = prompt_length + max_new_tokens + 25

        t0 = time.perf_counter()
        y = generate(
            model,
            encoded_prompt,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_tokens,
        )
        try:
            t = time.perf_counter() - t0
            tokens_generated = token_decode(fabric, tokenizer, y)
            model.reset_cache()
            fabric.print(
                f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
                file=sys.stderr,
            )
            if fabric.device.type == "cuda":
                fabric.print(
                    f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
                    file=sys.stderr,
                )
        except KeyboardInterrupt:
            pass
        fabric.print()


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore",
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
