from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock

import torch
from lightning import Fabric


def test_lora_layer_replacement():
    from lit_gpt.lora import CausalSelfAttention as LoRACausalSelfAttention, GPT, Config

    config = Config(n_layer=2, n_head=4, n_embd=8, block_size=8, vocab_size=8, r=8, alpha=8, dropout=0.1)
    model = GPT(config)

    assert isinstance(model.transformer.h[0].attn, LoRACausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, LoRACausalSelfAttention)


def test_lora_merge_unmerge():
    from lit_gpt.lora import mark_only_lora_as_trainable, GPT, Config

    config = Config(n_layer=1, n_head=2, n_embd=8, block_size=8, vocab_size=8, r=8, alpha=8, dropout=0.1, to_query=True, to_value=True)
    model = GPT(config)

    initial_weight = model.transformer.h[0].attn.attn.weight.clone()
    model.train()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, initial_weight)

    # perform an update to the LoRA weights
    mark_only_lora_as_trainable(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = model(torch.randint(0, 8, size=(2, 4), dtype=torch.int64))
    y.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    # the weight remains unchanged (only lora A and B change)
    assert torch.equal(model.transformer.h[0].attn.attn.weight, initial_weight)

    # 'merge' and then 'unmerge' should neutralize themselves
    weight_before = model.transformer.h[0].attn.attn.weight.clone()
    model.eval()
    assert not torch.equal(model.transformer.h[0].attn.attn.weight, weight_before)
    model.train()
    # note: numerically, `W + (A * B) - (A * B) == W` does not hold exactly
    torch.testing.assert_close(model.transformer.h[0].attn.attn.weight, weight_before)

    # calling eval/train multiple times in a row should not merge/unmerge multiple times
    model.eval()
    assert model.transformer.h[0].attn.attn.merged
    weight_after = model.transformer.h[0].attn.attn.weight.clone()
    model.eval()
    model.eval()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, weight_after)
    model.train()
    assert not model.transformer.h[0].attn.attn.merged
    weight_after = model.transformer.h[0].attn.attn.weight.clone()
    model.train()
    model.train()
    assert torch.equal(model.transformer.h[0].attn.attn.weight, weight_after)


def test_lora_mqa_gqa():
    from lit_gpt.lora import GPT, Config

    # MHA
    config = Config(n_layer=1, n_head=4, n_embd=8, block_size=1, vocab_size=1, r=2, alpha=8, dropout=0.1, to_query=True, to_value=True)
    assert config.n_query_groups == config.n_head
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (24, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (16, 2)
    assert attn.lora_ind.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]
    x = torch.randint(0, 8, size=(3, 5, 16), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 24)

    # MQA
    config.n_query_groups = 1
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (12, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (10, 2)
    assert attn.lora_ind.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
    x = torch.randint(0, 8, size=(3, 5, 10), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 12)

    # GQA
    config.n_query_groups = 2
    model = GPT(config)
    attn = model.transformer.h[0].attn.attn
    assert attn.weight.shape == (16, 8)
    assert attn.lora_A.shape == (4, 8)
    assert attn.lora_B.shape == (12, 2)
    assert attn.lora_ind.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    x = torch.randint(0, 8, size=(3, 5, 12), dtype=torch.int64)
    assert attn.zero_pad(x).shape == (3, 5, 16)


def test_lora_filter(tmp_path):
    from lit_gpt.lora import lora_filter, GPT

    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-70m", n_layer=3, r=1, to_query=True, to_value=True)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": lora_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.1.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_B",
        "transformer.h.2.attn.attn.lora_A",
        "transformer.h.1.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_A",
        "transformer.h.0.attn.attn.lora_B",
    }
    assert set(saved) == expected


def test_lora_script(tmp_path, fake_checkpoint_dir, monkeypatch):
    import finetune.lora as module

    module.gradient_accumulation_iters = 1
    module.save_interval = 2
    module.eval_interval = 2
    module.eval_iters = 2
    module.max_iters = 6

    data = [
        {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
    ]
    torch.save(data, tmp_path / "train.pt")
    torch.save(data, tmp_path / "test.pt")

    from lit_gpt.config import name_to_config

    model_config = dict(block_size=128, n_layer=2, n_embd=8, n_head=4, padded_vocab_size=8)
    monkeypatch.setitem(name_to_config, "tmp", model_config)

    load_mock = Mock()
    load_mock.return_value = load_mock
    load_mock.__enter__ = Mock()
    load_mock.__exit__ = Mock()
    monkeypatch.setattr(module, "lazy_load", load_mock)

    tokenizer_mock = Mock()
    tokenizer_mock.return_value = tokenizer_mock
    tokenizer_mock.encode = lambda *_, **kwargs: torch.tensor([3, 2, 1], **kwargs)
    monkeypatch.setattr(module, "Tokenizer", tokenizer_mock)

    stdout = StringIO()
    with redirect_stdout(stdout):
        module.setup(data_dir=tmp_path, checkpoint_dir=fake_checkpoint_dir, out_dir=tmp_path, precision="32-true")

    assert set(p.name for p in tmp_path.glob("*.pth")) == {
        "iter-000001-ckpt.pth",
        "iter-000003-ckpt.pth",
        "iter-000005-ckpt.pth",
        "lit_model_lora_finetuned.pth",
    }
    assert (tmp_path / "version_0" / "metrics.csv").is_file()

    logs = stdout.getvalue()
    assert logs.count("optimizer.step") == module.max_iters
    assert logs.count("val loss") == module.max_iters // module.eval_interval
    assert "of trainable parameters: 512" in logs


def test_lora_init_when_linear_overridden():
    from lit_gpt.lora import MergedLinear

    class MyLinear(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            # this needs to be implemented to demonstrate the failure
            super().__init__(*args, **kwargs)

    original_linear = torch.nn.Linear
    # Our bnb does this sort of monkey patching
    torch.nn.Linear = MyLinear
    layer = MergedLinear(1, 1, 1, 1)
    assert isinstance(layer, original_linear)
    torch.nn.Linear = original_linear
