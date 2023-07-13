"""Utility functions for training and inference."""

from functools import partial
import pickle
import sys
import warnings
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import MethodType
from typing import Optional, Any, Union, List, TypeVar, Type

import torch
import torch.utils._device
from lightning.fabric.loggers import CSVLogger
from torch.serialization import normalize_storage_type


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@contextmanager
def quantization(mode: Optional[str] = None):
    if mode is None:
        yield
        return

    if mode == "bnb.int8":
        from quantize.bnb import InferenceLinear8bitLt

        quantized_linear_cls = InferenceLinear8bitLt
    elif mode == "bnb.fp4":
        from quantize.bnb import Linear4bit

        # Use a class instead `functools.partial` to respect `isinstance` checks and attribute accesses
        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=False, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.fp4-dq":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=True, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=False, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4-dq":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=True, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "gptq.int4":
        from quantize.gptq import ColBlockQuantizedLinear

        class QuantizedLinear(ColBlockQuantizedLinear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, bits=4, tile_cols=-1, **kwargs)

        quantized_linear_cls = QuantizedLinear
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    torch_linear_cls = torch.nn.Linear
    torch.nn.Linear = quantized_linear_cls
    yield
    torch.nn.Linear = torch_linear_cls


# this is taken from torchhacks https://github.com/lernapparat/torchhacks


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func, new_type, args, state, *, archiveinfo=None):
        ret = func(*args)
        if isinstance(ret, NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)

            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(cls, data, requires_grad, backward_hooks, *, archiveinfo=None):
        if isinstance(data, NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls, storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None, *, archiveinfo=None
    ):
        rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        metatensor = torch._utils._rebuild_tensor_v2(
            storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata
        )
        storageinfo = storage.archiveinfo
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        name, storage_cls, fn, device, size = self.storageinfo
        dtype = self.metatensor.dtype

        uts = (
            self.archiveinfo.zipfile_context.zf.get_storage_from_record(
                f"data/{fn}", size * torch._utils._element_size(dtype), torch.UntypedStorage
            )
            ._typed_storage()
            ._untyped_storage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = torch.storage.TypedStorage(wrap_storage=uts, dtype=self.metatensor.dtype, _internal=True)
        tensor = torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [(a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args]
        res = func(*loaded_args, **kwargs)
        # gc.collect would be costly here, maybe do it optionally
        return res

    def __getattr__(self, name):
        # properties
        ## TODO: device, is_...??
        ## TODO: mH, mT, H, T, data, imag, real
        ## name ???
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        # materializing with contiguous is needed for quantization
        if name in {"contiguous"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({repr(self.metatensor)})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile_context):
        super().__init__(file)
        self.zipfile_context = zipfile_context

    def find_class(self, module, name):
        res = super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return partial(NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        elif module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return partial(NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        elif module == "torch._utils" and name == "_rebuild_parameter":
            return partial(NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return res

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        s.archiveinfo = pid
        return s


class lazy_load:
    def __init__(self, fn):
        self.zf = torch._C.PyTorchFileReader(str(fn))
        with BytesIO(self.zf.get_record("data.pkl")) as pkl:
            mup = LazyLoadingUnpickler(pkl, self)
            self.sd = mup.load()

    def __enter__(self):
        return self.sd

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.zf  # I don't think there is a way to force closing...
        self.zf = None


def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    files = {
        "lit_model.pth": (checkpoint_dir / "lit_model.pth").is_file(),
        "lit_config.json": (checkpoint_dir / "lit_config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or (
            checkpoint_dir / "tokenizer.model"
        ).is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            # we're good
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    # list locally available checkpoints
    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = f"\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials\n"
        f"{extra}\nSee all download options by running:\n python scripts/download.py"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, (storage, *other_reduce_args) = tensor.__reduce_ex__(protocol_version)
        assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
        storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
        self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


T = TypeVar("T")


def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    logger = cls(*args, **kwargs)

    def merge_by(dicts, key):
        from collections import defaultdict

        out = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]

    def save(self) -> None:
        """Overridden to merge CSV by the step number."""
        import csv

        if not self.metrics:
            return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

    logger.experiment.save = MethodType(save, logger.experiment)

    return logger


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        return torch.cat(loss_chunks).mean()

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    return torch.cat(loss_chunks).mean()
