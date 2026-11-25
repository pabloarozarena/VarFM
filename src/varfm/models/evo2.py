"""Evo 2 7B (Arc Institute): log-likelihood of the REF and ALT DNA windows.

REF is the 8,001 bp window of GRCh38 centred on the variant (4,000 bases on each side of the first REF base) and ALT
the same window carrying the alternative allele. Evo 2 reads DNA left to right, so the two sequences share their
prefix: when their lengths differ (indels), the longer one is trimmed on the right to the length L of the shorter one,
and the anchor a is the first position where they differ. For each sequence

    ll = sum of log p(x_t | x_<t) over t = a .. L-1      (natural log; the shared prefix adds the same to both)

and delta = ll_alt - ll_ref (lower = the ALT sequence is less likely). Log-probabilities are taken with log_softmax in
float32 and summed in float64; the model runs in bfloat16, never in FP8.

Evo 2 needs a CUDA GPU with bfloat16 (compute capability 8.0 or more) and about 20 GB of memory: an A100, L4 or H100
on Colab, not a T4.

Attention runs on PyTorch's scaled_dot_product_attention (vortex's own non-flash path) everywhere, so flash-attn, which
has to be compiled, is not needed, and every machine runs the same code. VARFM_EVO2_ATTENTION=flash uses flash-attn
instead, when it is installed. The two kernels round differently in bfloat16: over 8,000 positions the log-likelihoods
move by about 1 nat, so values computed with different settings should not be mixed (the setting is part of the
fingerprint). A batch only holds sequences of the same length: padding moves the results by a similar amount.
"""

import os
import sys
import types

import numpy as np

from .base import Item, Model, Unsupported

MODEL = "evo2_7b"
WINDOW = 8001
OFFSET = 4000  # index of the first REF base in the window
PAD = 1        # pad id of Evo 2's byte tokenizer
CHUNK = 1024   # positions turned into log-probabilities at a time
ACGT = frozenset("ACGT")


def build_pair(context: str, offset: int, ref: str, alt: str) -> tuple[str, str, int]:
    """REF and ALT sequences of equal length and their anchor; ValueError(reason) if the variant cannot run."""
    if set(ref) - ACGT or set(alt) - ACGT or set(context) - ACGT:
        raise ValueError("non_acgt_sequence")
    if context[offset:offset + len(ref)] != ref:
        raise ValueError("ref_mismatch")
    x_alt = context[:offset] + alt + context[offset + len(ref):]
    length = min(len(context), len(x_alt))
    x_ref, x_alt = context[:length], x_alt[:length]
    anchor = next((i for i in range(min(offset, length), length) if x_ref[i] != x_alt[i]), None)
    if anchor is None or anchor < 1:
        raise ValueError("no_difference_in_window")
    return x_ref, x_alt, anchor


def attention_backend() -> str:
    """'sdpa' (the default) or 'flash' (VARFM_EVO2_ATTENTION=flash)."""
    backend = os.environ.get("VARFM_EVO2_ATTENTION", "sdpa")
    if backend not in ("sdpa", "flash"):
        raise ValueError(f"VARFM_EVO2_ATTENTION must be sdpa or flash, not {backend!r}")
    return backend


def _flash_attn_installed() -> bool:
    """vortex imports flash-attn's CUDA extension as soon as it is imported, even when attention does not use it.
    Without flash-attn, a stand-in module lets vortex import; its functions fail if anything calls them."""
    module = sys.modules.get("flash_attn_2_cuda")
    if module is not None:
        return not getattr(module, "VARFM_STUB", False)
    try:
        import flash_attn_2_cuda  # noqa: F401
        return True
    except ImportError:
        stub = types.ModuleType("flash_attn_2_cuda")
        stub.VARFM_STUB = True

        def missing(*args, **kwargs):
            raise RuntimeError("flash-attn is not installed")

        def attribute(name):
            if name.startswith("__"):  # __file__, __path__...: tools such as inspect look these up on every module
                raise AttributeError(name)
            return missing

        stub.__getattr__ = attribute
        sys.modules["flash_attn_2_cuda"] = stub
        return False


class Evo2(Model):
    name = "evo2"
    columns = ("ll_ref", "ll_alt", "delta")
    batch_step = 2         # REF and ALT of a variant in the same batch
    equal_lengths = True   # no padding

    def load_peak_gb(self, shared: bool) -> float:
        # Shared memory: the 13.8 GB checkpoint read into RAM, the model on the GPU and the build's float32 copies, all
        # in the same memory. Otherwise the model is built on the GPU from a memory-mapped checkpoint.
        return 42.0 if shared else 3.0

    def __init__(self):
        self.net = self.device = None
        self.attention = ""

    def fingerprint(self) -> dict:
        return {"model": MODEL, "window": WINDOW, "offset": OFFSET, "trim": "right", "anchor": "first difference",
                "ll": "sum of log p(x_t | x_<t) for t >= anchor, float64", "precision": "bfloat16", "fp8": False,
                "attention": attention_backend(), "batches": "equal lengths"}

    def prepare(self, variant, annotation, reference):
        try:
            context = reference.fetch(variant.chrom, variant.pos - 1 - OFFSET, variant.pos + OFFSET)
            x_ref, x_alt, anchor = build_pair(context, OFFSET, variant.ref, variant.alt)
        except ValueError as e:
            return str(e)
        return [Item(f"{variant.key}|ref", variant.key, x_ref, {"anchor": anchor}),
                Item(f"{variant.key}|alt", variant.key, x_alt, {"anchor": anchor})]

    def load(self, device: str, guard) -> None:
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise Unsupported("Evo 2 needs a CUDA GPU")
        props = torch.cuda.get_device_properties(torch.device(device))
        if props.major < 8:
            raise Unsupported(f"Evo 2 needs bfloat16 (compute capability 8.0+): {props.name} is "
                              f"{props.major}.{props.minor}")
        if not props.is_integrated and props.total_memory < 20e9:
            raise Unsupported(f"Evo 2 7B needs about 20 GB of GPU memory: {props.name} has "
                              f"{props.total_memory / 1e9:.0f} GB")
        installed = _flash_attn_installed()  # without flash-attn, this also lets vortex import
        flash = attention_backend() == "flash"
        if flash and not installed:
            raise RuntimeError("VARFM_EVO2_ATTENTION=flash, but flash-attn is not installed")
        import evo2.models

        build, torch_load = evo2.models.StripedHyena, torch.load

        def build_without_fp8(config, *args, **kwargs):
            config["use_fp8_input_projections"] = False
            config["use_flash_attn"] = flash
            return build(config, *args, **kwargs)

        def load_without_mmap(*args, **kwargs):
            return torch_load(*args, **{**kwargs, "mmap": False})

        # On shared memory, copying the weights to the GPU from a memory-mapped file is pathologically slow (over an
        # hour on a GB10 instead of seconds): the checkpoint is read into RAM in one go instead.
        evo2.models.StripedHyena = build_without_fp8
        if guard.shared:
            torch.load = load_without_mmap
        try:
            wrapper = evo2.models.Evo2(MODEL)
        finally:
            evo2.models.StripedHyena, torch.load = build, torch_load
        if [int(i) for i in wrapper.tokenizer.tokenize("ACGT")] != list(b"ACGT") or wrapper.tokenizer.pad_id != PAD:
            raise RuntimeError("Evo 2's tokenizer is no longer the byte tokenizer this code reproduces")
        if any("float8" in str(p.dtype) for p in wrapper.model.parameters()):
            raise RuntimeError("Evo 2 was built with FP8 weights")
        self.net = wrapper.model.eval().requires_grad_(False)
        self.device = next(self.net.parameters()).device
        self.attention = "flash-attn" if flash else "sdpa"
        print(f"evo2: attention on {self.attention}", flush=True)

    def batch_sizes(self, guard) -> tuple[int, int]:
        # Measured on a GB10 (SDPA, 8,001 bases): 13.2 GB of weights, then ~2.5 GB plus ~2.2 GB per sequence, i.e. a
        # peak of 18.5 GB for REF + ALT (fits an L4) and 31 GB for 8 sequences (an A100 40 GB takes about 8).
        most = max(1, min(16, int((guard.gpu_room_gb() - 3.0) / 2.5)))
        if most >= 2:
            most -= most % 2
        start = max(1, most // 2)
        return (start - start % 2 if start >= 2 else start), most

    def run(self, items: list) -> list[dict]:
        import torch

        lengths = np.array([len(it.seq) for it in items])
        ids = np.full((len(items), lengths.max()), PAD, dtype=np.int64)
        for i, it in enumerate(items):
            ids[i, :len(it.seq)] = np.frombuffer(it.seq.encode("ascii"), dtype=np.uint8)
        with torch.inference_mode():
            x = torch.from_numpy(ids).to(self.device)
            out = self.net(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            anchor = torch.tensor([it.extra["anchor"] for it in items], device=logits.device)
            length = torch.from_numpy(lengths).to(logits.device)
            ll = torch.zeros(len(items), dtype=torch.float64, device=logits.device)
            width = x.shape[1]
            for j0 in range(0, width - 1, CHUNK):  # column j predicts token j + 1
                j1 = min(j0 + CHUNK, width - 1)
                lp = torch.log_softmax(logits[:, j0:j1].float(), dim=-1)
                lp = lp.gather(-1, x[:, j0 + 1:j1 + 1, None]).squeeze(-1).double()
                j = torch.arange(j0, j1, device=logits.device)
                keep = (j[None, :] >= anchor[:, None] - 1) & (j[None, :] < length[:, None] - 1)
                ll += lp.masked_fill(~keep, 0.0).sum(-1)
            values = ll.cpu().tolist()
        return [{"ll": v} for v in values]

    def combine(self, key: str, values: dict) -> dict:
        ref, alt = values[f"{key}|ref"]["ll"], values[f"{key}|alt"]["ll"]
        return {"ll_ref": ref, "ll_alt": alt, "delta": alt - ref}

    def unload(self) -> None:
        self.net = None
