"""ESM2 650M (fair-esm): masked marginal of missense SNVs.

For a missense SNV of a MANE Select protein, the protein is read in a window of at most 1,022 residues (1,024 tokens
with BOS and EOS, ESM2's training length) that starts 511 residues before the site, shifted to stay inside the
protein. The site is masked and

    lp_wt  = log p(reference residue | window with the site masked)
    lp_mut = log p(mutant residue    | window with the site masked)
    llr    = lp_mut - lp_wt          (the masked marginal score; lower = more damaging)

with log_softmax over the whole vocabulary in float32, natural log. The model runs in float32 with TF32 off.
"""

from ..annotate import Annotation
from .base import Item, Model

MODEL = "esm2_t33_650M_UR50D"
URL = f"https://dl.fbaipublicfiles.com/fair-esm/models/{MODEL}.pt"
WINDOW = 1022
HALF = WINDOW // 2
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REASONS = {"synonymous": "protein_unchanged", "start_lost": "no_mutant_protein", "intergenic": "no_transcript",
           "5_prime_utr": "outside_cds", "3_prime_utr": "outside_cds", "intron": "outside_cds",
           "splice_site": "outside_cds"}  # other coding changes (stop, indels, MNVs): unsupported_kind


def window_start(site: int, length: int) -> int:
    """First residue of the window: HALF residues before the site, shifted to stay inside the protein."""
    return min(max(site - HALF, 0), max(length - WINDOW, 0))


class ESM2(Model):
    name = "esm2"
    columns = ("lp_wt", "lp_mut", "llr")

    def load_peak_gb(self, shared: bool) -> float:
        # the 2.6 GB float32 checkpoint in CPU memory, the model built on the CPU, then (shared memory) its GPU copy
        return 8.0 if shared else 6.0

    def __init__(self):
        self.net = self.alphabet = self.device = None

    def fingerprint(self) -> dict:
        return {"model": MODEL, "score": "masked marginal", "window": WINDOW, "precision": "float32", "tf32": False}

    def prepare(self, variant, ann: Annotation, reference):
        if ann.consequence != "missense":
            return REASONS.get(ann.consequence, "unsupported_kind")
        if not ann.protein_ok:
            return "fasta_mismatch"
        if not AMINO_ACIDS.issuperset(ann.protein + ann.aa_alt):
            return "non_standard_residue"
        site = ann.protein_pos - 1
        start = window_start(site, len(ann.protein))
        window = ann.protein[start:start + WINDOW]
        return [Item(variant.key, variant.key, window, {"site": site - start, "wt": ann.aa_ref, "mut": ann.aa_alt})]

    def load(self, device: str, guard) -> None:
        import argparse
        import warnings

        import esm
        import torch

        # fair-esm's checkpoint holds the training arguments as an argparse.Namespace: allow that class only
        with torch.serialization.safe_globals([argparse.Namespace]):
            data = torch.hub.load_state_dict_from_url(URL, map_location="cpu", progress=True, weights_only=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Regression weights not found")
            net, alphabet = esm.pretrained.load_model_and_alphabet_core(MODEL, data, None)
        del data
        self.net = net.float().to(device).eval().requires_grad_(False)
        self.alphabet, self.device = alphabet, device

    def batch_sizes(self, guard) -> tuple[int, int]:
        if not self.device.startswith("cuda"):
            return 4, 16
        most = max(1, min(64, int((guard.gpu_room_gb() - 1.0) / 0.25)))  # ~0.25 GB per 1,024-token sequence
        return max(1, most // 4), most

    def run(self, items: list) -> list[dict]:
        import torch

        a = self.alphabet
        tokens = torch.full((len(items), max(len(it.seq) for it in items) + 2), a.padding_idx, dtype=torch.long)
        tokens[:, 0] = a.cls_idx
        for i, it in enumerate(items):
            tokens[i, 1:1 + len(it.seq)] = torch.tensor([a.get_idx(c) for c in it.seq])
            tokens[i, 1 + len(it.seq)] = a.eos_idx
            tokens[i, 1 + it.extra["site"]] = a.mask_idx
        rows = torch.arange(len(items))
        sites = torch.tensor([1 + it.extra["site"] for it in items])
        wt = torch.tensor([a.get_idx(it.extra["wt"]) for it in items])
        mut = torch.tensor([a.get_idx(it.extra["mut"]) for it in items])
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.inference_mode():
                logits = self.net(tokens.to(self.device))["logits"]
                lp = torch.log_softmax(logits[rows, sites.to(logits.device)].float(), dim=-1).double().cpu()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return [{"lp_wt": float(lp[i, wt[i]]), "lp_mut": float(lp[i, mut[i]])} for i in range(len(items))]

    def combine(self, key: str, values: dict) -> dict:
        v = values[key]
        return {"lp_wt": v["lp_wt"], "lp_mut": v["lp_mut"], "llr": v["lp_mut"] - v["lp_wt"]}

    def unload(self) -> None:
        self.net = self.alphabet = None
