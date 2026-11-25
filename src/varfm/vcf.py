"""Variants: read from a VCF (plain or gzipped) or written as strings such as '1-930204-G-A'."""

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

from .genome import normalize_contig


@dataclass(frozen=True)
class Variant:
    chrom: str  # normalized contig: 1..22, X, Y, MT
    pos: int    # 1-based position of the first REF base, as in the VCF
    ref: str
    alt: str
    id: str = "."

    @property
    def key(self) -> str:
        return f"GRCh38:{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def read_vcf(path) -> list[Variant]:
    """The variants of a VCF, one per ALT allele (multi-allelic records are split), in file order and without
    duplicates. Only CHROM, POS, ID, REF and ALT are read. The VCF must be on GRCh38."""
    with open(path, "rb") as fh:
        gzipped = fh.read(2) == b"\x1f\x8b"
    variants, seen = [], set()
    with (gzip.open if gzipped else open)(path, "rt") as fh:
        for n, line in enumerate(fh, 1):
            if line.startswith("##"):
                if line.lower().startswith("##reference") and re.search(r"grch37|hg19|b37|hs37d5", line, re.I):
                    raise ValueError(f"{path}: {line.strip()}: VarFM works on GRCh38 only")
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise ValueError(f"{path}, line {n}: a VCF record has at least 5 tab-separated columns")
            chrom, pos, vid, ref, alts = fields[:5]
            for alt in alts.split(","):
                variant = Variant(normalize_contig(chrom), int(pos), ref.upper(), alt.upper(), vid)
                if variant.key not in seen:
                    seen.add(variant.key)
                    variants.append(variant)
    return variants


def parse_variant(text: str) -> Variant:
    """'1-930204-G-A', '1:930204:G:A' or 'chr1 930204 G A' -> Variant."""
    parts = [p for p in re.split(r"[\s:\-]+", text.strip()) if p]
    if len(parts) != 4 or not parts[1].isdigit():
        raise ValueError(f"{text!r}: write a variant as chrom-pos-ref-alt, e.g. 1-930204-G-A")
    return Variant(normalize_contig(parts[0]), int(parts[1]), parts[2].upper(), parts[3].upper())


def load_variants(source) -> list[Variant]:
    """A VCF path, or a list of variants (strings or Variant)."""
    if isinstance(source, (str, Path)):
        return read_vcf(source)
    return [v if isinstance(v, Variant) else parse_variant(v) for v in source]
