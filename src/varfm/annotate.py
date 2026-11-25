"""Where a variant falls on MANE Select transcripts, and what it does to the protein (MANE v1.5, Ensembl IDs).

The transcript models come from MANE's GFF and the proteins from MANE's FASTA. For a SNV in a coding sequence, the
codon change is read from the reference: the coding sequence of the transcript is rebuilt from GRCh38 and translated,
and it must give MANE's own protein (otherwise `protein_ok` is False and ESM2 does not use that transcript).
"""

import gzip
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .assets import cache_dir, download
from .genome import LENGTHS, normalize_contig

MANE_RELEASE = "1.5"
URL = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_{v}/{name}"
GFF = "MANE.GRCh38.v{v}.ensembl_genomic.gff.gz"
PROTEINS = "MANE.GRCh38.v{v}.ensembl_protein.faa.gz"
BIN = 1_000_000  # transcripts are indexed in bins of 1 Mb
SPLICE = 2       # intronic bases on each side of an exon that make the splice site

_BASES = "TCAG"
_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS = {a + b + c: _AMINO[16 * i + 4 * j + k]
          for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def translate(cds: str) -> str:
    return "".join(CODONS.get(cds[i:i + 3], "X") for i in range(0, len(cds) - 2, 3))


@dataclass
class Transcript:
    id: str
    gene: str
    protein_id: str
    contig: str
    strand: str
    start: int  # 1-based, inclusive
    end: int
    exons: list = field(default_factory=list)   # (start, end), sorted
    coding: list = field(default_factory=list)  # CDS and stop codon, merged and sorted


@dataclass
class Annotation:
    gene: str = ""
    transcript: str = ""
    consequence: str = "intergenic"
    protein: str = ""      # the MANE protein, for a SNV in the coding sequence
    protein_pos: int = 0   # 1-based residue changed by that SNV
    aa_ref: str = ""
    aa_alt: str = ""
    protein_ok: bool = True

    @property
    def protein_change(self) -> str:
        return f"p.{self.aa_ref}{self.protein_pos}{self.aa_alt}" if self.protein_pos else ""


def _merge(segments):
    merged = []
    for start, end in sorted(segments):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlaps(segments, start, end) -> bool:
    return any(s <= end and start <= e for s, e in segments)


def _trim(pos: int, ref: str, alt: str) -> tuple[int, str, str]:
    """The changed bases only: shared leading and trailing bases removed (an insertion keeps an empty REF)."""
    while ref and alt and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    while ref and alt and ref[0] == alt[0]:
        ref, alt, pos = ref[1:], alt[1:], pos + 1
    return pos, ref, alt


class Mane:
    """MANE Select transcripts and proteins, downloaded once into <cache>/MANE and read on first use."""

    def __init__(self, reference, root=None):
        self.reference = reference
        self.root = Path(root) if root else cache_dir() / "MANE"
        self._bins = None
        self._proteins = None
        self._cds = {}

    def annotate(self, v) -> Annotation:
        self._load()
        start, ref, alt = _trim(v.pos, v.ref, v.alt)
        end = start + len(ref) - 1
        if not ref:  # an insertion between start - 1 and start: both flanking bases
            start, end = start - 1, start
        hits = {t.id: t for b in {start // BIN, end // BIN} for t in self._bins.get((v.chrom, b), ())
                if t.start <= end and start <= t.end}
        if not hits:
            return Annotation()
        coding = [t for t in hits.values() if _overlaps(t.coding, start, end)]
        t = min(coding or hits.values(), key=lambda t: (t.gene, t.id))
        ann = Annotation(gene=t.gene, transcript=t.id)
        if coding:
            if len(ref) == len(alt) == 1:
                self._codon_change(t, start, alt, ann)
            elif len(ref) == len(alt):
                ann.consequence = "coding_mnv"
            else:
                ann.consequence = "frameshift" if (len(alt) - len(ref)) % 3 else "inframe_indel"
        elif _overlaps(t.exons, start, end):
            five_prime = start < t.coding[0][0] if t.strand == "+" else end > t.coding[-1][1]
            ann.consequence = "5_prime_utr" if five_prime else "3_prime_utr"
        else:
            sites = [(e + 1, e + SPLICE) for _, e in t.exons[:-1]] + [(s - SPLICE, s - 1) for s, _ in t.exons[1:]]
            ann.consequence = "splice_site" if _overlaps(sites, start, end) else "intron"
        return ann

    def _codon_change(self, t: Transcript, pos: int, alt: str, ann: Annotation) -> None:
        cds = self._coding_sequence(t)
        offset = 0
        for s, e in (t.coding if t.strand == "+" else t.coding[::-1]):
            if s <= pos <= e:
                offset += pos - s if t.strand == "+" else e - pos
                break
            offset += e - s + 1
        index, frame = divmod(offset, 3)
        codon = cds[3 * index:3 * index + 3]
        base = alt if t.strand == "+" else alt.translate(_COMPLEMENT)
        ann.protein_pos = index + 1
        ann.aa_ref = CODONS.get(codon, "X")
        ann.aa_alt = CODONS.get(codon[:frame] + base + codon[frame + 1:], "X")
        ann.protein = self._proteins.get(t.protein_id, "")
        ann.protein_ok = bool(ann.protein) and translate(cds).rstrip("*") == ann.protein
        if ann.aa_ref == "*":
            ann.consequence = "synonymous" if ann.aa_alt == "*" else "stop_lost"
        elif ann.aa_alt == "*":
            ann.consequence = "stop_gained"
        elif index == 0 and ann.aa_ref == "M":
            ann.consequence = "synonymous" if ann.aa_alt == "M" else "start_lost"
        else:
            ann.consequence = "synonymous" if ann.aa_ref == ann.aa_alt else "missense"

    def _coding_sequence(self, t: Transcript) -> str:
        if t.id not in self._cds:
            cds = "".join(self.reference.fetch(t.contig, s - 1, e) for s, e in t.coding)
            self._cds[t.id] = cds if t.strand == "+" else reverse_complement(cds)
        return self._cds[t.id]

    def _load(self) -> None:
        if self._bins is not None:
            return
        gff = download(URL.format(v=MANE_RELEASE, name=GFF.format(v=MANE_RELEASE)), self.root / GFF.format(v=MANE_RELEASE))
        faa = download(URL.format(v=MANE_RELEASE, name=PROTEINS.format(v=MANE_RELEASE)),
                       self.root / PROTEINS.format(v=MANE_RELEASE))
        transcripts = {}
        with gzip.open(gff, "rt") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if line.startswith("#") or len(f) < 9 or f[2] not in ("transcript", "exon", "CDS", "stop_codon"):
                    continue
                contig = normalize_contig(f[0])
                if contig not in LENGTHS:
                    continue
                attrs = dict(kv.split("=", 1) for kv in f[8].split(";") if "=" in kv)
                if f[2] == "transcript":
                    if "MANE_Select" in attrs.get("tag", "").split(","):
                        transcripts[attrs["ID"]] = Transcript(attrs["ID"], attrs.get("gene_name", ""),
                                                              attrs.get("protein_id", ""), contig, f[6],
                                                              int(f[3]), int(f[4]))
                elif (t := transcripts.get(attrs.get("Parent"))) is not None:
                    (t.exons if f[2] == "exon" else t.coding).append((int(f[3]), int(f[4])))
        bins = defaultdict(list)
        for t in transcripts.values():
            t.exons, t.coding = _merge(t.exons), _merge(t.coding)
            if t.coding:
                for b in range(t.start // BIN, t.end // BIN + 1):
                    bins[(t.contig, b)].append(t)
        proteins, name = {}, None
        with gzip.open(faa, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    name = line[1:].split()[0]
                    proteins[name] = []
                else:
                    proteins[name].append(line.strip())
        self._proteins = {k: "".join(v) for k, v in proteins.items()}
        self._bins = dict(bins)
