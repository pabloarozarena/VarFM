"""A toy genome and toy MANE files, so that the tests run offline, on the CPU, in seconds.

Contig T1 (600 random bases) holds two designed genes, positions 1-based:

    PLUS  (+)  exons 101-160, 201-260   coding 121-160, 201-217, stop codon 218-220   protein MAKWQEFGLQDPSCNRIGE
    MINUS (-)  exons 401-450, 501-560   coding 520-501, 450-432, stop codon 431-429   protein MFGKHEWLNCGSR
"""

import gzip
import random

import pytest

from varfm import annotate, genome

CONTIG, LENGTH = "T1", 600
PLUS_CDS = "ATGGCTAAATGGCAGGAATTTGGTCTTCAAGATCCGTCATGTAACCGTATTGGAGAATAA"  # 60 bases, stop included
MINUS_CDS = "ATGTTTGGCAAACATGAGTGGCTGAATTGCGGTTCTCGATGA"  # 42 bases, 5'->3' on the - strand


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def build_chromosome() -> str:
    rng = random.Random(7)
    seq = [rng.choice("ACGT") for _ in range(LENGTH)]

    def put(start, text):
        seq[start - 1:start - 1 + len(text)] = list(text)

    put(121, PLUS_CDS[:40])
    put(201, PLUS_CDS[40:])
    minus = _revcomp(MINUS_CDS)  # + strand, genomic order: 429-450 then 501-520
    put(429, minus[:22])
    put(501, minus[22:])
    return "".join(seq)


GFF_ROWS = [
    ("transcript", 101, 260, "+", "ID=ENST01;gene_name=PLUS;tag=MANE_Select;protein_id=ENSP01"),
    ("exon", 101, 160, "+", "Parent=ENST01"), ("exon", 201, 260, "+", "Parent=ENST01"),
    ("CDS", 121, 160, "+", "Parent=ENST01"), ("CDS", 201, 217, "+", "Parent=ENST01"),
    ("stop_codon", 218, 220, "+", "Parent=ENST01"),
    ("transcript", 401, 560, "-", "ID=ENST02;gene_name=MINUS;tag=MANE_Select;protein_id=ENSP02"),
    ("exon", 501, 560, "-", "Parent=ENST02"), ("exon", 401, 450, "-", "Parent=ENST02"),
    ("CDS", 501, 520, "-", "Parent=ENST02"), ("CDS", 432, 450, "-", "Parent=ENST02"),
    ("stop_codon", 429, 431, "-", "Parent=ENST02"),
    ("transcript", 50, 90, "+", "ID=ENST03;gene_name=OTHER;tag=MANE_Plus_Clinical;protein_id=ENSP03"),
]


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """VARFM_HOME holding the toy genome and MANE files; returns the chromosome sequence."""
    monkeypatch.setenv("VARFM_HOME", str(tmp_path))
    monkeypatch.setitem(genome.LENGTHS, CONTIG, LENGTH)
    chrom = build_chromosome()
    (tmp_path / "GRCh38").mkdir()
    (tmp_path / "GRCh38" / f"{CONTIG}.seq").write_text(chrom)
    (tmp_path / "MANE").mkdir()
    v = annotate.MANE_RELEASE
    with gzip.open(tmp_path / "MANE" / annotate.GFF.format(v=v), "wt") as fh:
        fh.write("##gff-version 3\n")
        for kind, start, end, strand, attrs in GFF_ROWS:
            fh.write(f"chr{CONTIG}\ttest\t{kind}\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n")
    with gzip.open(tmp_path / "MANE" / annotate.PROTEINS.format(v=v), "wt") as fh:
        fh.write(f">ENSP01 pep\n{annotate.translate(PLUS_CDS)[:-1]}\n>ENSP02 pep\n{annotate.translate(MINUS_CDS)[:-1]}\n")
    return chrom
