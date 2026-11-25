import gzip

import pytest

import varfm
from varfm.annotate import Mane, translate
from varfm.genome import Reference, normalize_contig
from varfm.models.esm2 import ESM2, window_start
from varfm.models.evo2 import OFFSET, WINDOW, Evo2, build_pair
from varfm.vcf import Variant, parse_variant, read_vcf

from .conftest import CONTIG, MINUS_CDS, PLUS_CDS


def test_example_vcf():
    variants = read_vcf(varfm.example_vcf())
    assert len(variants) == 100
    assert variants[0].key == "GRCh38:1:66926:AG:A" and variants[0].id == "3385321"


def test_vcf_split_gzip_and_assembly(tmp_path):
    path = tmp_path / "x.vcf.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\nchr1\t10\trs1\tA\tC,G\nchr1\t10\trs1\tA\tC\n")
    assert [v.key for v in read_vcf(path)] == ["GRCh38:1:10:A:C", "GRCh38:1:10:A:G"]
    old = tmp_path / "old.vcf"
    old.write_text("##reference=GRCh37\n#CHROM\tPOS\tID\tREF\tALT\n1\t10\t.\tA\tC\n")
    with pytest.raises(ValueError, match="GRCh38"):
        read_vcf(old)


def test_parse_variant_and_contigs():
    assert parse_variant("chr1-930204-G-A") == Variant("1", 930204, "G", "A")
    assert parse_variant("X:10:a:t").key == "GRCh38:X:10:A:T"
    assert [normalize_contig(c) for c in ("chr1", "1", "chrx", "chrM", "M", "MT")] == ["1", "1", "X", "MT", "MT", "MT"]


def test_reference_fetch(toy):
    ref = Reference()
    assert ref.fetch(CONTIG, 120, 123) == "ATG"
    with pytest.raises(ValueError):
        ref.fetch(CONTIG, 590, 610)


@pytest.mark.parametrize("pos, ref, alt, consequence, change", [
    (124, "G", "C", "missense", "p.A2P"),
    (126, "T", "C", "synonymous", "p.A2A"),
    (132, "G", "A", "stop_gained", "p.W4*"),
    (121, "A", "G", "start_lost", "p.M1V"),
    (201, "G", "A", "missense", "p.C14Y"),     # codon split by the intron
    (218, "T", "C", "stop_lost", "p.*20Q"),
    (517, "A", "G", "missense", "p.F2L"),      # - strand
    (450, "C", "T", "stop_gained", "p.W7*"),   # - strand, codon split by the intron
    (124, "GC", "G", "frameshift", ""),
    (124, "GCTA", "G", "inframe_indel", ""),
    (110, None, None, "5_prime_utr", ""),
    (230, None, None, "3_prime_utr", ""),
    (540, None, None, "5_prime_utr", ""),      # - strand
    (410, None, None, "3_prime_utr", ""),      # - strand
    (180, None, None, "intron", ""),
    (161, None, None, "splice_site", ""),
    (200, None, None, "splice_site", ""),
    (70, None, None, "intergenic", ""),        # only a MANE Plus Clinical transcript there
])
def test_consequences(toy, pos, ref, alt, consequence, change):
    if ref is None:
        ref = toy[pos - 1]
        alt = {"A": "C", "C": "G", "G": "T", "T": "A"}[ref]
    assert toy[pos - 1:pos - 1 + len(ref)] == ref
    ann = Mane(Reference()).annotate(Variant(CONTIG, pos, ref, alt))
    assert (ann.consequence, ann.protein_change) == (consequence, change)
    if change:
        assert ann.protein_ok and ann.protein in (translate(PLUS_CDS)[:-1], translate(MINUS_CDS)[:-1])


def test_evo2_pairs():
    context = "AC" * 4000 + "G"  # 8,001 bases
    x_ref, x_alt, a = build_pair(context, 4000, "A", "T")
    assert a == 4000 and len(x_ref) == len(x_alt) == 8001 and x_alt[4000] == "T"
    x_ref, x_alt, a = build_pair(context, 4000, "ACA", "A")  # deletion: REF trimmed to the ALT length
    assert len(x_ref) == len(x_alt) == 7999 and x_ref[:a] == x_alt[:a] and x_ref[a] != x_alt[a]
    x_ref, x_alt, a = build_pair(context, 4000, "A", "AT")  # insertion: ALT trimmed on the right
    assert len(x_ref) == len(x_alt) == 8001 and a == 4001
    for ref, alt, reason in (("G", "T", "ref_mismatch"), ("A", "N", "non_acgt_sequence")):
        with pytest.raises(ValueError, match=reason):
            build_pair(context, 4000, ref, alt)


def test_evo2_prepare(toy, monkeypatch):
    from varfm import genome
    long = toy * 20  # 12,000 bases, long enough for a window
    (Reference().root / f"{CONTIG}.seq").write_text(long)
    monkeypatch.setitem(genome.LENGTHS, CONTIG, len(long))
    v = Variant(CONTIG, 5000, long[4999], "A" if long[4999] != "A" else "C")
    ref_item, alt_item = Evo2().prepare(v, None, Reference())
    assert len(ref_item.seq) == len(alt_item.seq) == WINDOW and ref_item.extra["anchor"] == OFFSET
    assert Evo2().prepare(Variant(CONTIG, 100, long[99], "A"), None, Reference()) == "window_outside_contig"
    assert Evo2().combine(v.key, {f"{v.key}|ref": {"ll": -10.0}, f"{v.key}|alt": {"ll": -12.5}})["delta"] == -2.5


def test_esm2_inputs(toy):
    assert window_start(5, 300) == 0 and window_start(600, 2000) == 89 and window_start(1990, 2000) == 978
    mane = Mane(Reference())
    missense = Variant(CONTIG, 124, "G", "C")
    (item,) = ESM2().prepare(missense, mane.annotate(missense), None)
    assert item.seq == translate(PLUS_CDS)[:-1] and item.extra == {"site": 1, "wt": "A", "mut": "P"}
    synonymous = Variant(CONTIG, 126, "T", "C")
    assert ESM2().prepare(synonymous, mane.annotate(synonymous), None) == "protein_unchanged"
    ann = mane.annotate(missense)
    ann.protein_ok = False
    assert ESM2().prepare(missense, ann, None) == "fasta_mismatch"
