"""VarFM: features of genetic variants from biological foundation models.

Pilot: two zero-shot log-likelihood features for the variants of a GRCh38 VCF.

    esm2   ESM2 650M: log-probability of the reference and the mutant residue at the masked site (masked marginal),
           for missense SNVs of MANE Select proteins
    evo2   Evo 2 7B: log-likelihood of the REF and ALT DNA windows (8,001 bp around the variant), summed from the
           first base where the two differ

    import varfm
    table = varfm.extract("variants.vcf", models=["esm2", "evo2"], out_dir="varfm_out")   # -> varfm_out/features.tsv
"""

from importlib.resources import files

__version__ = "0.1.0"


def example_vcf() -> str:
    """Path of the bundled example: the first 100 ClinVar GRCh38 records of chromosome 1."""
    return str(files("varfm") / "data" / "100_varianti_umane.vcf")


from .runner import extract  # noqa: E402

__all__ = ["extract", "example_vcf", "__version__"]
