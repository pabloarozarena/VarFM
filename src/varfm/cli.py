"""Command line: varfm variants.vcf --models esm2 evo2 --out varfm_out"""

import argparse

from . import __version__
from .models import MODELS
from .runner import extract


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="varfm", description="Foundation-model features for the variants of a GRCh38 VCF.",
        usage="varfm VCF [--models esm2 evo2] [--out DIR] [--limit N] [--device DEVICE]",
        epilog="example: varfm variants.vcf --models esm2 evo2 --out varfm_out")
    parser.add_argument("vcf", help="VCF file (plain or gzipped), GRCh38")
    parser.add_argument("-m", "--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="models to run (default: all)")
    parser.add_argument("-o", "--out", default="varfm_out", help="output folder (default: varfm_out); "
                        "an interrupted run continues when started again with the same folder")
    parser.add_argument("--limit", type=int, default=None, help="only the first N variants (a quick trial)")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available)")
    parser.add_argument("--version", action="version", version=f"varfm {__version__}")
    args = parser.parse_args(argv)
    extract(args.vcf, models=list(dict.fromkeys(args.models)), out_dir=args.out, device=args.device,
            limit=args.limit)


if __name__ == "__main__":
    main()
