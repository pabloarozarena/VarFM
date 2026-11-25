"""Command line: varfm extract variants.vcf [-m esm2] [-m evo2] [-o varfm_out]"""

import argparse

from .models import MODELS
from .runner import extract


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="varfm", description="Features of genetic variants from foundation models")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("extract", help="extract features for the variants of a GRCh38 VCF")
    cmd.add_argument("vcf", help="VCF file (plain or gzipped), GRCh38")
    cmd.add_argument("-m", "--model", action="append", choices=sorted(MODELS),
                     help="model to run; repeat it for more (default: all)")
    cmd.add_argument("-o", "--out", default="varfm_out", help="output folder (default: varfm_out); "
                     "an interrupted run continues when started again with the same folder")
    cmd.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if available)")
    cmd.add_argument("--limit", type=int, default=None, help="only the first N variants (a quick trial)")
    args = parser.parse_args(argv)
    extract(args.vcf, models=args.model or list(MODELS), out_dir=args.out, device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
