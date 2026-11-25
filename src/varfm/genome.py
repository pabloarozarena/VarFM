"""The GRCh38 reference sequence, downloaded from Ensembl one chromosome at a time, when a variant first needs it."""

import gzip
import os
from pathlib import Path

from .assets import cache_dir, download

ENSEMBL_RELEASE = 115  # GRCh38 chromosomes are the same in every release: the release only pins the source
URL = ("https://ftp.ensembl.org/pub/release-{release}/fasta/homo_sapiens/dna/"
       "Homo_sapiens.GRCh38.dna.chromosome.{contig}.fa.gz")
LENGTHS = {  # GRCh38 primary assembly
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259, "6": 170805979,
    "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345, "17": 83257441, "18": 80373285,
    "19": 58617616, "20": 64444167, "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415, "MT": 16569,
}


def normalize_contig(name: str) -> str:
    """'chr1' and '1' -> '1'; 'chrX' -> 'X'; 'chrM', 'M' and 'MT' -> 'MT'. Other names (alt contigs, patches) stay."""
    contig = name[3:] if name[:3].lower() == "chr" else name
    if contig.upper() in ("X", "Y", "M", "MT"):
        contig = contig.upper()
    return "MT" if contig == "M" else contig


class Reference:
    """Random access to GRCh38.

    Each chromosome is stored as a single line of uppercase bases, without header or newlines
    (<cache>/GRCh38/<contig>.seq), so that a window is one seek and one read.
    """

    def __init__(self, root=None):
        self.root = Path(root) if root else cache_dir() / "GRCh38"
        self._files = {}

    def fetch(self, contig: str, start: int, end: int) -> str:
        """Bases [start, end) of a chromosome, 0-based. ValueError if the window leaves the chromosome."""
        if contig not in LENGTHS:
            raise KeyError(f"unknown GRCh38 contig {contig!r}")
        if not 0 <= start <= end <= LENGTHS[contig]:
            raise ValueError("window_outside_contig")
        fh = self._files.get(contig)
        if fh is None:
            fh = self._files[contig] = open(self._sequence(contig), "rb")
        fh.seek(start)
        return fh.read(end - start).decode("ascii")

    def _sequence(self, contig: str) -> Path:
        path = self.root / f"{contig}.seq"
        if path.exists():
            return path
        gz = download(URL.format(release=ENSEMBL_RELEASE, contig=contig), self.root / f"{contig}.fa.gz")
        print(f"unpacking chromosome {contig}", flush=True)
        tmp = path.with_name(path.name + ".part")
        with gzip.open(gz, "rt") as fasta, open(tmp, "w") as out:
            for line in fasta:
                if not line.startswith(">"):
                    out.write(line.strip().upper())
        if tmp.stat().st_size != LENGTHS[contig]:
            tmp.unlink()
            raise RuntimeError(f"{gz}: chromosome {contig} does not have its GRCh38 length")
        os.replace(tmp, path)
        gz.unlink()
        return path
