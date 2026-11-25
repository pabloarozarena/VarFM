"""The cache of downloaded reference files, and the one function that downloads them."""

import os
import shutil
import time
import urllib.request
from pathlib import Path


def cache_dir() -> Path:
    """Where VarFM keeps reference files: $VARFM_HOME, else ~/.cache/varfm."""
    return Path(os.environ.get("VARFM_HOME") or Path.home() / ".cache" / "varfm")


def download(url: str, dest, attempts: int = 3) -> Path:
    """Download url to dest, unless dest already exists.

    The file is written under a temporary name and renamed at the end, so an interrupted download never leaves a
    truncated file behind. Network errors are retried a few times (public FTP servers fail now and then).
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            os.replace(tmp, dest)
            return dest
        except OSError as e:  # urllib's URLError is an OSError
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise RuntimeError(f"could not download {url}: {e}") from e
            time.sleep(5 * attempt)
    return dest
