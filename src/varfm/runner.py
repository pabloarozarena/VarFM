"""The extraction: variants in, one table of features out.

    VCF ─► variants ─► REF checked on GRCh38 ─► MANE annotation
        ─► for each model: sequences ─► batches on the GPU ─► checkpoint <out_dir>/work/<model>.jsonl
        ─► <out_dir>/features.tsv (one row per variant) and <out_dir>/run.json

Models run one after the other, each loaded, run over all its sequences and unloaded, so only one is ever in memory.
Results are appended to the model's checkpoint after every batch: a run that stops (an error, a kill, the memory
guard) continues where it stopped when it is started again with the same out_dir. Batches start from the memory left
once the model is loaded, grow after calm batches and halve when memory runs low or a batch does not fit.
"""

import json
import os
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

from . import __version__
from .annotate import MANE_RELEASE, Annotation, Mane
from .genome import ENSEMBL_RELEASE, LENGTHS, Reference
from .memory import GREEN, RED, YELLOW, MemoryGuard, free_memory, gpu_lock, is_oom
from .models import MODELS
from .models.base import Unsupported
from .vcf import load_variants

BASE_COLUMNS = ("variant", "id", "chrom", "pos", "ref", "alt", "gene", "transcript", "consequence", "protein_change")
GROW_AFTER = 4     # calm batches before a batch grows
MAX_FAILURES = 20  # sequences failing in a row that stop the run: something is broken


def extract(source, models=("esm2", "evo2"), out_dir="varfm_out", device=None, limit=None) -> Path:
    """Compute the features of `models` for the variants of `source` (a VCF path, or a list such as
    ["1-930204-G-A"]), write them to <out_dir>/features.tsv and return its path."""
    started = time.time()
    models = list(models)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown models {unknown}: choose among {sorted(MODELS)}")
    out = Path(out_dir)
    (out / "work").mkdir(parents=True, exist_ok=True)
    variants = load_variants(source)[:limit]
    reference = Reference()
    mane = Mane(reference)
    checks = {v.key: check_variant(v, reference) for v in variants}
    annotations = {v.key: mane.annotate(v) if not checks[v.key] else Annotation(consequence="") for v in variants}
    unusable = Counter(reason for reason in checks.values() if reason)
    print(f"{len(variants)} variants" + (f", not usable: {dict(unusable)}" if unusable else ""), flush=True)
    device = device or _default_device()
    results, runs = {}, {}
    for name in models:
        results[name], runs[name] = _run_model(MODELS[name](), variants, checks, annotations, reference, device,
                                               out / "work")
    table = _write_table(out / "features.tsv", variants, annotations, results, models)
    for name in models:
        counts = Counter(status for status, _, _ in results[name].values())
        runs[name]["status"] = dict(counts)
        runs[name]["reasons"] = dict(Counter(r for s, r, _ in results[name].values() if s != "ok"))
        print(f"{name}: {dict(counts)}", flush=True)
    _write_json(out / "run.json", {
        "varfm": __version__, "created": datetime.now().isoformat(timespec="seconds"),
        "input": str(source) if isinstance(source, (str, Path)) else f"{len(variants)} variants",
        "variants": len(variants), "not_usable": dict(unusable), "device": device,
        "reference": f"GRCh38, Ensembl release {ENSEMBL_RELEASE}", "annotation": f"MANE v{MANE_RELEASE} (Ensembl)",
        "models": runs, "seconds": round(time.time() - started, 1)})
    print(f"features: {table}", flush=True)
    return table


def check_variant(v, reference) -> str:
    """'' if the variant can be computed, otherwise why not."""
    if v.chrom not in LENGTHS:
        return "unsupported_contig"
    if not v.ref or not v.alt or set(v.ref + v.alt) - set("ACGT"):
        return "unsupported_allele"
    if v.ref == v.alt:
        return "ref_equals_alt"
    try:
        genome = reference.fetch(v.chrom, v.pos - 1, v.pos - 1 + len(v.ref))
    except ValueError:
        return "outside_contig"
    return "" if genome == v.ref else "ref_mismatch"


def _run_model(model, variants, checks, annotations, reference, device, work: Path):
    """{variant key: (status, reason, values)} of one model, and what run.json records about it."""
    results, items = {}, []
    for v in variants:
        prepared = checks[v.key] or model.prepare(v, annotations[v.key], reference)
        if isinstance(prepared, str):
            results[v.key] = ("not_applicable", prepared, {})
        else:
            items.extend(prepared)
    path = work / f"{model.name}.jsonl"
    fingerprint = {"varfm": __version__, **model.fingerprint()}
    records = _read_checkpoint(path, fingerprint)
    todo = [it for it in items if records.get(it.id, {}).get("status") != "ok"]
    keys = {it.key for it in items}
    print(f"{model.name}: {len(keys)} variants to score ({len(items)} sequences), {len(results)} not applicable, "
          f"{len(todo)} sequences left to compute", flush=True)
    run = {"fingerprint": fingerprint}
    if todo:
        try:
            run.update(_compute(model, todo, device, path, fingerprint))
        except Unsupported as e:
            print(f"{model.name}: skipped: {e}", flush=True)
            run["skipped"] = str(e)
            results.update({key: ("not_applicable", "unsupported_device", {}) for key in keys})
            return results, run
        records = _read_checkpoint(path, fingerprint)
    by_variant = defaultdict(dict)
    for it in items:
        by_variant[it.key][it.id] = records.get(it.id)
    for key, recs in by_variant.items():
        bad = [r for r in recs.values() if r is None or r["status"] != "ok"]
        if bad:
            results[key] = ("failed", bad[0]["reason"] if bad[0] else "not_computed", {})
        else:
            results[key] = ("ok", "", model.combine(key, {i: r["values"] for i, r in recs.items()}))
    return results, run


def _compute(model, items, device, path, fingerprint) -> dict:
    """Load the model behind the memory guard, run its items in batches, unload it."""
    lock = gpu_lock() if device.startswith("cuda") else nullcontext()
    with MemoryGuard(device) as guard, lock:
        guard.require(model.load_peak_gb(guard.shared), f"loading {model.name}")
        guard.cap_gpu()
        started = time.time()
        model.load(device, guard)
        loaded = time.time() - started
        print(f"{model.name}: loaded in {loaded:.0f} s", flush=True)
        try:
            stats = _batches(model, items, guard, _Checkpoint(path, fingerprint))
        finally:
            model.unload()
            free_memory()
    return {"load_seconds": round(loaded, 1), **stats, "lowest_available_gb": round(guard.lowest_gb, 1)}


def _batches(model, items, guard, checkpoint) -> dict:
    """Run the items in batches, longest first; every finished batch goes to the checkpoint."""
    items = sorted(items, key=lambda it: (-len(it.seq), it.id))
    step = model.batch_step
    size, most = model.batch_sizes(guard)
    print(f"{model.name}: first batch {size} sequences, at most {most}", flush=True)
    done = calm = failures = batches = ooms = 0
    started = last_report = time.time()
    while done < len(items):
        if guard.critical:
            raise MemoryError("memory guard: memory critically low, stopped (run again to resume)")
        state = guard.state()
        if state == RED and not guard.wait():
            raise MemoryError(f"memory guard: memory stayed below the reserve ({guard.available_gb:.1f} GB "
                              "available), stopped (run again to resume)")
        if state in (YELLOW, RED):
            size, calm = _shrink(size, step), 0
            free_memory()
        batch = items[done:done + size]
        if model.equal_lengths:  # items are sorted by length: the batch stops where the length changes
            batch = [it for it in batch if len(it.seq) == len(batch[0].seq)]
        try:
            values, oom, reason = model.run(batch), False, ""
        except Exception as e:  # noqa: BLE001 - every error is handled below
            values, oom, reason = None, is_oom(e), _describe(e)
        if values is None:
            free_memory()  # outside the except clause: the traceback no longer holds the batch's tensors
            if oom and len(batch) > 1:
                ooms += 1
                most = _align(len(batch) - 1, step)
                size, calm = min(_shrink(size, step), most), 0
                print(f"{model.name}: out of memory with {len(batch)} sequences, now {size}", flush=True)
                continue
            rows = [_run_one(model, it) for it in batch] if len(batch) > 1 else [
                {"id": batch[0].id, "status": "failed", "reason": "oom_at_batch_1" if oom else reason}]
        else:
            rows = [{"id": it.id, "status": "ok", "values": v} for it, v in zip(batch, values)]
        checkpoint.add(rows)
        done += len(batch)
        failed = [r for r in rows if r["status"] != "ok"]
        failures = failures + len(failed) if failed else 0
        if failures >= MAX_FAILURES:
            raise RuntimeError(f"{model.name}: {failures} sequences failed in a row (last: {failed[-1]['reason']})")
        if not failed:
            batches += 1
            if guard.state() == GREEN:
                calm += 1
                if calm >= GROW_AFTER:
                    size, calm = min(most, (size // step + 1) * step), 0
        if time.time() - last_report > 30 or done == len(items):
            last_report = time.time()
            rate = done / max(last_report - started, 1e-9)
            print(f"{model.name}: {done}/{len(items)} sequences, {rate:.1f}/s, "
                  f"~{(len(items) - done) / max(rate, 1e-9) / 60:.0f} min left, batch {size}, "
                  f"{guard.available_gb:.0f} GB of memory available", flush=True)
    return {"batches": batches, "out_of_memory": ooms, "last_batch": size, "seconds": round(time.time() - started, 1)}


def _run_one(model, item) -> dict:
    try:
        return {"id": item.id, "status": "ok", "values": model.run([item])[0]}
    except Exception as e:  # noqa: BLE001
        reason = "oom_at_batch_1" if is_oom(e) else _describe(e)
    free_memory()
    return {"id": item.id, "status": "failed", "reason": reason}


def _shrink(size: int, step: int) -> int:
    half = size // 2
    return half - half % step if half >= step else 1


def _align(size: int, step: int) -> int:
    return size - size % step if size >= step else max(size, 1)


def _describe(e: BaseException) -> str:
    return "error: " + " ".join(f"{type(e).__name__}: {e}".split())[:200]


class _Checkpoint:
    """One model's results, a JSON line per sequence after a header with the fingerprint, synced after every batch."""

    def __init__(self, path: Path, fingerprint: dict):
        self.path = path
        if not path.exists():
            with open(path, "w") as fh:
                fh.write(json.dumps({"fingerprint": fingerprint}) + "\n")

    def add(self, rows: list) -> None:
        with open(self.path, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _read_checkpoint(path: Path, fingerprint: dict) -> dict:
    """{item id: its last record}; a later record (a retry) replaces an earlier one."""
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if not lines or json.loads(lines[0]).get("fingerprint") != fingerprint:
        raise ValueError(f"{path} holds results computed with other settings: use another out_dir, or delete it")
    records = {}
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # the last line of a run that was killed while writing
            continue
        records[row["id"]] = row
    return records


def _write_table(path: Path, variants, annotations, results, models) -> Path:
    columns = list(BASE_COLUMNS)
    for name in models:
        columns += [f"{name}_{c}" for c in MODELS[name].columns] + [f"{name}_status", f"{name}_reason"]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for v in variants:
            a = annotations[v.key]
            row = [v.key, v.id, v.chrom, str(v.pos), v.ref, v.alt, a.gene, a.transcript, a.consequence,
                   a.protein_change]
            for name in models:
                status, reason, values = results[name][v.key]
                row += [_number(values.get(c)) for c in MODELS[name].columns] + [status, reason]
            fh.write("\t".join(row) + "\n")
    os.replace(tmp, path)
    return path


def _number(x) -> str:
    return "" if x is None else f"{x:.6f}"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
