import csv
import json

import pytest

from varfm import memory, runner
from varfm.memory import GREEN, MemoryGuard, limits_for
from varfm.models.base import Item, Model

from .conftest import CONTIG


class Fake(Model):
    """A model that scores a sequence by its GC count, and runs out of memory above `fits` sequences."""

    name = "fake"
    columns = ("gc",)
    fits = 3
    calls = []

    def fingerprint(self):
        return {"model": "fake"}

    def prepare(self, variant, annotation, reference):
        return [Item(variant.key, variant.key, reference.fetch(variant.chrom, variant.pos - 1, variant.pos + 9))]

    def load(self, device, guard):
        pass

    def batch_sizes(self, guard):
        return 8, 8

    def run(self, items):
        Fake.calls.append(len(items))
        if len(items) > self.fits:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [{"gc": float(sum(c in "GC" for c in it.seq))} for it in items]

    def combine(self, key, values):
        return values[key]


def test_extract_end_to_end(toy, tmp_path, monkeypatch):
    monkeypatch.setitem(runner.MODELS, "fake", Fake)
    Fake.calls = []
    variants = [f"{CONTIG}-{p}-{toy[p - 1]}-{'A' if toy[p - 1] != 'A' else 'C'}" for p in range(300, 310)]
    variants.append(f"{CONTIG}-124-T-C")  # wrong REF: the genome has G there
    table = runner.extract(variants, models=["fake"], out_dir=tmp_path / "out", device="cpu")
    rows = list(csv.DictReader(open(table), delimiter="\t"))
    assert len(rows) == 11
    assert rows[-1]["fake_status"] == "not_applicable" and rows[-1]["fake_reason"] == "ref_mismatch"
    ok = [r for r in rows if r["fake_status"] == "ok"]
    assert len(ok) == 10 and all(float(r["fake_gc"]) >= 0 for r in ok)
    assert Fake.calls[:2] == [8, 4] and max(Fake.calls[2:]) <= 3  # halved after each out-of-memory error
    run = json.loads((tmp_path / "out" / "run.json").read_text())
    assert run["models"]["fake"]["out_of_memory"] == 2
    assert run["models"]["fake"]["status"] == {"ok": 10, "not_applicable": 1}
    Fake.calls = []
    runner.extract(variants, models=["fake"], out_dir=tmp_path / "out", device="cpu")  # resumes: nothing left to do
    assert Fake.calls == []


def test_checkpoint_survives_a_cut_line_and_checks_its_settings(tmp_path):
    path = tmp_path / "m.jsonl"
    cp = runner._Checkpoint(path, {"model": "a"})
    cp.add([{"id": "x", "status": "failed", "reason": "boom"}, {"id": "x", "status": "ok", "values": {"v": 1}}])
    with open(path, "a") as fh:
        fh.write('{"id": "y", "sta')  # a kill in the middle of a line
    assert runner._read_checkpoint(path, {"model": "a"}) == {"x": {"id": "x", "status": "ok", "values": {"v": 1}}}
    with pytest.raises(ValueError, match="other settings"):
        runner._read_checkpoint(path, {"model": "b"})


def test_equal_length_batches():
    class Guard:
        critical, available_gb = False, 100.0

        def state(self):
            return GREEN

    class Lengths(Fake):
        equal_lengths, fits = True, 99

    items = [Item(f"v{i}", f"v{i}", "A" * n) for i, n in enumerate((5, 5, 5, 4, 4, 3))]
    Fake.calls = []
    runner._batches(Lengths(), items, Guard(), type("C", (), {"add": lambda self, rows: None})())
    assert Fake.calls == [3, 2, 1]


def test_batch_steps():
    assert [runner._shrink(s, 2) for s in (16, 6, 2, 1)] == [8, 2, 1, 1]
    assert [runner._shrink(s, 1) for s in (8, 3, 1)] == [4, 1, 1]
    assert [runner._align(s, 2) for s in (7, 2, 1)] == [6, 2, 1]


def test_memory_states(tmp_path):
    meminfo = tmp_path / "meminfo"

    def write(available_gb):
        kib = int(available_gb * 1e9 / 1024)
        meminfo.write_text(f"MemTotal: {int(130.6e9 / 1024)} kB\nMemAvailable: {kib} kB\nSwapTotal: 0 kB\n"
                           "SwapFree: 0 kB\n")

    write(100)
    guard = MemoryGuard("cpu", meminfo=str(meminfo))
    guard.limits = limits_for(130.6, shared=True)  # the GB10 thresholds
    assert (guard.limits.reserve_gb, guard.limits.yellow_gb, guard.limits.critical_gb) == pytest.approx((24, 36, 8))
    states = []
    for gb in (100, 30, 20, 5):
        write(gb)
        states.append(guard.state())
    assert states == [GREEN, memory.YELLOW, memory.RED, memory.CRITICAL]
    write(40)
    with pytest.raises(MemoryError):
        guard.require(42, "loading evo2")
    assert memory.is_oom(RuntimeError("CUDA out of memory")) and not memory.is_oom(ValueError("x"))
