"""What the runner needs from a model.

A model turns each variant into one or more sequences (on the CPU, without loading anything), runs batches of those
sequences through its network, and combines the results of a variant's sequences into a few output columns.
"""

from dataclasses import dataclass, field


@dataclass
class Item:
    """One sequence to run: `id` is unique in the run, `key` is the variant it belongs to."""

    id: str
    key: str
    seq: str
    extra: dict = field(default_factory=dict)


class Unsupported(RuntimeError):
    """The model cannot run on this device (e.g. Evo 2 on a GPU without bfloat16): its variants are skipped."""


class Model:
    name = ""
    columns: tuple = ()   # output columns, prefixed with the model name in the table
    batch_step = 1        # batches grow and shrink by this many items (2 keeps a variant's REF and ALT together)
    equal_lengths = False  # True: a batch only holds sequences of the same length (no padding)

    def load_peak_gb(self, shared: bool) -> float:
        """System memory taken while loading, at the peak; `shared`: the GPU uses the system memory."""
        return 0.0

    def fingerprint(self) -> dict:
        """What determines the results: a checkpoint is reused only with the same fingerprint."""
        raise NotImplementedError

    def prepare(self, variant, annotation, reference):
        """The items of a variant, or a snake_case reason why the model does not apply to it."""
        raise NotImplementedError

    def load(self, device: str, guard) -> None:
        raise NotImplementedError

    def batch_sizes(self, guard) -> tuple[int, int]:
        """First and largest batch, in items, given the memory left once the model is loaded."""
        return 1, 1

    def run(self, items: list) -> list[dict]:
        """One dict of values per item, in order."""
        raise NotImplementedError

    def combine(self, key: str, values: dict) -> dict:
        """The output columns of a variant, from {item id: values} of all its items."""
        raise NotImplementedError

    def unload(self) -> None:
        pass
