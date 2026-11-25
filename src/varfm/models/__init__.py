"""The models VarFM runs, by name."""

from .esm2 import ESM2
from .evo2 import Evo2

MODELS = {"esm2": ESM2, "evo2": Evo2}
