"""Analysis registry. Pure computation: no Qt, no file I/O, no project mutation.

Every analysis module exposes ``KINDS: dict[str, AnalysisKind]``. ``run`` is what the
background worker calls; it must stay importable in a spawned process.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

PARAM_TYPES = ("choice", "float", "float?", "int", "bool", "str", "dataset", "datasets", "constraints")


@dataclass(frozen=True)
class Param:
    """One user option. "float?" allows None. "dataset(s)" pick data-set ids of the input table.
    "constraints" is {param_name: {"fixed": v} | {"min": a, "max": b}} for the chosen model."""
    name: str
    label: str
    type: str
    default: Any = None
    choices: tuple = ()  # for "choice": tuple of (value, label)

    def __post_init__(self):
        if self.type not in PARAM_TYPES:
            raise ValueError(f"{self.name}: unknown param type {self.type!r}")


@dataclass(frozen=True)
class AnalysisKind:
    id: str
    label: str
    table_kinds: tuple[str, ...]
    func: Callable  # func(table, dataset_ids, params) -> dict (see DESIGN.md "Result contract")
    params: tuple[Param, ...] = ()
    output: str = "result"  # "result" or "table" (transforms return {"output_table": Table, ...})

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


MODULES = ("transforms", "fit", "stats", "designs", "categorical", "regression", "multivariate")


def all_kinds() -> dict[str, AnalysisKind]:
    import importlib
    import importlib.util
    kinds: dict[str, AnalysisKind] = {}
    for name in MODULES:
        if importlib.util.find_spec(f"{__name__}.{name}") is None:  # ponytail: module not written yet
            continue
        mod = importlib.import_module(f"{__name__}.{name}")
        for k, v in mod.KINDS.items():
            if k in kinds:
                raise ValueError(f"duplicate analysis kind {k}")
            kinds[k] = v
    return kinds


def run(kind: str, table, dataset_ids: list[str], params: dict) -> dict:
    spec = all_kinds()[kind]
    if table.kind not in spec.table_kinds:
        raise ValueError(f"{spec.label} needs a {'/'.join(spec.table_kinds)} table, got {table.kind}")
    return spec.func(table, dataset_ids, {**spec.defaults(), **params})
