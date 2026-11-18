"""Core data model: Prism-style wide tables, analyses and the project container.

Analysis and plotting code only reads these objects; the UI/application layer mutates them
and calls ``Table.bump()`` / ``Analysis.bump_params()`` after every edit.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TABLE_KINDS = ("XY", "Column", "Grouped", "Contingency", "Survival", "Nested", "MultipleVariables")
# Kinds whose data sets always hold exactly one subcolumn of raw values.
ONE_SUBCOLUMN_KINDS = ("Contingency", "Survival", "MultipleVariables")

# Subcolumn layout of every data set. "replicates": n raw values per row.
Y_FORMATS: dict[str, tuple[str, ...] | None] = {
    "replicates": None,
    "mean_sd": ("mean", "sd"),
    "mean_sem": ("mean", "sem"),
    "mean_sd_n": ("mean", "sd", "n"),
    "mean_sem_n": ("mean", "sem", "n"),
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class DataSet:
    """One Y data set (a Prism data-set column). values[row, sub]; NaN = blank cell.

    Excluded cells keep their value and are flagged in ``excluded`` (same shape).
    ``sub_titles``: optional subcolumn titles (Nested: one per subject/experiment).
    ``categories``: MultipleVariables only — the data set is categorical; values are integer codes
    0..len-1 into these labels (NaN = blank).
    """
    id: str
    title: str
    values: np.ndarray
    excluded: np.ndarray | None = None
    sub_titles: list[str] = field(default_factory=list)
    categories: list[str] | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        if self.excluded is None:
            self.excluded = np.zeros(self.values.shape, dtype=bool)
        self.excluded = np.asarray(self.excluded, dtype=bool)


@dataclass
class Table:
    """Prism-style wide table.

    XY:      rows = X values (``x``), subcolumns = replicates (or mean/error fields).
    Column:  one data set per group, rows = individual values, one subcolumn.
             Row index pairs values across data sets (paired / repeated designs).
    Grouped: rows = row groups (``row_titles``), data sets = column groups,
             subcolumns = replicates (or mean/error fields). Repeated measures: a subcolumn is one subject
             followed across rows (the analysis option says so; the layout is the same).
    Contingency: rows = groups/exposures (``row_titles``), data sets = outcome categories, one subcolumn
             of non-negative integer counts.
    Survival: ``x`` = elapsed time per subject row; data sets = groups, one subcolumn holding 1 (event /
             death) or 0 (censored) in the column of the subject's group, blank elsewhere (Prism layout).
    Nested:  data sets = treatment groups, subcolumns = subjects/independent experiments (``sub_titles``),
             rows = technical replicates within a subject.
    MultipleVariables: data sets = variables, rows = observations, one subcolumn; categorical variables
             carry ``categories`` (values are codes). ``row_titles`` optional observation labels.
    """
    id: str
    title: str
    kind: str
    y_format: str = "replicates"
    datasets: list[DataSet] = field(default_factory=list)
    x: np.ndarray | None = None
    x_title: str = "X"
    row_titles: list[str] = field(default_factory=list)
    revision: int = 0
    source: str | None = None  # analysis id when this table is an analysis output

    def __post_init__(self):
        if self.kind not in TABLE_KINDS:
            raise ValueError(f"unknown table kind {self.kind!r}")
        if self.y_format not in Y_FORMATS:
            raise ValueError(f"unknown y_format {self.y_format!r}")
        if self.x is not None:
            self.x = np.asarray(self.x, dtype=float).ravel()
        self.validate()

    @property
    def n_rows(self) -> int:
        if self.datasets:
            return self.datasets[0].values.shape[0]
        return 0 if self.x is None else len(self.x)

    def validate(self) -> None:
        fields = Y_FORMATS[self.y_format]
        for ds in self.datasets:
            if ds.values.shape != ds.excluded.shape:
                raise ValueError(f"{ds.id}: excluded mask shape mismatch")
            if ds.values.shape[0] != self.n_rows:
                raise ValueError(f"{ds.id}: row count {ds.values.shape[0]} != {self.n_rows}")
            if fields is not None and ds.values.shape[1] != len(fields):
                raise ValueError(f"{ds.id}: {self.y_format} needs {len(fields)} subcolumns")
        if self.kind == "XY" and (self.x is None or len(self.x) != self.n_rows):
            raise ValueError("XY table needs one X value per row")
        if self.kind == "Column" and self.y_format == "replicates":
            if any(ds.values.shape[1] != 1 for ds in self.datasets):
                raise ValueError("Column replicate data sets have exactly one subcolumn")
        if self.kind in ONE_SUBCOLUMN_KINDS + ("Nested",) and self.y_format != "replicates":
            raise ValueError(f"{self.kind} tables hold raw values (y_format 'replicates')")
        if self.kind in ONE_SUBCOLUMN_KINDS and any(ds.values.shape[1] != 1 for ds in self.datasets):
            raise ValueError(f"{self.kind} data sets have exactly one subcolumn")
        if self.kind == "Survival" and (self.x is None or len(self.x) != self.n_rows):
            raise ValueError("Survival table needs one time (X) value per row")
        for ds in self.datasets:
            v = ds.values[~np.isnan(ds.values)]
            if self.kind == "Contingency" and (np.any(v < 0) or np.any(v != np.round(v))):
                raise ValueError(f"{ds.title}: contingency counts must be non-negative integers")
            if self.kind == "Survival" and not np.all(np.isin(v, (0, 1))):
                raise ValueError(f"{ds.title}: survival codes must be 1 (event) or 0 (censored)")
            if ds.categories is not None:
                if self.kind != "MultipleVariables":
                    raise ValueError(f"{ds.title}: categorical data sets exist only in MultipleVariables tables")
                if np.any(v < 0) or np.any(v >= len(ds.categories)) or np.any(v != np.round(v)):
                    raise ValueError(f"{ds.title}: category codes out of range")
            if ds.sub_titles and len(ds.sub_titles) != ds.values.shape[1]:
                raise ValueError(f"{ds.title}: {len(ds.sub_titles)} subcolumn titles for {ds.values.shape[1]} subcolumns")

    def bump(self) -> None:
        """Call after any edit to data, exclusions or structure."""
        self.revision += 1

    def dataset(self, ds_id: str) -> DataSet:
        for ds in self.datasets:
            if ds.id == ds_id:
                return ds
        raise KeyError(ds_id)

    def points(self, ds_id: str, include_excluded: bool = False) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Raw replicate points of one data set, blanks (and exclusions) dropped.

        Returns (x, y, row): x is None for Column/Grouped tables; row is the source row index.
        Requires y_format == "replicates".
        """
        if self.y_format != "replicates":
            raise ValueError("raw points need a replicates table; use summary()")
        ds = self.dataset(ds_id)
        rows = np.broadcast_to(np.arange(self.n_rows)[:, None], ds.values.shape)
        keep = ~np.isnan(ds.values)
        if not include_excluded:
            keep &= ~ds.excluded
        if self.x is not None:
            xs = np.broadcast_to(self.x[:, None], ds.values.shape)
            keep &= ~np.isnan(xs)
            return xs[keep], ds.values[keep], rows[keep]
        return None, ds.values[keep], rows[keep]

    def summary(self, ds_id: str) -> dict[str, np.ndarray]:
        """Per-row mean, sd, sem and n (n = NaN when not known), from either layout."""
        ds = self.dataset(ds_id)
        fields = Y_FORMATS[self.y_format]
        if fields is None:
            v = np.where(ds.excluded, np.nan, ds.values)
            n = np.sum(~np.isnan(v), axis=1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, np.nansum(v, axis=1) / np.where(n > 0, n, 1), np.nan)
                sd = np.array([np.std(r[~np.isnan(r)], ddof=1) if c > 1 else np.nan for r, c in zip(v, n)])
                sem = sd / np.sqrt(n)
            return {"mean": mean, "sd": sd, "sem": sem, "n": n}
        cols = {f: np.where(ds.excluded[:, i], np.nan, ds.values[:, i]) for i, f in enumerate(fields)}
        n = cols.get("n", np.full(self.n_rows, np.nan))
        if "sd" in cols:
            sd, sem = cols["sd"], cols["sd"] / np.sqrt(n)
        else:
            sem, sd = cols["sem"], cols["sem"] * np.sqrt(n)
        return {"mean": cols["mean"], "sd": sd, "sem": sem, "n": n}


@dataclass
class Analysis:
    """One analysis (or transform) bound to a table.

    A result is current only if it was computed from the present table revision and params
    revision. Background jobs record ``key()`` at submit time and the app adopts a finished
    result only if ``key()`` is unchanged on completion.
    """
    id: str
    title: str
    kind: str
    table_id: str
    dataset_ids: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    params_revision: int = 0
    result: dict | None = None
    result_key: tuple[int, int] | None = None
    status: str = "not_run"  # not_run | running | ok | warning | failed
    output_table_id: str | None = None  # transforms only

    def key(self, table: Table) -> tuple[int, int]:
        return (table.revision, self.params_revision)

    def is_current(self, table: Table) -> bool:
        return self.result is not None and self.result_key == self.key(table)

    def bump_params(self) -> None:
        self.params_revision += 1


@dataclass
class Project:
    title: str = "Untitled"
    tables: dict[str, Table] = field(default_factory=dict)
    analyses: dict[str, Analysis] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)   # id -> plot.PlotSpec
    layouts: dict[str, Any] = field(default_factory=dict)  # id -> plot.LayoutSpec
    schema_version: int = 1
