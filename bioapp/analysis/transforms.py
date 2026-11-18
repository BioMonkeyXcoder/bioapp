"""Table-to-table transforms: Normalize and Remove baseline (minimal).

Normalize follows Prism's manual: 0% / 100% are defined from the per-row mean of the replicates
(the mean column of a mean/error layout), every replicate cell is then transformed with
(Y - zero) / (hundred - zero); SD and SEM columns are divided by |hundred - zero| (baseline
subtraction leaves them unchanged) and ``n`` is copied. X values are copied unchanged.
"""
from __future__ import annotations

import numpy as np

from ..data import DataSet, Table, Y_FORMATS, new_id
from . import AnalysisKind, Param

_ZERO = (("min", "Smallest value in each data set"), ("first", "Value in the first row"), ("value", "A value I enter"))
_HUNDRED = (("max", "Largest value in each data set"), ("last", "Value in the last row"), ("value", "A value I enter"))
_BASE = (("first", "First row"), ("last", "Last row"), ("first_last", "Mean of first and last rows"),
         ("dataset", "A chosen data set (row by row)"),
         ("value", "A constant value"))
_REL_TOL = 1e-9  # |100% - 0%| below this fraction of the larger baseline counts as zero


def _cell(v):
    return None if np.isnan(v) else float(v)


def table_display(table: Table, title: str) -> dict:
    """Results-view table of a Table: X (or row title) followed by every subcolumn."""
    fields = Y_FORMATS[table.y_format]
    cols = [table.x_title if table.x is not None else "Row"]
    for ds in table.datasets:
        subs = fields or [f"rep{i + 1}" for i in range(ds.values.shape[1])]
        cols += [f"{ds.title}:{s}" for s in subs]
    rows = []
    for i in range(table.n_rows):
        head = _cell(table.x[i]) if table.x is not None else (table.row_titles[i] if i < len(table.row_titles) else i + 1)
        rows.append([head] + [_cell(v) for ds in table.datasets for v in ds.values[i]])
    return {"title": title, "columns": cols, "rows": rows}


def _new_table(table: Table, suffix: str, datasets: list[DataSet], y_format: str | None = None) -> Table:
    return Table(id=new_id("tbl"), title=f"{table.title} ({suffix})", kind=table.kind, y_format=y_format or table.y_format,
                 datasets=datasets, x=None if table.x is None else table.x.copy(), x_title=table.x_title,
                 row_titles=list(table.row_titles), source=None)


def _failed(kind: str, messages: list[str]) -> dict:
    return {"kind": kind, "status": "failed", "messages": messages, "display": [], "output_table": None}


def _check_ids(table: Table, dataset_ids) -> list[str]:
    ids = list(dataset_ids)
    known = {d.id for d in table.datasets}
    return [f"unknown data set {i!r}" for i in ids if i not in known] or ([] if ids else ["no data set selected"])


def _baseline_value(mode: str, mean: np.ndarray, given: float | None, which: str) -> float:
    if mode == "min":
        return float(np.nanmin(mean)) if np.any(~np.isnan(mean)) else np.nan
    if mode == "max":
        return float(np.nanmax(mean)) if np.any(~np.isnan(mean)) else np.nan
    if mode == "first":
        return float(mean[0]) if len(mean) else np.nan
    if mode == "last":
        return float(mean[-1]) if len(mean) else np.nan
    if mode == "value":
        return np.nan if given is None else float(given)
    raise ValueError(f"unknown {which} option {mode!r}")


def normalize(table: Table, dataset_ids, params: dict) -> dict:
    errs = _check_ids(table, dataset_ids)
    if errs:
        return _failed("normalize", errs)
    units = params.get("units", "percent")
    if units not in ("percent", "fraction"):
        return _failed("normalize", [f"unknown units {units!r}"])
    scale = 100.0 if units == "percent" else 1.0
    fields = Y_FORMATS[table.y_format]
    out, msgs = [], []
    for ds_id in dataset_ids:
        ds = table.dataset(ds_id)
        mean = table.summary(ds_id)["mean"]
        try:
            z = _baseline_value(params.get("zero", "min"), mean, params.get("zero_value"), "0%")
            h = _baseline_value(params.get("hundred", "max"), mean, params.get("hundred_value"), "100%")
        except ValueError as e:
            return _failed("normalize", [str(e)])
        if np.isnan(z) or np.isnan(h):
            msgs.append(f"data set {ds.title!r}: the 0% or 100% value is blank, cannot normalize")
            continue
        d = h - z
        if d == 0 or abs(d) <= _REL_TOL * max(abs(h), abs(z)):
            msgs.append(f"data set {ds.title!r}: 100% ({h:g}) and 0% ({z:g}) are equal or nearly equal, cannot normalize")
            continue
        v = ds.values.copy()
        if fields is None:
            v = (v - z) / d * scale
        else:
            for j, f in enumerate(fields):
                if f == "mean":
                    v[:, j] = (v[:, j] - z) / d * scale
                elif f in ("sd", "sem"):
                    v[:, j] = v[:, j] / abs(d) * scale
        out.append(DataSet(ds.id, ds.title, v, ds.excluded.copy()))
    if msgs:
        return _failed("normalize", msgs)
    t = _new_table(table, "normalized", out)
    return {"kind": "normalize", "status": "ok", "messages": [], "display": [table_display(t, "Normalized data")],
            "output_table": t}


def _baseline(mode: str, s: dict, k: int, given, base: dict | None):
    """Baseline (mean, SEM) of one data set summary ``s``: scalars, or per-row arrays for mode "dataset".

    "first_last": mean of the row means of the first k and last k rows (Prism default k = 3), SEM = SD of
    those row means / sqrt(count). "first"/"last": that row's mean and SEM. "value": a constant, SEM 0.
    """
    mean, sem = s["mean"], s["sem"]
    if mode == "first":
        return (mean[0], sem[0]) if len(mean) else (np.nan, np.nan)
    if mode == "last":
        return (mean[-1], sem[-1]) if len(mean) else (np.nan, np.nan)
    if mode == "value":
        return float(given), 0.0
    if mode == "dataset":
        return base["mean"], base["sem"]
    n = len(mean)  # first_last; overlapping rows (n < 2k) are counted once
    m = mean[sorted(set(range(min(k, n))) | set(range(max(n - k, 0), n)))]
    m = m[~np.isnan(m)]
    if not len(m):
        return np.nan, np.nan
    return m.mean(), (m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.nan)


def subtract_baseline(table: Table, dataset_ids, params: dict) -> dict:
    """Prism "Remove baseline and column math" with the subtraction (difference) expression.

    subcolumns "means" (Prism default, treat subcolumns as MEANS): output is a mean / SEM table (no n) with
    SEM_out = sqrt(SEM_row^2 + SEM_baseline^2). "replicates": every replicate is shifted, the layout is kept
    (mean/error layouts shift only the mean column).
    """
    errs = _check_ids(table, dataset_ids)
    if errs:
        return _failed("subtract_baseline", errs)
    mode = params.get("baseline", "first")
    if mode not in ("first", "last", "first_last", "dataset", "value"):
        return _failed("subtract_baseline", [f"unknown baseline option {mode!r}"])
    sub = params.get("subcolumns", "means")
    if sub not in ("means", "replicates"):
        return _failed("subtract_baseline", [f"unknown subcolumns option {sub!r}"])
    k = params.get("baseline_rows", 3)
    if mode == "first_last" and (not isinstance(k, int) or k < 1):
        return _failed("subtract_baseline", ["rows at each end must be a whole number of at least 1"])
    fields = Y_FORMATS[table.y_format]
    cols = slice(None) if fields is None else [fields.index("mean")]
    base = None
    if mode == "dataset":
        bid = params.get("baseline_dataset")
        if bid not in {d.id for d in table.datasets}:
            return _failed("subtract_baseline", [f"baseline data set {bid!r} is not in the table"])
        base = table.summary(bid)
    elif mode == "value" and params.get("baseline_value") is None:
        return _failed("subtract_baseline", ["no baseline value given"])
    out, msgs = [], []
    for ds_id in dataset_ids:
        ds = table.dataset(ds_id)
        s = table.summary(ds_id)
        b, sb = _baseline(mode, s, k, params.get("baseline_value"), base)
        if np.ndim(b) == 0 and np.isnan(b):
            msgs.append(f"data set {ds.title!r}: baseline row is blank, cannot subtract")
            continue
        if sub == "means":
            with np.errstate(invalid="ignore"):
                v = np.column_stack([s["mean"] - b, np.hypot(s["sem"], sb)])
            out.append(DataSet(ds.id, ds.title, v))
        else:
            v = ds.values.copy()
            v[:, cols] = v[:, cols] - (b if np.ndim(b) == 0 else b[:, None])
            out.append(DataSet(ds.id, ds.title, v, ds.excluded.copy()))
    if msgs:
        return _failed("subtract_baseline", msgs)
    t = _new_table(table, "baseline subtracted", out, "mean_sem" if sub == "means" else None)
    base_row = None if base is None else base["mean"]
    warn = []
    if mode == "dataset" and np.any(np.isnan(base_row)):
        warn.append("the baseline data set is blank in some rows; those rows are blank in the output")
    return {"kind": "subtract_baseline", "status": "warning" if warn else "ok", "messages": warn,
            "display": [table_display(t, "Data with baseline removed")], "output_table": t}


KINDS = {
    "normalize": AnalysisKind(
        "normalize", "Normalize", ("XY", "Column", "Grouped"), normalize,
        (Param("zero", "0% is defined as", "choice", "min", _ZERO),
         Param("zero_value", "0% value", "float?", None),
         Param("hundred", "100% is defined as", "choice", "max", _HUNDRED),
         Param("hundred_value", "100% value", "float?", None),
         Param("units", "Results as", "choice", "percent", (("percent", "Percentages"), ("fraction", "Fractions")))),
        output="table"),
    "subtract_baseline": AnalysisKind(
        "subtract_baseline", "Remove baseline", ("XY", "Column", "Grouped"), subtract_baseline,
        (Param("baseline", "Subtract", "choice", "first", _BASE),
         Param("baseline_dataset", "Baseline data set", "dataset", None),
         Param("baseline_value", "Baseline value", "float?", None),
         Param("baseline_rows", "Rows at each end (mean of first and last rows)", "int", 3),
         Param("subcolumns", "Treat subcolumns as", "choice", "means",
               (("means", "Means (output mean and SEM)"), ("replicates", "Replicates")))),
        output="table"),
}
