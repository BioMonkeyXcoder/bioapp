"""Headless agent API: create tables, run analyses, make and style graphs, export figures. No Qt.

``Session`` holds one ``Project`` (and its .biofig path) and mirrors the GUI's semantics synchronously:
every data edit bumps the table revision and re-runs the dependent analyses (transform output tables and
chains included), new graphs get the GUI's defaults and auto-linking, figures are rebuilt from a PlotSpec plus
stored results. Errors are ``ApiError`` (a ValueError) with a message that says what is valid.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.util
import io
import math
import time
from pathlib import Path

import numpy as np

from . import analysis, plot
from .data import Analysis, DataSet, Project, Table, Y_FORMATS, new_id
from .project import atomic_write_bytes, load_project, save_project

HISTORY_KEEP = 10
X_KINDS = ("XY", "Survival")
TITLE_KINDS = ("Grouped", "Contingency", "MultipleVariables")
PAD_KINDS = ("Column", "Nested", "MultipleVariables")  # data sets may differ in length (padded with blanks)


class ApiError(ValueError):
    """A request the API cannot carry out; the message is meant to be read by the caller."""


# ---------------------------------------------------------------- JSON helpers

def jsonable(o):
    """Numpy -> plain Python; NaN -> None, +-inf -> "inf" / "-inf"."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    return o


def _sig(o, digits: int = 4):
    """jsonable + floats rounded to ``digits`` significant digits (result tables)."""
    if isinstance(o, (list, tuple)):
        return [_sig(v, digits) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and math.isfinite(o):
        return float(f"{o:.{digits}g}")
    return jsonable(o)


@functools.cache
def _catalog():
    """bioapp/ui/catalog.py read by path: pure data, and it must not pull in the Qt package."""
    spec = importlib.util.spec_from_file_location("_bioapp_catalog", Path(__file__).parent / "ui" / "catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick(d: dict, key: str, what: str):
    if key not in d:
        raise ApiError(f"unknown {what} {key!r}; known: {sorted(d)}")
    return d[key]


# ---------------------------------------------------------------- table construction

def _array(values, what: str) -> np.ndarray:
    try:
        return np.array(values, dtype=float)  # None -> NaN; ragged or text -> ValueError
    except (ValueError, TypeError) as e:
        raise ApiError(f"{what}: values must be numbers (null = blank cell) in equal-length rows: {e}") from None


def _is_text(v) -> bool:
    if not isinstance(v, str):
        return False
    try:
        float(v)
        return False
    except ValueError:
        return bool(v.strip())


def _make_dataset(spec: dict, kind: str, y_format: str) -> DataSet:
    if not isinstance(spec, dict) or not isinstance(spec.get("values"), (list, tuple, np.ndarray)) or len(spec["values"]) == 0:
        raise ApiError("each data set needs {'title': str, 'values': non-empty list}")
    title = str(spec.get("title") or "Data set")
    cats = spec.get("categories")
    flat = np.ravel(np.array(spec["values"], dtype=object))
    if kind == "MultipleVariables" and (cats is not None or any(_is_text(v) for v in flat)):
        labels = [str(c) for c in cats] if cats is not None else list(dict.fromkeys(str(v) for v in flat if _is_text(v)))
        try:
            vals = [(np.nan if v is None or v == "" else float(v) if not _is_text(v) else labels.index(str(v)))
                    for v in spec["values"]]
        except ValueError:
            raise ApiError(f"{title}: a value is not among the categories {labels}") from None
        arr, cats = np.array(vals, dtype=float), labels
    else:
        arr = _array(spec["values"], title)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ApiError(f"{title}: values must be 1-D or 2-D (rows x replicates)")
    return DataSet(new_id("ds"), title, arr, sub_titles=[str(s) for s in spec.get("sub_titles") or []], categories=cats)


def _pad(ds: DataSet, n: int) -> None:
    add = n - ds.values.shape[0]
    if add > 0:
        ds.values = np.vstack([ds.values, np.full((add, ds.values.shape[1]), np.nan)])
        ds.excluded = np.vstack([ds.excluded, np.zeros((add, ds.values.shape[1]), bool)])


def build_table(kind: str, title: str, x=None, datasets=(), row_titles=None, y_format: str = "replicates",
                x_title: str | None = None) -> Table:
    """Validate and build a Table (see ``Session.add_table``)."""
    if not datasets:
        raise ApiError("give at least one data set")
    if y_format not in Y_FORMATS:
        raise ApiError(f"unknown y_format {y_format!r}; valid: {list(Y_FORMATS)}")
    sets =[_make_dataset(d, kind, y_format) for d in datasets]
    if kind in PAD_KINDS and x is None and not row_titles:
        n = max(d.values.shape[0] for d in sets)
        for d in sets:
            _pad(d, n)
    if kind in X_KINDS and x is None:
        raise ApiError(f"{kind} tables need x (one value per row)")
    if kind not in X_KINDS and x is not None:
        raise ApiError(f"x is only for XY and Survival tables, not {kind}")
    if row_titles and kind not in TITLE_KINDS:
        raise ApiError(f"row_titles are for {'/'.join(TITLE_KINDS)} tables, not {kind}")
    xa = _array(x, "x").ravel() if x is not None else None
    n = len(xa) if xa is not None else (len(row_titles) if row_titles else sets[0].values.shape[0])
    for d in sets:
        if d.values.shape[0] != n:
            raise ApiError(f"data set {d.title!r} has {d.values.shape[0]} rows, expected {n}"
                           f" (from {'x' if xa is not None else 'row_titles' if row_titles else 'the first data set'})")
    fields = Y_FORMATS[y_format]
    if fields is not None:
        for d in sets:
            if d.values.shape[1] != len(fields):
                raise ApiError(f"y_format {y_format!r} needs {len(fields)} columns {list(fields)} per data set; "
                               f"{d.title!r} has {d.values.shape[1]}")
    titles = [str(r) for r in row_titles] if row_titles else [""] * n if kind in TITLE_KINDS else []
    try:
        return Table(new_id("tbl"), title, kind, y_format, sets, xa, x_title or ("Time" if kind == "Survival" else "X"), titles)
    except ValueError as e:
        raise ApiError(str(e)) from None


def _ds_id(t: Table, ref) -> str:
    """Data set id from an id, a unique title or a 0-based index."""
    if isinstance(ref, int) and not isinstance(ref, bool):
        if not -len(t.datasets) <= ref < len(t.datasets):
            raise ApiError(f"data set index {ref} out of range (table has {len(t.datasets)})")
        return t.datasets[ref].id
    hits = [d.id for d in t.datasets if d.id == ref] or [d.id for d in t.datasets if d.title == ref]
    if len(hits) != 1:
        raise ApiError(f"{'no' if not hits else 'ambiguous'} data set {ref!r} in table {t.id}; "
                       f"data sets: {[(d.id, d.title) for d in t.datasets]}")
    return hits[0]


# ---------------------------------------------------------------- patch checks

def _enums() -> dict[str, tuple]:
    g = lambda n: getattr(plot, n, None)  # noqa: E731  (plot.py is edited by other workers: read at call time)
    e = {"error": g("ERRORS"), "band": g("BANDS"), "band_style": g("BAND_STYLES"), "legend_loc": g("LEGEND_LOCS"),
         "curve_style": g("LINE_STYLES"), "err_dir": g("ERR_DIRS"), "hatch": g("HATCHES"),
         "normalize": g("TRACE_NORMALIZE"), "baseline": g("TRACE_BASELINE"), "show": g("BLI_SHOW"),
         "palette": g("BLI_PALETTES"), "fit_style": g("LINE_STYLES")}
    return {k: v for k, v in e.items() if v}


def _set(obj, name: str, value) -> None:
    """Set a dataclass field after a type / choice / colour check (None is allowed for ``... | None`` fields)."""
    ftype = next(f.type for f in dataclasses.fields(obj) if f.name == name)
    if value is None and "None" in str(ftype):
        return setattr(obj, name, None)
    cur = getattr(obj, name)
    if isinstance(cur, bool):
        ok = isinstance(value, bool)
    elif isinstance(cur, str):
        ok = isinstance(value, str)
    else:  # numbers, and fields that are currently None (number or text)
        ok = isinstance(value, (int, float, str)) and not isinstance(value, bool) and (
            cur is None or isinstance(value, (int, float)))
        if ok and isinstance(cur, int) and isinstance(value, float) and value.is_integer():
            value = int(value)
    if not ok:
        raise ApiError(f"{name}: expected {'true/false' if isinstance(cur, bool) else 'text' if isinstance(cur, str) else 'a number'}"
                       f"{' (or null)' if 'None' in str(ftype) else ''}, got {value!r}")
    enum = _enums().get(name)
    if enum and value not in enum:
        raise ApiError(f"{name}: {value!r} is not valid; choose from {list(enum)}")
    if isinstance(value, str) and value and (name.endswith("color") or name == "background") and value != "transparent":
        import matplotlib.colors as mc
        if not mc.is_color_like(value):
            raise ApiError(f"{name}: {value!r} is not a colour; use '#RRGGBB' or a Matplotlib colour name")
    setattr(obj, name, value)


def _merge_fields(target, patch: dict, what: str, skip=()) -> list[str]:
    valid = sorted({f.name for f in dataclasses.fields(target)} - set(skip))
    for k, v in patch.items():
        if k not in valid:
            raise ApiError(f"unknown {what} key {k!r}; valid keys: {valid}")
        _set(target, k, v)
    return list(patch)


def _patch_trace(spec, patch) -> list[str]:
    """Merge a ``trace`` patch (see plot.TraceSpec) into ``spec.trace``: scalars replace, ``y2_axis`` merges key by key,
    lists / dicts replace. Series in ``right`` / ``peak_series`` may be given by data set id or label."""
    tr = spec.trace
    valid = sorted(f.name for f in dataclasses.fields(tr))
    if not isinstance(patch, dict):
        raise ApiError("trace must be an object")

    def series_ids(v, key):
        if not isinstance(v, list):
            raise ApiError(f"trace.{key} must be a list of data set ids or series labels")
        out = []
        for ref in v:
            hits = [s for s in spec.series if s.dataset_id == ref] or [s for s in spec.series if s.label == ref]
            if len(hits) != 1:
                raise ApiError(f"trace.{key}: no series {ref!r}; series: {[(s.dataset_id, s.label) for s in spec.series]}")
            out.append(hits[0].dataset_id)
        return out

    def window(v, key):
        if not isinstance(v, list) or (v and (len(v) != 2 or not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in v)
                                              or not v[0] < v[1])):
            raise ApiError(f"trace.{key} must be [] or [x0, x1] with x0 < x1")
        return [float(a) for a in v]

    def events(v):
        keys = {"x", "label", "kind", "run"}
        if not isinstance(v, list) or any(not isinstance(e, dict) or not {"x", "label"} <= set(e) <= keys for e in v):
            raise ApiError('trace.events must be a list of {"x": number, "label": text, "kind"?: text, "run"?: text}')
        try:
            return [{"x": float(e["x"]), "label": str(e["label"]), "kind": str(e.get("kind", "")), "run": str(e.get("run", ""))}
                    for e in v]
        except (TypeError, ValueError):
            raise ApiError("trace.events: x must be a number") from None

    changed = []
    for k, v in patch.items():
        if k not in valid:
            raise ApiError(f"unknown trace key {k!r}; valid keys: {valid}")
        changed.append(k)
        if k == "y2_axis":
            if not isinstance(v, dict):
                raise ApiError("trace.y2_axis must be an object")
            changed.remove(k)
            changed += [f"y2_axis.{f}" for f in _merge_fields(tr.y2_axis, v, "trace.y2_axis")]
        elif k in ("right", "peak_series"):
            setattr(tr, k, series_ids(v, k))
        elif k == "event_kinds":
            if not isinstance(v, list) or not all(isinstance(a, str) for a in v):
                raise ApiError("trace.event_kinds must be a list of text, e.g. [\"Fractions\"]")
            tr.event_kinds = list(v)
        elif k in ("norm_window", "baseline_window"):
            setattr(tr, k, window(v, k))
        elif k == "events":
            tr.events = events(v)
        elif k == "bli":
            if not isinstance(v, dict):
                raise ApiError("trace.bli must be an object")
            if not tr.bli.sensors:
                raise ApiError("this graph has no BLI data (import a BLI export with trace_import)")
            changed.remove(k)
            for f in ("boundary", "t0"):  # a number or null; _set would let text through for a field that is None now
                if v.get(f) is not None and (isinstance(v[f], bool) or not isinstance(v[f], (int, float))):
                    raise ApiError(f"trace.bli.{f}: expected a number (or null), got {v[f]!r}")
            for f in v:
                if f == "metrics":
                    have = plot.bli_metric_names(tr.bli)
                    if not isinstance(v[f], list) or not all(isinstance(a, str) for a in v[f]):
                        raise ApiError("trace.bli.metrics must be a list of metric names")
                    bad = [a for a in v[f] if plot.bli_metric_key(a) not in {plot.bli_metric_key(h) for h in have}]
                    if bad:
                        raise ApiError(f"trace.bli.metrics: the file has no {bad}; metrics found: {have}")
                    tr.bli.metrics = list(v[f])
                    changed.append("bli.metrics")
                elif f == "kinetics":  # rows are checked when the graph renders (plot._check_bli)
                    if not isinstance(v[f], list) or not all(isinstance(k, dict) and {"name"} <= set(k) <= {"name", "value", "unit"} for k in v[f]):
                        raise ApiError('trace.bli.kinetics must be a list of {"name": text, "value": number or null, "unit": text}')
                    tr.bli.kinetics = [{"name": k["name"], "value": k.get("value"), "unit": k.get("unit", "")} for k in v[f]]
                    changed.append("bli.kinetics")
            changed += [f"bli.{f}" for f in _merge_fields(tr.bli, {a: b for a, b in v.items() if a not in ("metrics", "kinetics")},
                                                          "trace.bli", skip=("sensors", "metrics", "kinetics"))]
            if "palette" in v:
                plot.apply_bli_palette(spec)  # recolours the raw curves by concentration
        elif k == "sources":
            if not isinstance(v, dict) or not all(isinstance(a, str) and isinstance(b, str) for a, b in v.items()):
                raise ApiError("trace.sources must be {data set id: table id}")
            tr.sources = dict(v)
        else:
            _set(tr, k, v)
    return changed


# ---------------------------------------------------------------- session

class Session:
    """One project in memory plus its file. Methods raise ``ApiError`` on bad requests."""

    def __init__(self, project: Project | None = None, path: str | Path | None = None):
        self.project = project or Project()
        self.path: Path | None = Path(path) if path else None
        self._mtime: int | None = None

    # ---- items
    def _table(self, tid: str) -> Table:
        return _pick(self.project.tables, tid, "table")

    def _analysis(self, aid: str) -> Analysis:
        return _pick(self.project.analyses, aid, "analysis")

    def _graph(self, gid: str):
        return _pick(self.project.graphs, gid, "graph")

    def results(self) -> dict[str, dict]:
        return {a.id: a.result for a in self.project.analyses.values() if a.result}

    # ---- project file
    def new(self, title: str = "Untitled") -> "Session":
        self.project, self.path, self._mtime = Project(title=title), None, None
        return self

    def open(self, path) -> "Session":
        path = Path(path)
        if not path.exists():
            raise ApiError(f"{path} does not exist")
        self.project, self.path = load_project(path), path
        self._mtime = path.stat().st_mtime_ns
        for a in self.project.analyses.values():  # a file saved mid-run carries no running state
            if a.status == "running":
                a.status = a.result["status"] if a.result else "not_run"
        self.refresh()
        return self

    def reload_if_changed(self) -> bool:
        """Re-read the file when someone else (the GUI) saved it since we last did."""
        if self.path and self.path.exists() and self.path.stat().st_mtime_ns != self._mtime:
            self.open(self.path)
            return True
        return False

    def _history_dir(self, path: Path) -> Path:
        return path.with_name(path.name + ".history")

    def _snapshots(self, path: Path) -> list[Path]:
        d = self._history_dir(path)
        return sorted(d.glob("*.biofig")) if d.is_dir() else []

    def save(self, path=None) -> str:
        """Atomic save. The file being replaced is first kept as a snapshot in '<name>.biofig.history/' (last 10)."""
        target = Path(path) if path else self.path
        if target is None:
            raise ApiError("no file name: give a path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old, snaps = target.read_bytes(), self._snapshots(target)
            if not snaps or snaps[-1].read_bytes() != old:  # a repeated identical state is not a new version
                d = self._history_dir(target)
                d.mkdir(exist_ok=True)
                atomic_write_bytes(d / f"{time.time_ns()}.biofig", old)
                for s in self._snapshots(target)[:-HISTORY_KEEP]:
                    s.unlink()
        save_project(self.project, target)
        self.path, self._mtime = target, target.stat().st_mtime_ns
        return str(target)

    def undo(self) -> str:
        """Restore the state the file had before its last save (repeatable, up to 10 steps)."""
        if self.path is None:
            raise ApiError("no project file; nothing to undo")
        snaps = self._snapshots(self.path)
        if not snaps:
            raise ApiError("nothing to undo: no earlier saved version")
        snap = snaps[-1]
        data = snap.read_bytes()
        atomic_write_bytes(self.path, data)
        snap.unlink()
        return str(self.open(self.path).path)

    # ---- summary
    def graph_title(self, gid: str) -> str:
        spec = self._graph(gid)
        t = self.project.tables.get(spec.table_id)
        what = "Graph" if spec.kind in ("xy", "column", "grouped") else spec.kind.replace("_", " ").capitalize()
        return spec.name or spec.title or f"{what} of {t.title if t else '?'}"

    def summary(self) -> dict:
        p = self.project
        return {
            "title": p.title, "path": str(self.path) if self.path else None,
            "tables": [{"id": t.id, "title": t.title, "kind": t.kind, "y_format": t.y_format, "rows": t.n_rows,
                        "revision": t.revision, "output_of": t.source,
                        "datasets": [{"id": d.id, "title": d.title, "subcolumns": d.values.shape[1]} for d in t.datasets]}
                       for t in p.tables.values()],
            "analyses": [{"id": a.id, "title": a.title, "kind": a.kind, "table_id": a.table_id, "status": a.status,
                          "stale": not a.is_current(p.tables[a.table_id]) if a.table_id in p.tables else True,
                          "output_table_id": a.output_table_id} for a in p.analyses.values()],
            "graphs": [{"id": g.id, "title": self.graph_title(g.id), "kind": g.kind, "table_id": g.table_id,
                        "fit_analysis_id": g.fit_analysis_id, "stats_analysis_id": g.stats_analysis_id,
                        "analysis_id": g.analysis_id} for g in p.graphs.values()],
            "layouts": [{"id": lay.id, "title": lay.title, "size_mm": [lay.width_mm, lay.height_mm],
                         "panels": [{"graph_id": q.graph_id, "label": q.label} for q in lay.panels]}
                        for lay in p.layouts.values()],
        }

    # ---- tables
    def add_table(self, kind: str, title: str | None = None, x=None, datasets=(), row_titles=None,
                  y_format: str = "replicates", x_title: str | None = None) -> dict:
        """Create a table. ``datasets``: [{title, values (1-D = one column, 2-D = rows x replicates / mean+error
        columns), sub_titles?, categories?}]. XY / Survival need ``x`` (one per row). Returns {id, ...}."""
        t = build_table(kind, title or f"Data {len(self.project.tables) + 1}", x, datasets, row_titles, y_format, x_title)
        self.project.tables[t.id] = t
        self.refresh()
        return {"id": t.id, "title": t.title, "kind": kind, "rows": t.n_rows, "datasets": [(d.id, d.title) for d in t.datasets]}

    def import_file(self, path, kind: str = "XY", **options) -> dict:
        """CSV / TSV / TXT / XLSX -> one table (``options`` go to io_tabular.grid_to_table, plus ``sheet``);
        .pzfx -> every supported table of the file (``kind`` ignored)."""
        path = Path(path)
        if not path.exists():
            raise ApiError(f"{path} does not exist")
        from . import io_prism, io_tabular
        try:
            if path.suffix.lower() == ".pzfx":
                tables, report = io_prism.read_pzfx(path)
            else:
                sheet = options.pop("sheet", None)
                options.setdefault("title", path.stem)
                t, report = io_tabular.grid_to_table(io_tabular.read_grid(path, sheet), kind, **options)
                tables = [t]
        except TypeError as e:
            raise ApiError(f"bad import option: {e}") from None
        for t in tables:
            self.project.tables[t.id] = t
        self.refresh()
        return {"tables": [{"id": t.id, "title": t.title, "kind": t.kind, "rows": t.n_rows,
                            "datasets": [(d.id, d.title) for d in t.datasets]} for t in tables], "report": report}

    def import_traces(self, paths, traces=None, right_axis=None, title: str | None = None) -> dict:
        """Chromatography / instrument CSV, TSV or TXT exports -> one XY table per trace (X = volume or time) plus one
        "trace" graph drawing them (lines straight from the raw samples, fraction / injection marks along the X axis).

        ``paths``: one file or a list (several files overlay several runs). ``traces``: names of the traces to keep
        (default all numeric curves; "run/name" picks one run's curve). ``right_axis``: names drawn against the right Y
        axis (default: traces whose unit differs from the first trace's; [] = none). Returns the graph id, the
        tables and, per trace, its table id, data set id, run and axis."""
        from . import io_traces
        paths = [paths] if isinstance(paths, (str, Path)) else list(paths or [])
        if not paths:
            raise ApiError("give at least one file path")
        try:
            found, events = io_traces.read_files(paths)
        except io_traces.TraceImportError as e:
            raise ApiError(str(e)) from None

        def pick(names, what):
            want = [str(n).casefold() for n in ([names] if isinstance(names, str) else names)]
            hit = [any(w in (t.name.casefold(), f"{t.run}/{t.name}".casefold()) for w in want) for t in found]
            lost = [n for n in want if not any(n in (t.name.casefold(), f"{t.run}/{t.name}".casefold()) for t in found)]
            if lost:
                raise ApiError(f"{what}: no trace named {lost}; traces found: {sorted({t.name for t in found})}")
            return hit
        keep = pick(traces, "traces") if traces is not None else [True] * len(found)
        sensors = {id(t.bli) for t, k in zip(found, keep) if k and t.bli}  # a BLI sensor keeps its raw and fit curve together
        keep = [k or id(t.bli) in sensors for t, k in zip(found, keep)]
        right = pick(right_axis, "right_axis") if right_axis is not None else None
        if right is not None:
            right = [r for r, k in zip(right, keep) if k]
        found = [t for t, k in zip(found, keep) if k]
        if not found:
            raise ApiError("no numeric traces to import")
        used = {t.run for t in found}
        try:
            tables, spec = io_traces.traces_to_graph(found, [e for e in events if e.run in used], right, title=title or "")
            self._render_check(spec, {**self.project.tables, **{t.id: t for t in tables}})
        except ValueError as e:
            raise ApiError(str(e)) from None
        for t in tables:
            self.project.tables[t.id] = t
        self.project.graphs[spec.id] = spec
        self.refresh()
        out = {"graph_id": spec.id, "events": len(spec.trace.events), "tables": [
            {"id": t.id, "title": t.title, "rows": t.n_rows, "dataset_id": t.datasets[0].id, "run": tr.run, "name": tr.name,
             "axis": "right" if t.datasets[0].id in spec.trace.right else "left", **({"role": tr.role} if tr.bli else {})}
            for t, tr in zip(tables, found)]}
        if spec.trace.bli.sensors:
            b = spec.trace.bli
            out["bli"] = {"sensors": len(b.sensors), "boundary": b.boundary, "t0": b.t0, "metrics": plot.bli_metric_names(b)}
        return out

    def get_table(self, tid: str, max_rows: int = 50) -> dict:
        t = self._table(tid)
        n = min(t.n_rows, max(0, max_rows))
        return jsonable({
            "id": t.id, "title": t.title, "kind": t.kind, "y_format": t.y_format, "revision": t.revision,
            "rows": t.n_rows, "shown_rows": n, "x_title": t.x_title if t.x is not None else None,
            "x": None if t.x is None else t.x[:n], "row_titles": t.row_titles[:n],
            "datasets": [{"id": d.id, "title": d.title, "sub_titles": d.sub_titles, "categories": d.categories,
                          "subcolumns": list(Y_FORMATS[t.y_format] or range(1, d.values.shape[1] + 1)),
                          "values": d.values[:n],
                          "excluded_cells": [[int(r), int(c)] for r, c in zip(*np.nonzero(d.excluded[:n]))]}
                         for d in t.datasets]})

    def set_values(self, table_id: str, dataset=None, values=None, cells=None, start_row: int = 0, x=None,
                   row_titles=None) -> dict:
        """Edit data, then re-run every dependent analysis.
        ``values``: rows written into ``dataset`` from ``start_row`` (the table grows when they run past the end).
        ``cells``: [{row, sub?=0, value?, excluded?}] for ``dataset`` (value null = blank; excluded true/false).
        ``x`` / ``row_titles``: written from ``start_row``. ``dataset`` is an id, a title or a 0-based index."""
        t = self._table(table_id)
        backup = copy.deepcopy((t.datasets, t.x, t.row_titles))
        try:
            self._apply_edits(t, dataset, values, cells, start_row, x, row_titles)
            t.validate()
        except Exception as e:
            t.datasets, t.x, t.row_titles = backup
            raise e if isinstance(e, ApiError) else ApiError(str(e)) from None
        t.bump()
        self.refresh()
        return {"table_id": t.id, "revision": t.revision, "rows": t.n_rows,
                "analyses": [{"id": a.id, "status": a.status} for a in self.project.analyses.values() if a.table_id == t.id]}

    def _apply_edits(self, t: Table, dataset, values, cells, start_row, x, row_titles) -> None:
        if start_row < 0:
            raise ApiError("start_row must be >= 0")
        if values is None and cells is None and x is None and row_titles is None:
            raise ApiError("give values, cells, x or row_titles")
        ds = t.dataset(_ds_id(t, dataset)) if (values is not None or cells) else None
        if (values is not None or cells) and dataset is None:
            raise ApiError("dataset is required with values / cells")

        def grow(n):
            _grow(t, n)

        if values is not None:
            arr = _array(values, "values")
            arr = arr[:, None] if arr.ndim == 1 else arr
            if arr.ndim != 2 or arr.shape[1] != ds.values.shape[1]:
                raise ApiError(f"{ds.title}: values need {ds.values.shape[1]} column(s) per row, got shape {arr.shape}")
            grow(start_row + len(arr))
            ds.values[start_row:start_row + len(arr)] = arr
        for c in cells or []:
            if not isinstance(c, dict) or "row" not in c:
                raise ApiError(f"each cell needs {{row, sub?, value?, excluded?}}, got {c!r}")
            r, s = int(c["row"]), int(c.get("sub", 0))
            if r < 0 or not 0 <= s < ds.values.shape[1]:
                raise ApiError(f"cell {c}: row must be >= 0 and sub in 0..{ds.values.shape[1] - 1}")
            grow(r + 1)
            if "value" in c:
                ds.values[r, s] = _array(c["value"], f"cell {c}")
            if "excluded" in c:
                ds.excluded[r, s] = bool(c["excluded"])
        if x is not None:
            if t.x is None:
                raise ApiError(f"{t.kind} tables have no X column")
            xa = _array(x, "x").ravel()
            grow(start_row + len(xa))
            t.x[start_row:start_row + len(xa)] = xa
        if row_titles is not None:
            if t.kind not in TITLE_KINDS:
                raise ApiError(f"{t.kind} tables have no row titles")
            grow(start_row + len(row_titles))
            t.row_titles += [""] * (t.n_rows - len(t.row_titles))
            t.row_titles[start_row:start_row + len(row_titles)] = [str(r) for r in row_titles]

    # ---- analyses
    def analysis_kinds(self, table_kind: str | None = None) -> list[dict]:
        cat = _catalog()
        out = []
        for k, spec in analysis.all_kinds().items():
            if table_kind and table_kind not in spec.table_kinds:
                continue
            params = []
            for p in spec.params:
                d = {"name": p.name, "label": p.label, "type": p.type, "default": jsonable(p.default)}
                if p.choices:
                    d["choices"] = [{"value": v, "label": lab} for v, lab in p.choices[:12]]
                    if len(p.choices) > 12:
                        d["n_choices"] = len(p.choices)
                        d["note"] = "long list: use model_search to find ids"
                params.append(d)
            out.append({"id": k, "label": spec.label, "category": cat.KIND_CATEGORY.get(k, ""),
                        "table_kinds": list(spec.table_kinds), "output": spec.output, "params": params})
        return out

    def models(self, query: str | None = None, limit: int = 25) -> dict:
        from .analysis.models import ALL
        eq = _catalog().EQUATIONS
        words = (query or "").lower().split()
        hits = [m for m in ALL if all(w in f"{m.id} {m.name} {m.family} {' '.join(m.params)}".lower() for w in words)]
        return {"total": len(hits), "models": [
            {"id": m.id, "name": m.name, "family": m.family, "parameters": list(m.params), "x_scale": m.x_scale,
             **({"constants": list(m.constants)} if m.constants else {}),
             **({"n_datasets": m.n_datasets} if m.n_datasets else {}),
             "equation": eq.get(m.id, "")} for m in hits[:limit]]}

    def _check_params(self, spec, params: dict) -> None:
        by = {p.name: p for p in spec.params}
        for k, v in params.items():
            if k not in by:
                raise ApiError(f"{spec.id}: unknown parameter {k!r}; valid: {list(by)}")
            p = by[k]
            if p.type == "choice" and v not in [c[0] for c in p.choices]:
                raise ApiError(f"{spec.id}.{k}: {v!r} is not valid; choose from {[c[0] for c in p.choices][:40]}"
                               f"{' (see model_search)' if k == 'model' else ''}")
            if p.type == "bool" and not isinstance(v, bool):
                raise ApiError(f"{spec.id}.{k}: expected true or false, got {v!r}")
            if p.type in ("float", "int") and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise ApiError(f"{spec.id}.{k}: expected a number, got {v!r}")

    def run_analysis(self, table_id: str, kind: str, dataset_ids=None, params: dict | None = None,
                     title: str | None = None) -> dict:
        """Create an analysis on a table and run it now. ``dataset_ids``: ids, titles or indices (default: all)."""
        t, kinds, params = self._table(table_id), analysis.all_kinds(), dict(params or {})
        spec = _pick(kinds, kind, "analysis kind")
        if t.kind not in spec.table_kinds:
            raise ApiError(f"{spec.label} needs a {'/'.join(spec.table_kinds)} table, {t.id} is {t.kind}")
        self._check_params(spec, params)
        ids = [t.datasets[i].id for i in range(len(t.datasets))] if dataset_ids is None else [_ds_id(t, r) for r in dataset_ids]
        a = Analysis(new_id("an"), title or f"{spec.label} of {t.title}", kind, t.id, ids, params)
        self.project.analyses[a.id] = a
        self.refresh()
        return self.result(a.id)

    def update_analysis(self, analysis_id: str, params: dict | None = None, dataset_ids=None,
                        title: str | None = None) -> dict:
        """Change an analysis' parameters (merged into the old ones) / data sets / title and re-run it."""
        a = self._analysis(analysis_id)
        t = self._table(a.table_id)
        spec = analysis.all_kinds()[a.kind]
        if params:
            self._check_params(spec, params)
            a.params = {**a.params, **params}
        if dataset_ids is not None:
            a.dataset_ids = [_ds_id(t, r) for r in dataset_ids]
        if title:
            a.title = title
        a.bump_params()
        self.refresh()
        return self.result(analysis_id)

    def result(self, analysis_id: str, raw: bool = False) -> dict:
        """Compact result (status, messages, display tables at 4 significant digits, comparisons) or the raw dict."""
        a = self._analysis(analysis_id)
        t = self.project.tables.get(a.table_id)
        stale = t is None or not a.is_current(t)
        if raw:
            return {"id": a.id, "kind": a.kind, "stale": stale, "result": jsonable(a.result)}
        r = a.result or {}
        titles = {d.id: d.title for d in t.datasets} if t else {}
        out = {"id": a.id, "title": a.title, "kind": a.kind, "table_id": a.table_id, "status": a.status, "stale": stale,
               "messages": r.get("messages", []),
               "tables": [{"title": d["title"], "columns": d["columns"], "rows": _sig(d["rows"])}
                          for d in r.get("display", [])]}
        if r.get("comparisons"):
            out["comparisons"] = [{"id": c["id"], "a": titles.get(c["a"], c["a"]), "b": titles.get(c["b"], c["b"]),
                                   "row": c.get("row"), "p_adj": _sig(c.get("p_adj")), "summary": c.get("summary")}
                                  for c in r["comparisons"]]
        if a.output_table_id:
            out["output_table_id"] = a.output_table_id
        return out

    # ---- synchronous (re-)run, like the GUI's auto re-run
    def refresh(self) -> None:
        """Run every stale analysis; transform output tables and chains settle in a few passes."""
        p = self.project
        for _ in range(len(p.analyses) + 1):
            stale = {a.id: a for a in p.analyses.values() if a.table_id in p.tables and not a.is_current(p.tables[a.table_id])}
            todo = [a for a in stale.values() if p.tables[a.table_id].source not in stale]  # upstream first
            if not todo:
                return
            for a in todo:
                self._run(a)

    def _run(self, a: Analysis) -> None:
        t = self.project.tables[a.table_id]
        key = a.key(t)
        try:
            res = analysis.run(a.kind, t, list(a.dataset_ids), dict(a.params))
        except Exception as e:  # a failed analysis is a stored result, as in the GUI
            res = {"kind": a.kind, "status": "failed", "messages": [f"{type(e).__name__}: {e}"], "display": []}
        out = res.pop("output_table", None)
        a.result, a.result_key, a.status = res, key, res.get("status", "ok")
        if out is not None:
            cur = self.project.tables.get(a.output_table_id) if a.output_table_id else None
            if cur is None:
                out.source, a.output_table_id = a.id, out.id
                self.project.tables[out.id] = out
            else:  # keep id / title, bump the revision so analyses on the output table re-run
                cur.__dict__.update({**out.__dict__, "id": cur.id, "title": cur.title, "revision": cur.revision + 1,
                                     "source": a.id})

    def delete(self, item_id: str) -> dict:
        """Delete a table (with its analyses, output tables and graphs), an analysis, a graph or a layout."""
        p = self.project
        for section in ("tables", "analyses", "graphs", "layouts"):
            if item_id in getattr(p, section):
                break
        else:
            raise ApiError(f"unknown id {item_id!r}")
        doomed = {item_id}
        if section in ("tables", "analyses"):
            changed = True
            while changed:  # analyses on a deleted table go; so do their output tables
                changed = False
                for a in p.analyses.values():
                    if (a.table_id in doomed or a.id in doomed) and not {a.id, a.output_table_id} <= doomed | {None}:
                        doomed |= {a.id, a.output_table_id} - {None}
                        changed = True
        plot.prune_trace_graphs(p.graphs, doomed)
        for d in (p.tables, p.analyses, p.layouts):
            for k in [k for k in d if k in doomed]:
                del d[k]
        for gid in [g for g, s in p.graphs.items()
                    if g in doomed or s.table_id in doomed or (s.analysis_id in doomed and s.kind != "xy")]:
            del p.graphs[gid]
        for k in ("fit_analysis_id", "stats_analysis_id", "analysis_id"):  # graphs that survive stop reading it
            for s in p.graphs.values():
                if getattr(s, k) in doomed:
                    setattr(s, k, None)
        return {"deleted": sorted(doomed)}

    # ---- graphs
    def graph_choices(self, table_id: str) -> list[dict]:
        t = self._table(table_id)
        res = {a.id: (a.title, a.result) for a in self.project.analyses.values() if a.table_id == table_id and a.result}
        return [{"label": lab, "kind": k, "analysis_id": aid} for lab, k, aid in plot.graph_choices(t, res)]

    def create_graph(self, table_id: str, kind: str | None = None, analysis_id: str | None = None) -> dict:
        """The GUI's "New graph": default spec of the table kind; unless ``analysis_id`` feeds it, an XY graph
        draws the table's latest fit and a Column / Grouped / Nested graph its latest comparison test."""
        t = self._table(table_id)
        if analysis_id is not None:
            self._analysis(analysis_id)
        try:
            spec = plot.default_spec(t, new_id("gr"), analysis_id, kind)
        except (KeyError, ValueError) as e:
            raise ApiError(f"cannot make a {kind or 'default'} graph of this table: {e}; options: {self.graph_choices(table_id)}") from None
        if analysis_id is None and t.kind != "MultipleVariables":
            plot.auto_link(spec, [(a.id, a.kind, a.result) for a in self.project.analyses.values() if a.table_id == t.id])
        self._render_check(spec)
        self.project.graphs[spec.id] = spec
        return {"id": spec.id, "kind": spec.kind, "spec": spec.to_dict()}

    def graphs_for_result(self, analysis_id: str) -> list[str]:
        """The GUI's "Graph this result": the natural graph(s) of an analysis ([] when it has none)."""
        a = self._analysis(analysis_id)
        t = self._table(a.table_id)
        specs = plot.specs_for_result(t, a.id, a.kind, a.result or {}, lambda k: new_id("gr"))
        out = self.project.tables.get(a.output_table_id or "")
        if specs and out is not None:  # a transform: the graph shows its output table
            s = plot.default_spec(out, specs[0].id)
            plot.auto_link(s, [(x.id, x.kind, x.result) for x in self.project.analyses.values() if x.table_id == out.id])
            specs = [s]
        for s in specs:
            self._render_check(s)
        self.project.graphs.update({s.id: s for s in specs})
        return [s.id for s in specs]

    def get_graph(self, graph_id: str) -> dict:
        self._sync_series(graph_id)
        return jsonable(self._graph(graph_id).to_dict())

    def _sync_series(self, gid: str) -> None:
        """Give data sets added after the graph was made a default style (as the GUI does)."""
        spec = self._graph(gid)
        t = self.project.tables.get(spec.table_id)
        if t is None or t.kind == "MultipleVariables":
            return
        have = {s.dataset_id for s in spec.series}
        spec.series += [s for s in plot.default_series(t) if s.dataset_id not in have]

    def _render_check(self, spec, tables: dict | None = None) -> None:
        tables = tables or self.project.tables
        t = _pick(tables, spec.table_id, "table")
        try:
            plot.build_figure(spec, t, self.results(), tables)
        except Exception as e:
            raise ApiError(f"the graph would not render: {type(e).__name__}: {e}") from None

    def update_graph(self, graph_id: str, patch: dict) -> dict:
        """Deep-merge ``patch`` into the PlotSpec. Scalars replace; ``x_axis`` / ``y_axis`` merge key by key;
        ``series`` is a list of {dataset_id (or its label), <SeriesStyle fields>}; ``brackets`` a list of
        {comparison_id, label?, show_ns?} (add or update) or {comparison_id, remove: true}.
        Unknown keys are rejected; the result must still render, or nothing changes."""
        self._sync_series(graph_id)
        new = plot.PlotSpec.from_dict(copy.deepcopy(self._graph(graph_id).to_dict()))
        if not isinstance(patch, dict) or not patch:
            raise ApiError("patch must be a non-empty object")
        valid = sorted({f.name for f in dataclasses.fields(new)} - {"id", "table_id"})
        for k in patch:
            if k not in valid:
                raise ApiError(f"unknown graph key {k!r}; valid keys: {valid}")
        changed: list[str] = []
        for k, v in patch.items():
            if k in ("series", "brackets"):
                continue
            if k in ("x_axis", "y_axis"):
                if not isinstance(v, dict):
                    raise ApiError(f"{k} must be an object")
                changed += [f"{k}.{f}" for f in _merge_fields(getattr(new, k), v, k)]
            elif k == "trace":
                changed += [f"trace.{f}" for f in _patch_trace(new, v)]
            elif k == "texts":  # replaces the whole list of free texts; each item completed from plot.new_text defaults
                if not isinstance(v, list) or not all(isinstance(t, dict) for t in v):
                    raise ApiError("texts must be a list of objects")
                proto = plot.new_text(0.5, 0.5)
                for t in v:
                    bad = sorted(set(t) - set(proto))
                    if bad or "text" not in t:
                        raise ApiError(f"each text needs 'text'; unknown keys {bad}; valid keys: {sorted(proto)}")
                new.texts = [{**plot.new_text(float(t.get("x", 0.5)), float(t.get("y", 0.5))), **t} for t in v]
                changed.append("texts")
            else:
                _set(new, k, v)
                changed.append(k)
        styles = getattr(plot, "STYLES", {}).get(new.kind)
        if styles and new.style not in styles:
            raise ApiError(f"style {new.style!r} is not valid for a {new.kind} graph; choose from {list(styles)}")
        for k in ("fit_analysis_id", "stats_analysis_id", "analysis_id"):
            if getattr(new, k) is not None and getattr(new, k) not in self.project.analyses:
                raise ApiError(f"{k}: unknown analysis {getattr(new, k)!r}; known: {sorted(self.project.analyses)}")
        for k in ("series", "brackets"):
            if k in patch and not isinstance(patch[k], list):
                raise ApiError(f"{k} must be a list of objects")
        for item in patch.get("series", []):
            if not isinstance(item, dict) or "dataset_id" not in item:
                raise ApiError("each series item needs 'dataset_id' (or the series label)")
            ref = item["dataset_id"]
            hits = [s for s in new.series if s.dataset_id == ref] or [s for s in new.series if s.label == ref]
            if len(hits) != 1:
                raise ApiError(f"no series {ref!r}; series: {[(s.dataset_id, s.label) for s in new.series]}")
            body = {k: v for k, v in item.items() if k != "dataset_id"}
            changed += [f"series[{hits[0].dataset_id}].{f}" for f in _merge_fields(hits[0], body, "series", skip=("dataset_id",))]
        if "brackets" in patch:
            res = self.results().get(new.stats_analysis_id or "", {})
            known = [c["id"] for c in res.get("comparisons") or []]
            for item in patch["brackets"]:
                cid = item.get("comparison_id") if isinstance(item, dict) else None
                if cid is None:
                    raise ApiError("each bracket item needs 'comparison_id'")
                if known and cid not in known:
                    raise ApiError(f"unknown comparison {cid!r}; comparisons of {new.stats_analysis_id}: {known}")
                if item.get("remove"):
                    new.brackets = [b for b in new.brackets if b.comparison_id != cid]
                else:
                    br = next((b for b in new.brackets if b.comparison_id == cid), None)
                    if br is None:
                        br = plot.BracketSpec(cid)
                        new.brackets.append(br)
                    _merge_fields(br, {k: v for k, v in item.items() if k != "comparison_id"}, "bracket",
                                  skip=("comparison_id",))
                changed.append(f"brackets[{cid}]")
        self._render_check(new)
        self.project.graphs[graph_id] = new
        return {"id": graph_id, "changed": changed}

    def apply_look(self, graph_id: str, name: str) -> dict:
        """Set the look fields of a graph (``plot.LOOKS``: "default" | "bold"); see ``plot.apply_look``."""
        if name not in plot.LOOKS:
            raise ApiError(f"unknown look {name!r}; choose from {list(plot.LOOKS)}")
        self._sync_series(graph_id)
        new = plot.PlotSpec.from_dict(copy.deepcopy(self._graph(graph_id).to_dict()))
        plot.apply_look(new, name)
        self._render_check(new)
        self.project.graphs[graph_id] = new
        return {"id": graph_id, "look": name, "width_mm": new.width_mm, "height_mm": new.height_mm}

    def palettes(self) -> list[str]:
        return list(getattr(plot, "PALETTES", {}))

    def apply_palette(self, graph_id: str, name: str) -> dict:
        names = self.palettes()
        if not names or not hasattr(plot, "apply_palette"):
            raise ApiError("this build of plot.py has no palettes")
        if name not in names:
            raise ApiError(f"unknown palette {name!r}; choose from {names}")
        self._sync_series(graph_id)
        new = plot.PlotSpec.from_dict(copy.deepcopy(self._graph(graph_id).to_dict()))
        plot.apply_palette(new, name)
        self._render_check(new)
        self.project.graphs[graph_id] = new
        return {"id": graph_id, "palette": name, "colors": [s.color for s in new.series]}

    # ---- layouts, figures
    def create_layout(self, graph_ids: list[str], rows: int = 1, cols: int | None = None, width_mm: float = 180.0,
                      height_mm: float | None = None, labels: bool = True, title: str = "") -> dict:
        """Grid of panels (left to right, top to bottom), labelled A, B, ... Height defaults to the aspect of
        the first graph. Panel boxes include their axis labels."""
        if not graph_ids:
            raise ApiError("give at least one graph id")
        specs = [self._graph(g) for g in graph_ids]
        cols = cols or math.ceil(len(graph_ids) / rows)
        if rows < 1 or cols < 1 or len(graph_ids) > rows * cols:
            raise ApiError(f"{len(graph_ids)} graphs do not fit a {rows} x {cols} grid")
        m, gap = 2.0, 4.0
        pw = (width_mm - 2 * m - (cols - 1) * gap) / cols
        ph = (pw * specs[0].height_mm / specs[0].width_mm) if height_mm is None else (height_mm - 2 * m - (rows - 1) * gap) / rows
        if pw <= 0 or ph <= 0:
            raise ApiError("page too small for this grid")
        height_mm = height_mm or 2 * m + rows * ph + (rows - 1) * gap
        panels = [plot.PanelSpec(g, m + (i % cols) * (pw + gap), m + (i // cols) * (ph + gap), pw, ph,
                                 chr(65 + i % 26) if labels else "") for i, g in enumerate(graph_ids)]
        lay = plot.LayoutSpec(new_id("lay"), title or f"Layout {len(self.project.layouts) + 1}", width_mm, height_mm, panels)
        self._layout_figure(lay)
        self.project.layouts[lay.id] = lay
        return {"id": lay.id, "title": lay.title, "size_mm": [round(width_mm, 1), round(height_mm, 1)],
                "panels": [{"graph_id": p.graph_id, "label": p.label} for p in panels]}

    def _layout_figure(self, lay):
        res, graphs = self.results(), {}
        for p in lay.panels:
            if p.graph_id in self.project.graphs:
                self._sync_series(p.graph_id)
                s = self.project.graphs[p.graph_id]
                graphs[p.graph_id] = (s, self.project.tables[s.table_id], res, self.project.tables)
        lay = type(lay).from_dict({**lay.to_dict(), "panels": [p.to_dict() for p in lay.panels if p.graph_id in graphs]})
        return plot.build_layout(lay, graphs)

    def figure(self, item_id: str):
        """Matplotlib figure of a graph or a layout."""
        p = self.project
        if item_id in p.graphs:
            self._sync_series(item_id)
            s = p.graphs[item_id]
            return plot.build_figure(s, p.tables[s.table_id], self.results(), p.tables)
        if item_id in p.layouts:
            return self._layout_figure(p.layouts[item_id])
        raise ApiError(f"unknown graph or layout {item_id!r}; graphs: {sorted(p.graphs)}, layouts: {sorted(p.layouts)}")

    def preview(self, item_id: str, dpi: int = 150) -> bytes:
        """PNG bytes of a graph or layout."""
        buf = io.BytesIO()
        self.figure(item_id).savefig(buf, format="png", dpi=dpi)
        return buf.getvalue()

    def export(self, item_id: str, path, dpi: int = 600, text_as_paths: bool = False) -> str:
        """Write svg / pdf / png / tif / eps (by extension) of a graph or layout."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot.export(self.figure(item_id), path, dpi=dpi, text_as_paths=text_as_paths)
        except ValueError as e:
            raise ApiError(str(e)) from None
        return str(path)


def _grow(t: Table, n_total: int) -> None:
    add = n_total - t.n_rows
    if add <= 0:
        return
    for ds in t.datasets:
        ds.values = np.vstack([ds.values, np.full((add, ds.values.shape[1]), np.nan)])
        ds.excluded = np.vstack([ds.excluded, np.zeros((add, ds.values.shape[1]), bool)])
    if t.x is not None:
        t.x = np.concatenate([t.x, np.full(add, np.nan)])
    if t.kind in TITLE_KINDS:
        t.row_titles = list(t.row_titles) + [""] * (n_total - len(t.row_titles))
