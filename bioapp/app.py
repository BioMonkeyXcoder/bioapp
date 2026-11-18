"""Application state: project, undo stack, dirty flag, background analyses, autosave.

The UI talks only to ``App``. Every user edit goes through a method here that mutates the
project, bumps revisions, pushes a QUndoCommand and emits a change signal. Analyses are
re-run automatically (debounced) when their table or parameters change; a finished
result is adopted only if the (table revision, params revision) key is unchanged.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from . import analysis, plot
from .data import ONE_SUBCOLUMN_KINDS, Analysis, DataSet, Project, Table, Y_FORMATS, new_id
from .jobs import JobPool

NAN = float("nan")
RERUN_DELAY_MS = 300
AUTOSAVE_MS = 60_000
TITLE_KINDS = ("Grouped", "Contingency", "MultipleVariables")  # kinds with a row-title column
X_KINDS = ("XY", "Survival")                                    # kinds with an X (time) column
SET_NAMES = {"Column": "Group", "Survival": "Group", "Nested": "Group", "Contingency": "Outcome"}


def recovery_dir() -> Path:
    base = os.environ.get("BIOAPP_RECOVERY_DIR")
    if base:
        return Path(base)
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "bioapp" / "recovery"


# ---------------------------------------------------------------- table helpers (pure data)

def sub_labels(t: Table, n_sub: int) -> list[str]:
    fields = Y_FORMATS[t.y_format]
    return [f.upper() if f in ("sd", "sem") else f.capitalize() for f in fields] if fields else [f"Y{i + 1}" for i in range(n_sub)]


def table_columns(t: Table) -> list[tuple]:
    """Editable columns in display order: ("x",) / ("title",) first, then ("v", ds_index, sub)."""
    cols: list[tuple] = []
    if t.kind in X_KINDS:
        cols.append(("x",))
    elif t.kind in TITLE_KINDS:
        cols.append(("title",))
    for di, ds in enumerate(t.datasets):
        cols += [("v", di, s) for s in range(ds.values.shape[1])]
    return cols


def make_table(kind: str, y_format: str = "replicates", n_sub: int = 3, n_datasets: int = 2,
               n_rows: int = 10, title: str | None = None, sub_titles: list[str] | None = None,
               variables: list[tuple[str, bool]] | None = None) -> Table:
    """``n_datasets``: data sets / groups / outcome columns / variables. Nested: ``n_sub`` subjects per group titled
    ``sub_titles``. MultipleVariables: ``variables`` = [(title, categorical)] replaces ``n_datasets``."""
    if kind in ONE_SUBCOLUMN_KINDS + ("Nested",):
        y_format = "replicates"
    fields = Y_FORMATS[y_format]
    n_sub = len(fields) if fields else (1 if kind == "Column" or kind in ONE_SUBCOLUMN_KINDS else max(1, n_sub))
    t = Table(new_id("tbl"), title or f"{kind} table", kind, y_format, x=np.full(n_rows, NAN) if kind in X_KINDS else None,
              x_title="Time" if kind == "Survival" else "X", row_titles=[""] * n_rows if kind in TITLE_KINDS else [])
    if kind == "MultipleVariables":
        for name, cat in variables or [(f"Variable {i + 1}", False) for i in range(n_datasets)]:
            t.datasets.append(DataSet(new_id("ds"), name, np.full((n_rows, 1), NAN), categories=[] if cat else None))
    else:
        for _ in range(max(0, n_datasets)):
            _add_dataset(t, n_sub, n_rows=n_rows)
    if kind == "Nested":
        subs = [s.strip() for s in (sub_titles or [])] + [""] * n_sub
        for ds in t.datasets:
            ds.sub_titles = [s or f"Subject {j + 1}" for j, s in enumerate(subs[:n_sub])]
    return t


def _add_dataset(t: Table, n_sub: int | None = None, title: str | None = None, n_rows: int | None = None) -> None:
    if n_sub is None:
        fields = Y_FORMATS[t.y_format]
        n_sub = len(fields) if fields else (1 if t.kind == "Column" or t.kind in ONE_SUBCOLUMN_KINDS
                                            else t.datasets[0].values.shape[1] if t.datasets else 3)
    i = len(t.datasets)
    base = SET_NAMES.get(t.kind)
    name = title or (f"Variable {i + 1}" if t.kind == "MultipleVariables" else
                     f"{base} {chr(65 + i % 26)}" if base else f"Data Set-{chr(65 + i % 26)}")
    rows = n_rows if n_rows is not None else (t.n_rows or len(t.row_titles))
    subs = list(t.datasets[0].sub_titles) if t.kind == "Nested" and t.datasets else []
    if t.kind == "Nested" and len(subs) != n_sub:
        subs = [f"Subject {j + 1}" for j in range(n_sub)]
    t.datasets.append(DataSet(new_id("ds"), name, np.full((rows, n_sub), NAN), sub_titles=subs))


def _grow_rows(t: Table, n_total: int) -> None:
    add = n_total - t.n_rows
    if add <= 0:
        return
    for ds in t.datasets:
        ds.values = np.vstack([ds.values, np.full((add, ds.values.shape[1]), NAN)])
        ds.excluded = np.vstack([ds.excluded, np.zeros((add, ds.values.shape[1]), bool)])
    if t.x is not None:
        t.x = np.concatenate([t.x, np.full(add, NAN)])
    if t.kind in TITLE_KINDS:
        t.row_titles = list(t.row_titles) + [""] * (n_total - len(t.row_titles))


def valid_cell(t: Table, col: tuple, v) -> bool:
    """False when ``v`` cannot be stored in this cell: survival codes are 0 / 1, contingency counts non-negative integers."""
    if col[0] != "v" or isinstance(v, str) or math.isnan(v):
        return True
    if t.kind == "Survival":
        return v in (0.0, 1.0)
    return t.kind != "Contingency" or (v >= 0 and v == round(v))


def _apply_cells(t: Table, edits: list[tuple]) -> list[tuple]:
    """Apply (row, col, value) edits; return the inverse edits (for undo). A text value in a categorical
    data set is a label (a new label extends ``categories``; the inverse restores the old list)."""
    inv = []
    for row, col, val in edits:
        k = col[0]
        if k == "x":
            old, t.x[row] = float(t.x[row]), val
        elif k == "xtitle":
            old, t.x_title = t.x_title, val
        elif k == "title":
            t.row_titles += [""] * (max(row + 1, t.n_rows) - len(t.row_titles))
            old, t.row_titles[row] = t.row_titles[row], val
        elif k == "dtitle":
            old, t.datasets[col[1]].title = t.datasets[col[1]].title, val
        elif k == "stitle":
            ds = t.datasets[col[1]]
            ds.sub_titles = ds.sub_titles or [f"Subject {j + 1}" for j in range(ds.values.shape[1])]
            old, ds.sub_titles[col[2]] = ds.sub_titles[col[2]], val
        elif k == "cats":
            old, t.datasets[col[1]].categories = copy.copy(t.datasets[col[1]].categories), val
        elif k == "v":
            ds = t.datasets[col[1]]
            if isinstance(val, str):
                val = val.strip()
                if val and val not in ds.categories:
                    inv.append((-1, ("cats", col[1]), list(ds.categories)))
                    ds.categories.append(val)
                val = ds.categories.index(val) if val else NAN
            old, ds.values[row, col[2]] = float(ds.values[row, col[2]]), val
        else:  # "ex"
            ds = t.datasets[col[1]]
            old, ds.excluded[row, col[2]] = bool(ds.excluded[row, col[2]]), val
        inv.append((row, col, old))
    return inv[::-1]


def parse_number(s: str) -> float:
    s = s.strip()
    if not s:
        return NAN
    try:
        return float(s)
    except ValueError:
        return NAN


class _Cmd(QUndoCommand):
    def __init__(self, text, redo, undo, skip_first=False):
        super().__init__(text)
        self._redo, self._undo, self._skip = redo, undo, skip_first

    def redo(self):
        if self._skip:  # the edit was already applied when the command was created
            self._skip = False
        else:
            self._redo()

    def undo(self):
        self._undo()


class App(QObject):
    project_changed = Signal()        # items added / removed / renamed / project replaced
    table_changed = Signal(str)       # table id; any data or structure change
    analysis_changed = Signal(str)    # analysis id; result adopted or params edited
    graph_changed = Signal(str)       # graph id
    layout_changed = Signal(str)      # layout id
    dirty_changed = Signal(bool)
    status = Signal(str)              # job state text
    file_changed_on_disk = Signal(str)  # the open project file was rewritten by another program (e.g. the MCP server)

    def __init__(self, project: Project | None = None, n_workers: int = 2, parent=None):
        super().__init__(parent)
        self.project = project or Project()
        self.path: str | None = None
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(200)
        self.undo.indexChanged.connect(lambda _=None: self._sync_dirty())
        self._forced_dirty = False
        self._dirty = False
        self.pool = JobPool(n_workers, self)
        self.pool.done.connect(self._job_done)
        self.pool.state.connect(lambda *_: self._emit_status())
        self.job_target = "analysis"  # tests substitute a module-level "module:function"
        self._inflight: dict[str, tuple[int, int]] = {}
        self._pending: set[str] = set()
        self._rerun = QTimer(self, singleShot=True, interval=RERUN_DELAY_MS)
        self._rerun.timeout.connect(self._flush_pending)
        self._autosave = QTimer(self, interval=AUTOSAVE_MS)
        self._autosave.timeout.connect(self.autosave)
        self._autosave.start()
        self.table_changed.connect(self._on_table_changed)
        # ponytail: 1 s mtime polling instead of QFileSystemWatcher, which loses the path on atomic os.replace saves
        self._disk_mtime: int | None = None
        self._disk_poll = QTimer(self, interval=1000)
        self._disk_poll.timeout.connect(self._check_disk)
        self._disk_poll.start()

    def _file_mtime(self) -> int | None:
        try:
            return os.stat(self.path).st_mtime_ns if self.path else None
        except OSError:
            return None

    def _check_disk(self):
        m = self._file_mtime()
        if m is not None and self._disk_mtime is not None and m != self._disk_mtime:
            self._disk_mtime = m  # report each external write once
            self.file_changed_on_disk.emit(self.path)

    # ------------------------------------------------------------ dirty / undo plumbing
    @property
    def dirty(self) -> bool:
        return self._dirty

    def _sync_dirty(self):
        import shiboken6
        if not shiboken6.isValid(self.undo):  # the stack emits indexChanged while it is being destroyed at exit
            return
        d = self._forced_dirty or not self.undo.isClean()
        if d != self._dirty:
            self._dirty = d
            self.dirty_changed.emit(d)

    def mark_dirty(self):
        self._forced_dirty = True
        self._sync_dirty()

    def _mark_clean(self):
        self._forced_dirty = False
        self.undo.setClean()
        self._sync_dirty()

    def _push(self, text, redo, undo, skip_first=False):
        self.undo.push(_Cmd(text, redo, undo, skip_first))

    def _structure_edit(self, text: str, mutate) -> None:
        """Run ``mutate()`` and make it undoable by restoring shallow copies of the project dicts."""
        p = self.project
        names = ("tables", "analyses", "graphs", "layouts")
        before = {n: dict(getattr(p, n)) for n in names}
        mutate()
        after = {n: dict(getattr(p, n)) for n in names}

        def restore(snap):
            def f():
                for n in names:
                    d = getattr(p, n)
                    d.clear()
                    d.update(snap[n])
                for a in p.analyses.values():  # re-run anything whose table came back or changed
                    if a.table_id in p.tables and not a.is_current(p.tables[a.table_id]):
                        self.schedule(a.id)
                self.project_changed.emit()
            return f

        self._push(text, restore(after), restore(before), skip_first=True)
        self.project_changed.emit()

    # ------------------------------------------------------------ table edits
    def _table_snapshot(self, t: Table):
        return copy.deepcopy((t.title, t.x_title, t.y_format, t.datasets, t.x, t.row_titles))

    def _table_restore(self, t: Table, snap):
        t.title, t.x_title, t.y_format, t.datasets, t.x, t.row_titles = copy.deepcopy(snap)
        t.bump()
        self.table_changed.emit(t.id)
        self.project_changed.emit()

    def _table_structure_edit(self, tid: str, text: str, mutate) -> None:
        t = self.project.tables[tid]
        before = self._table_snapshot(t)
        mutate(t)
        after = self._table_snapshot(t)
        t.bump()
        self._push(text, lambda: self._table_restore(t, after), lambda: self._table_restore(t, before), skip_first=True)
        self.table_changed.emit(tid)
        self.project_changed.emit()

    def set_cells(self, tid: str, edits: list[tuple], text: str = "Edit cells") -> None:
        """edits = [(row, col, value)], col from ``table_columns`` or ("ex", ds, sub) / ("dtitle", ds)."""
        t = self.project.tables[tid]
        if not edits:
            return
        need = max(r for r, _, _ in edits) + 1
        if need > t.n_rows:  # growing the table is a structural edit
            self._table_structure_edit(tid, text, lambda t: (_grow_rows(t, need), _apply_cells(t, edits)))
            return
        inv = _apply_cells(t, edits)
        t.bump()
        self._push(text, lambda: (_apply_cells(t, edits), t.bump(), self.table_changed.emit(tid)),
                   lambda: (_apply_cells(t, inv), t.bump(), self.table_changed.emit(tid)), skip_first=True)
        self.table_changed.emit(tid)

    def paste(self, tid: str, row0: int, col0: int, text: str) -> int:
        """Paste tab-separated text at (row0, col0); grows rows and data sets. Returns the count of non-numeric cells left blank."""
        t = self.project.tables[tid]
        lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        grid = [ln.split("\t") for ln in lines]
        if not grid:
            return 0
        width = max(len(r) for r in grid)
        bad = 0

        def do(t):
            nonlocal bad
            bad = 0
            _grow_rows(t, row0 + len(grid))
            while len(table_columns(t)) < col0 + width:
                _add_dataset(t)
            cols = table_columns(t)
            edits = []
            for i, r in enumerate(grid):
                for j, s in enumerate(r):
                    if col0 + j >= len(cols):
                        break
                    col = cols[col0 + j]
                    if col[0] == "title" or (col[0] == "v" and t.datasets[col[1]].categories is not None):
                        edits.append((row0 + i, col, s))  # text cell: row title or categorical label
                    else:
                        v = parse_number(s)
                        if not valid_cell(t, col, v):
                            v = NAN
                        bad += int(math.isnan(v) and bool(s.strip()))
                        edits.append((row0 + i, col, v))
            _apply_cells(t, edits)

        self._table_structure_edit(tid, "Paste", do)
        return bad

    def toggle_exclusion(self, tid: str, cells: list[tuple[int, int, int]]) -> None:
        """cells = [(row, ds_index, sub)]; if any is not excluded, exclude all, else include all."""
        t = self.project.tables[tid]
        cells = [(r, d, s) for r, d, s in cells if r < t.n_rows]
        if not cells:
            return
        new = not all(t.datasets[d].excluded[r, s] for r, d, s in cells)
        self.set_cells(tid, [(r, ("ex", d, s), new) for r, d, s in cells], "Exclude cells" if new else "Include cells")

    def add_rows(self, tid: str, n: int = 1, at: int | None = None) -> None:
        def do(t):
            at_ = t.n_rows if at is None else at
            old = t.n_rows
            _grow_rows(t, old + n)
            if at_ < old:  # move the appended blank rows to position `at_`
                order = list(range(at_)) + list(range(old, old + n)) + list(range(at_, old))
                _reorder(t, order)
        self._table_structure_edit(tid, "Add rows", do)

    def remove_rows(self, tid: str, rows: list[int]) -> None:
        t = self.project.tables[tid]
        keep = [r for r in range(t.n_rows) if r not in set(rows)]
        if len(keep) != t.n_rows:
            self._table_structure_edit(tid, "Remove rows", lambda t: _reorder(t, keep))

    def add_dataset(self, tid: str) -> None:
        self._table_structure_edit(tid, "Add data set", _add_dataset)

    def remove_datasets(self, tid: str, idxs: list[int]) -> None:
        def do(t):
            t.datasets = [d for i, d in enumerate(t.datasets) if i not in set(idxs)]
        self._table_structure_edit(tid, "Remove data sets", do)

    def rename_dataset(self, tid: str, di: int, title: str) -> None:
        self.set_cells(tid, [(-1, ("dtitle", di), title)], "Rename data set")

    def set_variable_type(self, tid: str, di: int, categorical: bool) -> None:
        """MultipleVariables: switch one variable numeric <-> categorical (codes follow the sorted values / the labels)."""
        if self.project.tables[tid].kind != "MultipleVariables":
            raise ValueError("only MultipleVariables tables have numeric and categorical variables")

        def do(t):
            ds = t.datasets[di]
            if (ds.categories is not None) == categorical:
                return
            v, blank = ds.values, np.isnan(ds.values)
            if categorical:
                u = np.unique(v[~blank])
                ds.categories = [format(x, ".10g") for x in u]
                ds.values = np.where(blank, NAN, np.searchsorted(u, v))
            else:
                nums = [parse_number(c) for c in ds.categories]
                lut = np.array((nums if not any(map(math.isnan, nums)) else list(range(len(nums)))) + [NAN], float)
                ds.values = lut[np.where(blank, len(lut) - 1, v).astype(int)]
                ds.categories = None
        self._table_structure_edit(tid, "Switch variable type", do)

    # ------------------------------------------------------------ project items
    def add_table(self, t: Table, text: str = "New table") -> None:
        self._structure_edit(text, lambda: self.project.tables.__setitem__(t.id, t))

    def rename(self, section: str, item_id: str, title: str) -> None:
        attr = "name" if section == "graphs" else "title"  # a graph's title is drawn on the figure; its name is not
        old = getattr(getattr(self.project, section)[item_id], attr)
        if title == old:
            return

        def setter(v):
            def f():
                setattr(getattr(self.project, section)[item_id], attr, v)  # looked up again: undo may have swapped the spec
                self.project_changed.emit()
            return f
        setter(title)()
        self._push("Rename", setter(title), setter(old), skip_first=True)

    def delete_table(self, tid: str) -> None:
        p = self.project

        def mutate():
            doomed = {tid}
            changed = True
            while changed:  # analyses on a deleted table go; so do their output tables
                changed = False
                for a in list(p.analyses.values()):
                    if a.table_id in doomed and a.id not in doomed:
                        doomed.add(a.id)
                        changed = True
                        if a.output_table_id and a.output_table_id not in doomed:
                            doomed.add(a.output_table_id)
            plot.prune_trace_graphs(p.graphs, doomed)  # trace graphs keep the series of their remaining tables
            for d in (p.tables, p.analyses):
                for k in [k for k in d if k in doomed]:
                    del d[k]
            for gid in [g for g, s in p.graphs.items() if s.table_id in doomed or self._needs_result(s, doomed)]:
                del p.graphs[gid]
        self._structure_edit("Delete table", mutate)

    @staticmethod
    def _needs_result(spec, aids) -> bool:
        """A result-driven graph (not an xy graph, which also draws without its result) of a deleted analysis."""
        return spec.analysis_id in aids and spec.kind != "xy"

    def delete_item(self, section: str, item_id: str) -> None:
        if section == "tables":
            return self.delete_table(item_id)

        def mutate():
            d = getattr(self.project, section)
            a = d.pop(item_id, None)
            if section == "analyses":
                for gid in [g for g, s in self.project.graphs.items() if self._needs_result(s, {item_id})]:
                    del self.project.graphs[gid]
                if a is not None and a.output_table_id:
                    self.project.tables.pop(a.output_table_id, None)
                    for gid in [g for g, s in self.project.graphs.items() if s.table_id == a.output_table_id]:
                        del self.project.graphs[gid]
        self._structure_edit("Delete", mutate)

    # ------------------------------------------------------------ analyses
    def add_analysis(self, table_id: str, kind: str, dataset_ids: list[str], params: dict, title: str | None = None) -> Analysis:
        spec = analysis.all_kinds()[kind]
        a = Analysis(new_id("an"), title or f"{spec.label} of {self.project.tables[table_id].title}", kind, table_id,
                     list(dataset_ids), dict(params))
        self._structure_edit("Analyze", lambda: self.project.analyses.__setitem__(a.id, a))
        self.schedule(a.id)
        return a

    def edit_analysis(self, aid: str, params: dict | None = None, dataset_ids: list[str] | None = None) -> None:
        a = self.project.analyses[aid]
        old = (copy.deepcopy(a.params), list(a.dataset_ids))
        new = (copy.deepcopy(params if params is not None else a.params),
               list(dataset_ids if dataset_ids is not None else a.dataset_ids))

        def setter(v):
            def f():
                a.params, a.dataset_ids = copy.deepcopy(v[0]), list(v[1])
                a.bump_params()
                self.schedule(aid)
                self.analysis_changed.emit(aid)
            return f
        setter(new)()
        self._push("Edit analysis", setter(new), setter(old), skip_first=True)

    def is_stale(self, aid: str) -> bool:
        a = self.project.analyses[aid]
        t = self.project.tables.get(a.table_id)
        return t is None or not a.is_current(t)

    def schedule(self, aid: str) -> None:
        """Queue an analysis for a (debounced) re-run."""
        self._pending.add(aid)
        self._rerun.start()
        self._emit_status()

    def _on_table_changed(self, tid: str) -> None:
        for a in self.project.analyses.values():
            if a.table_id == tid:
                self.schedule(a.id)
                self.analysis_changed.emit(a.id)

    def schedule_stale(self) -> None:
        for a in self.project.analyses.values():
            t = self.project.tables.get(a.table_id)
            if t is not None and not a.is_current(t):
                self.schedule(a.id)

    def _flush_pending(self) -> None:
        pending, self._pending = sorted(self._pending), set()
        for aid in pending:
            a = self.project.analyses.get(aid)
            t = self.project.tables.get(a.table_id) if a else None
            if t is None or aid in self._inflight:
                continue  # an in-flight run re-schedules itself when it finishes stale
            self._inflight[aid] = a.key(t)
            a.status = "running"
            self.pool.submit(aid, self.job_target, (a.kind, t, list(a.dataset_ids), dict(a.params)))
            self.analysis_changed.emit(aid)
        self._emit_status()

    def cancel_jobs(self) -> None:
        """Cancel running and queued analyses; they keep a failed 'cancelled' result until edited."""
        for aid in list(self._inflight):
            self.pool.cancel(aid)
            self._finish(aid, self._inflight.pop(aid), False, "Cancelled by the user.")
        self._pending.clear()
        self._emit_status()

    def _job_done(self, aid: str, ok: bool, res) -> None:
        key = self._inflight.pop(aid, None)
        if key is not None:
            self._finish(aid, key, ok, res)
        self._emit_status()

    def _finish(self, aid: str, key, ok: bool, res) -> None:
        a = self.project.analyses.get(aid)
        t = self.project.tables.get(a.table_id) if a else None
        if t is None:
            return
        if a.key(t) != key:  # edited while running: drop the result and run again
            self.schedule(aid)
            return
        if not ok:
            res = {"kind": a.kind, "status": "failed", "messages": [str(res)], "display": []}
        out = res.pop("output_table", None)
        a.result, a.result_key, a.status = res, key, res.get("status", "ok")
        if out is not None:
            self._adopt_output(a, out)
        self.analysis_changed.emit(aid)

    def _adopt_output(self, a: Analysis, out: Table) -> None:
        cur = self.project.tables.get(a.output_table_id) if a.output_table_id else None
        if cur is None:
            out.source = a.id
            a.output_table_id = out.id
            self.project.tables[out.id] = out
            self.project_changed.emit()
            self.table_changed.emit(out.id)
            return
        title, rev = cur.title, cur.revision
        cur.__dict__.update({**out.__dict__, "id": cur.id, "title": title, "revision": rev + 1, "source": a.id})
        self.table_changed.emit(cur.id)

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Spin the event loop until no analysis is pending or running (tests, --selftest)."""
        import time
        from PySide6.QtCore import QCoreApplication
        end = time.time() + timeout
        while (self._pending or self._inflight or self._rerun.isActive()) and time.time() < end:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        return not (self._pending or self._inflight)

    def _emit_status(self) -> None:
        run, queued = self.pool.counts()
        wait = len(self._pending)
        if not (run or queued or wait):
            self.status.emit("Idle")
        else:
            self.status.emit(f"Computing: {run} running, {queued + wait} waiting")

    # ------------------------------------------------------------ graphs and layouts
    def results(self) -> dict[str, dict]:
        return {a.id: a.result for a in self.project.analyses.values() if a.result}

    def graph_title(self, gid: str) -> str:
        spec = self.project.graphs[gid]
        t = self.project.tables.get(spec.table_id)
        what = "Graph" if spec.kind in ("xy", "column", "grouped") else spec.kind.replace("_", " ").capitalize()
        return spec.name or spec.title or f"{what} of {t.title if t else '?'}"

    def graph_choices(self, table_id: str) -> list[tuple[str, str, str | None]]:
        """(label, plot kind, analysis id) of the graphs "New graph" offers for a table."""
        res = {a.id: (a.title, a.result) for a in self.project.analyses.values() if a.table_id == table_id and a.result}
        return plot.graph_choices(self.project.tables[table_id], res)

    def _linked_spec(self, t: Table, gid: str, analysis_id: str | None = None, kind: str | None = None):
        """Default spec of a table; unless it is fed by one result, it draws the table's most recent fit (curves)
        and most recent comparison test (brackets for the significant comparisons)."""
        spec = plot.default_spec(t, gid, analysis_id, kind)
        if analysis_id is None and t.kind != "MultipleVariables":
            plot.auto_link(spec, [(a.id, a.kind, a.result) for a in self.project.analyses.values() if a.table_id == t.id])
        return spec

    def new_graph(self, table_id: str, kind: str | None = None, analysis_id: str | None = None) -> str:
        spec = self._linked_spec(self.project.tables[table_id], new_id("gr"), analysis_id, kind)
        self._structure_edit("New graph", lambda: self.project.graphs.__setitem__(spec.id, spec))
        return spec.id

    def graph_result(self, aid: str) -> list[str]:
        """"Graph this result": add the natural default graph(s) of an analysis; returns their ids ([] if it has none)."""
        a = self.project.analyses[aid]
        specs = plot.specs_for_result(self.project.tables[a.table_id], aid, a.kind, a.result or {}, lambda k: new_id("gr"))
        out = self.project.tables.get(a.output_table_id or "")
        if specs and out is not None:  # a transform: the graph shows its output table
            specs = [self._linked_spec(out, specs[0].id)]
        if specs:
            self._structure_edit("Graph this result", lambda: self.project.graphs.update({s.id: s for s in specs}))
        return [s.id for s in specs]

    def new_layout(self) -> str:
        lay = plot.LayoutSpec(new_id("lay"), f"Layout {len(self.project.layouts) + 1}")
        self._structure_edit("New layout", lambda: self.project.layouts.__setitem__(lay.id, lay))
        return lay.id

    def edit_spec(self, section: str, item_id: str, mutate, text: str = "Edit graph") -> None:
        """Undoable edit of a PlotSpec / LayoutSpec: ``mutate(spec)`` changes it in place; no re-analysis."""
        d = getattr(self.project, section)
        before = d[item_id].to_dict()
        mutate(d[item_id])
        after = d[item_id].to_dict()
        if before == after:
            return
        sig = self.graph_changed if section == "graphs" else self.layout_changed

        def setter(snap):
            def f():
                d[item_id] = type(d[item_id]).from_dict(copy.deepcopy(snap))
                sig.emit(item_id)
            return f
        self._push(text, setter(after), setter(before), skip_first=True)
        sig.emit(item_id)

    def edit_graph(self, gid: str, mutate, text: str = "Edit graph") -> None:
        self.edit_spec("graphs", gid, mutate, text)

    def edit_layout(self, lid: str, mutate, text: str = "Edit layout") -> None:
        self.edit_spec("layouts", lid, mutate, text)

    def sync_series(self, gid: str) -> None:
        """Give data sets added after the graph was made a default style (silent, not an undo step)."""
        spec, t = self.project.graphs[gid], self.project.tables.get(self.project.graphs[gid].table_id)
        if t is None or t.kind == "MultipleVariables":  # its graphs pick their variables themselves
            return
        have = {s.dataset_id for s in spec.series}
        spec.series += [s for s in plot.default_series(t) if s.dataset_id not in have]

    def graph_figure(self, gid: str):
        self.sync_series(gid)
        spec = self.project.graphs[gid]
        return plot.build_figure(spec, self.project.tables[spec.table_id], self.results(), self.project.tables)

    def layout_figure(self, lid: str):
        lay, res = self.project.layouts[lid], self.results()
        graphs = {}
        for p in lay.panels:
            if p.graph_id in self.project.graphs:
                self.sync_series(p.graph_id)
                s = self.project.graphs[p.graph_id]
                graphs[p.graph_id] = (s, self.project.tables[s.table_id], res, self.project.tables)
        lay = type(lay).from_dict({**lay.to_dict(), "panels": [p.to_dict() for p in lay.panels if p.graph_id in graphs]})
        return plot.build_layout(lay, graphs)

    def export_figure(self, kind: str, item_id: str, path, dpi: int = 600, text_as_paths: bool = False) -> None:
        fig = self.graph_figure(item_id) if kind == "graph" else self.layout_figure(item_id)
        plot.export(fig, path, dpi=dpi, text_as_paths=text_as_paths)

    def batch_export(self, folder, fmt: str, dpi: int = 600, text_as_paths: bool = False) -> list[tuple[str, bool, str]]:
        """Export every graph into ``folder``; returns (file name, ok, message) per graph, never stops at a failure."""
        folder, report, used = Path(folder), [], set()
        folder.mkdir(parents=True, exist_ok=True)
        for gid in self.project.graphs:
            stem = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in self.graph_title(gid)).strip() or gid
            while stem.lower() in used:
                stem += "_"
            used.add(stem.lower())
            name = f"{stem}.{fmt}"
            try:
                self.export_figure("graph", gid, folder / name, dpi, text_as_paths)
                report.append((name, True, "ok"))
            except Exception as exc:
                report.append((name, False, str(exc)))
        return report

    def add_tables(self, tables: list[Table], text: str = "Import") -> None:
        def mutate():
            for t in tables:
                self.project.tables[t.id] = t
        self._structure_edit(text, mutate)

    def new_table(self, kind: str, y_format: str = "replicates", n_sub: int = 3, n_datasets: int = 2, **opts) -> str:
        """``opts``: n_rows, sub_titles (Nested), variables (MultipleVariables); see ``make_table``."""
        t = make_table(kind, y_format, n_sub, n_datasets, title=f"Data {len(self.project.tables) + 1}", **opts)
        self.add_table(t)
        return t.id

    # ------------------------------------------------------------ project file
    def reset(self, project: Project, path: str | None = None) -> None:
        self.cancel_jobs()
        self.project, self.path = project, path
        self.undo.clear()
        for a in project.analyses.values():  # a project saved mid-run carries no running state
            if a.status == "running":
                a.status = a.result["status"] if a.result else "not_run"
        self._mark_clean()
        self._disk_mtime = self._file_mtime()
        self.project_changed.emit()
        self.schedule_stale()

    def new_project(self) -> None:
        self.reset(Project())

    def open_project(self, path: str) -> None:
        from . import project as project_io
        self.reset(project_io.load_project(path), str(path))

    def save(self, path: str | None = None) -> None:
        from . import project as project_io
        path = path or self.path
        if not path:
            raise ValueError("no file name")
        project_io.save_project(self.project, path)
        self.path = str(path)
        self._disk_mtime = self._file_mtime()  # our own write is not an external change
        self._mark_clean()
        self.discard_recovery()

    # ------------------------------------------------------------ autosave recovery
    def recovery_file(self) -> Path:
        return recovery_dir() / "recovery.biofig"

    def autosave(self) -> None:
        """Write a recovery copy (never the user's file) while the project has unsaved edits."""
        if not self._dirty:
            return
        from . import project as project_io
        try:
            recovery_dir().mkdir(parents=True, exist_ok=True)
            project_io.save_project(self.project, str(self.recovery_file()))
        except Exception as exc:  # a failed recovery copy must never disturb editing
            self.status.emit(f"Autosave failed: {exc}")

    def has_recovery(self) -> bool:
        return self.recovery_file().exists()

    def discard_recovery(self) -> None:
        try:
            self.recovery_file().unlink(missing_ok=True)
        except OSError:
            pass

    def recover(self) -> None:
        from . import project as project_io
        self.reset(project_io.load_project(str(self.recovery_file())), None)
        self.mark_dirty()  # the recovered content is not saved anywhere yet

    def shutdown(self) -> None:
        self._autosave.stop()
        self._rerun.stop()
        self.undo.blockSignals(True)  # the stack may emit while Qt tears the objects down
        self.pool.shutdown()


def _reorder(t: Table, order: list[int]) -> None:
    idx = np.asarray(order, dtype=int)
    for ds in t.datasets:
        ds.values, ds.excluded = ds.values[idx], ds.excluded[idx]
    if t.x is not None:
        t.x = t.x[idx]
    if t.kind in TITLE_KINDS:
        rt = list(t.row_titles) + [""] * (max(order, default=-1) + 1 - len(t.row_titles))
        t.row_titles = [rt[i] for i in order]





