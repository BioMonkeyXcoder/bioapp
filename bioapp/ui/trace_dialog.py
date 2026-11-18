"""Import dialog for instrument traces (chromatography UV / conductivity / gradient exports).

``open_trace_import(main_window)`` is the single entry point for the File menu: choose one or more files (or drop them
on the dialog), tick the detected traces, give each a legend label and a left / right Y axis, check the preview, and the
tables (one XY table per trace) and a "trace" graph are created in one undo step and the graph is selected.
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import io_traces, plot

FILE_FILTER = "Instrument exports (*.csv *.tsv *.txt);;All files (*)"
COLUMNS = ("Use", "Run", "Trace", "Legend label", "Unit", "Points", "Axis")


class TraceImportDialog(QDialog):
    """Detected traces with checkboxes, legend labels and axis choice; a preview of the graph that will be created."""

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import instrument traces")
        self.setAcceptDrops(True)
        self.resize(760, 640)
        self.files: list[Path] = []
        self.traces: list[io_traces.Trace] = []
        self.events: list[io_traces.Event] = []
        self._busy = False
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("background: #fde7e9; color: #7a0010; padding: 6px")
        self.error.hide()
        self.summary = QLabel()
        add = QPushButton("Add files...")
        add.clicked.connect(self._browse)
        top = QHBoxLayout()
        top.addWidget(self.summary, 1)
        top.addWidget(add)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.itemChanged.connect(self._item_changed)
        self._partner: dict[int, int] = {}  # BLI: row of a raw curve <-> row of its fit
        self.preview = QLabel("Drop files here or use Add files...")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.title = QLineEdit()
        self.title.setPlaceholderText("Graph title (optional)")
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        for w in (self.error, self.table, self.preview, self.title, self.buttons):
            lay.addWidget(w)
        lay.insertLayout(0, top)
        lay.setStretchFactor(self.table, 1)
        self.add_files(paths)

    # ------------------------------------------------------------ files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add instrument export files", "", FILE_FILTER)
        self.add_files(paths)

    def add_files(self, paths) -> None:
        """Read more files; unreadable ones are reported above the table and skipped."""
        errors: list[str] = []
        for p in (Path(p) for p in paths if Path(p) not in self.files):
            tr, ev = io_traces.read_files([p], errors)
            if not (tr or ev):
                continue
            known = {t.run for t in self.traces}
            for o in (*tr, *ev):  # a run name already in the list: tell the two apart
                if o.run in known:
                    o.run = f"{o.run} ({p.stem})"
            self.files.append(p)
            self.traces += tr
            self.events += ev
        self.error.setText("\n".join(errors))
        self.error.setVisible(bool(errors))
        self._fill()

    # ------------------------------------------------------------ table
    def _fill(self) -> None:
        self._busy = True
        right = io_traces.auto_right(self.traces)
        self.table.setRowCount(len(self.traces))
        for i, t in enumerate(self.traces):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Checked)
            cells = [use] + [QTableWidgetItem(s) for s in (t.run, t.name, "", io_traces.pretty_unit(t.unit_y), f"{len(t.x):,}")]
            for c, item in enumerate(cells):
                if c not in (0, 3):
                    item.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(i, c, item)
            combo = QComboBox()
            combo.addItems(["Left", "Right"])
            combo.setCurrentIndex(int(right[i]))
            combo.currentIndexChanged.connect(lambda _: self._refresh())
            self.table.setCellWidget(i, 6, combo)
        # legend labels: the trace name, the run when every run has one trace, else "run - name"
        for i, lab in enumerate(self._default_labels()):
            self.table.item(i, 3).setText(lab)
        self.table.resizeColumnsToContents()
        kinds = Counter(e.kind for e in self.events)
        bli = {id(t.bli): t.bli for t in self.traces if t.bli}
        rows = {}
        for i, t in enumerate(self.traces):
            rows.setdefault(id(t.bli), []).append(i)
        self._partner = {i: j for k, r in rows.items() if k in bli and len(r) == 2 for i, j in (r, r[::-1])}
        for i in self._partner:  # a BLI sensor is one choice: its fit curve follows the raw curve (and the reverse)
            self.table.cellWidget(i, 6).setEnabled(False)
        text = f"{len(self.files)} file(s), {len(self.traces)} traces"
        if bli:
            b = next(iter(bli.values())).boundary
            text += f", BLI kinetic data: {len(bli)} sensors (data + fit)" + (f", association / dissociation at {b:g} s" if b is not None else "")
        self.summary.setText(text + (f", {len(self.events)} marks ({', '.join(f'{k}: {n}' for k, n in kinds.items())})" if kinds else ""))
        self._busy = False
        self._refresh()

    def _item_changed(self, item) -> None:
        if not self._busy and item.column() == 0 and item.row() in self._partner:
            self._busy = True  # keep the pair together without recursing
            self.table.item(self._partner[item.row()], 0).setCheckState(item.checkState())
            self._busy = False
        self._refresh()

    def _default_labels(self) -> list[str]:
        runs, per_run = {t.run for t in self.traces}, Counter(t.run for t in self.traces)
        return [t.name if len(runs) == 1 else t.run if max(per_run.values()) == 1 else f"{t.run} - {t.name}"
                for t in self.traces]

    def chosen(self) -> list[tuple[io_traces.Trace, str, bool]]:
        """(trace, legend label, right axis) of every ticked row."""
        return [(t, self.table.item(i, 3).text().strip() or t.name, self.table.cellWidget(i, 6).currentIndex() == 1)
                for i, t in enumerate(self.traces) if self.table.item(i, 0).checkState() == Qt.Checked]

    def build(self):
        """(tables, PlotSpec) of the ticked traces."""
        sel = self.chosen()
        runs = {t.run for t, _, _ in sel}
        return io_traces.traces_to_graph([t for t, _, _ in sel], [e for e in self.events if e.run in runs],
                                         [r for _, _, r in sel], [lab for _, lab, _ in sel], self.title.text().strip())

    def _refresh(self) -> None:
        if self._busy:
            return
        ok = bool(self.chosen())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if not ok:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Tick at least one trace.")
            return
        try:
            tables, spec = self.build()
            spec.width_mm, spec.height_mm, spec.font_size_pt = 110.0, 52.0, 6.0
            fig = plot.build_figure(spec, tables[0], {}, {t.id: t for t in tables})  # thumbnail: pixel-column envelope
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            pm = QPixmap()
            pm.loadFromData(buf.getvalue(), "PNG")
            self.preview.setPixmap(pm)
        except Exception as exc:  # noqa: BLE001  a preview problem is shown, never raised into the event loop
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)  # what cannot be drawn cannot be imported
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"Cannot draw the preview: {exc}")


def open_trace_import(main_window, paths=None):
    """File menu entry: import instrument trace files into the open project. ``paths``: skip the file chooser.
    Creates one XY table per ticked trace and a trace graph (one undo step) and selects the graph; returns its id,
    or None when cancelled."""
    if paths is None:
        paths, _ = QFileDialog.getOpenFileNames(main_window, "Import instrument traces", "", FILE_FILTER)
    if not paths:
        return None
    dlg = TraceImportDialog(paths, main_window)
    if not dlg.exec():
        return None
    tables, spec = dlg.build()
    app = main_window.app

    def mutate():
        app.project.tables.update({t.id: t for t in tables})
        app.project.graphs[spec.id] = spec
    app._structure_edit("Import traces", mutate)  # one undo step for the tables and the graph
    main_window.select_item("graphs", spec.id)
    return spec.id
