"""Table editor: a QTableView over data.Table (no per-cell widgets, so 100k rows scroll smoothly)."""
from __future__ import annotations

import math

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QDialog, QFormLayout, QHeaderView, QInputDialog, QLineEdit, QMenu, QTableView, QVBoxLayout,
                               QWidget)

from ..app import App, parse_number, sub_labels, table_columns, valid_cell
from . import fmt as numfmt
from . import theme

EXCLUDED_COLOR = QColor(theme.NOTE)
ROW_HEADER = {"Grouped": "Row title", "Contingency": "Row", "MultipleVariables": "Observation"}
MIN_COL_PX = 76  # fits "-1234.56" plus padding
DATA_DIGITS = 6 # significant digits shown in a cell; editing and copying use full precision


def fmt(v: float) -> str:
    """Display text of a data cell."""
    return numfmt.fmt_data(v, DATA_DIGITS)


def _is_text(t, c) -> bool:
    """Cells that hold text: row titles and the labels of categorical variables."""
    return c[0] == "title" or (c[0] == "v" and t.datasets[c[1]].categories is not None)


class TableModel(QAbstractTableModel):
    def __init__(self, app: App, tid: str, parent=None):
        super().__init__(parent)
        self.app, self.tid = app, tid
        self._italic = QFont(QGuiApplication.font())
        self._italic.setItalic(True)
        self._rebuild()
        app.table_changed.connect(self._changed)

    def _rebuild(self):
        self.t = self.app.project.tables[self.tid]
        self.cols = table_columns(self.t)
        self.shape = (self.t.n_rows, len(self.cols))

    def _changed(self, tid):
        if tid != self.tid or tid not in self.app.project.tables:
            return
        old = self.shape
        self._rebuild()
        new = self.shape
        if old[1] != new[1] or new[0] < old[0]:
            self.beginResetModel()
            self.endResetModel()
            return
        if new[0] > old[0]:  # rows appended: keep selection and scroll position
            self.shape = old
            self.beginInsertRows(QModelIndex(), old[0] + 1, new[0])
            self.shape = new
            self.endInsertRows()
        n, m = new
        if m:
            self.dataChanged.emit(self.index(0, 0), self.index(n, m - 1))
            self.headerDataChanged.emit(Qt.Horizontal, 0, m - 1)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.shape[0] + 1  # one blank row to type into

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.shape[1]

    def _raw(self, row, col):
        k = col[0]
        if k == "x":
            return self.t.x[row]
        if k == "title":
            return self.t.row_titles[row] if row < len(self.t.row_titles) else ""
        ds = self.t.datasets[col[1]]
        v = ds.values[row, col[2]]
        if ds.categories is not None:  # the label of the code
            return "" if math.isnan(v) else ds.categories[int(v)]
        return v

    def data(self, index, role=Qt.DisplayRole):
        row, c = index.row(), self.cols[index.column()]
        if row >= self.shape[0]:
            return None
        if role == Qt.TextAlignmentRole:
            return (Qt.AlignLeft if _is_text(self.t, c) else Qt.AlignRight) | Qt.AlignVCenter
        if role in (Qt.DisplayRole, Qt.EditRole):
            v = self._raw(row, c)
            text = v if isinstance(v, str) else fmt(float(v)) if role == Qt.DisplayRole else numfmt.full(v)
            if role == Qt.DisplayRole and c[0] == "v" and self.t.datasets[c[1]].excluded[row, c[2]]:
                text += "*"
            return text
        if c[0] != "v" or not self.t.datasets[c[1]].excluded[row, c[2]]:
            return None
        if role == Qt.ForegroundRole:
            return EXCLUDED_COLOR
        if role == Qt.FontRole:
            return self._italic
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole:
            return False
        c = self.cols[index.column()]
        value = str(value)
        if not _is_text(self.t, c):
            if value.strip() and math.isnan(parse_number(value)):
                return False  # reject non-numeric text
            value = parse_number(value)
            if not valid_cell(self.t, c, value):
                return False  # survival code not 0 / 1, contingency count not a non-negative integer
        self.app.set_cells(self.tid, [(index.row(), c, value)])
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Vertical:
            return str(section + 1)
        c, t = self.cols[section], self.t
        if c[0] == "x":
            return t.x_title
        if c[0] == "title":
            return ROW_HEADER[t.kind]
        ds = t.datasets[c[1]]
        if t.kind == "MultipleVariables":
            return f"{ds.title}\n{'categorical' if ds.categories is not None else 'numeric'}"
        if t.kind == "Nested":  # group / subject (editable) / the values below
            sub = ds.sub_titles[c[2]] if c[2] < len(ds.sub_titles) else f"Subject {c[2] + 1}"
            return f"{ds.title}\n{sub}" if c[2] == 0 else f"\n{sub}"
        sub = sub_labels(t, ds.values.shape[1])[c[2]]
        return f"{ds.title}\n{sub}" if c[2] == 0 else f"\n{sub}"


class _Grid(QTableView):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):
            self.editor.copy()
        elif e.matches(QKeySequence.Paste):
            self.editor.paste()
        elif e.key() == Qt.Key_Delete and self.state() != QTableView.EditingState:
            self.editor.clear_cells()
        else:
            super().keyPressEvent(e)


class _TwoTitles(QDialog):
    """Group title and subject title of one Nested column."""

    def __init__(self, parent, group: str, subject: str):
        super().__init__(parent)
        self.setWindowTitle("Rename group and subject")
        self.group, self.subject = QLineEdit(group), QLineEdit(subject)
        box = QFormLayout()
        box.addRow("Group title", self.group)
        box.addRow("Subject title", self.subject)
        lay = QVBoxLayout(self)
        from .dialogs import _buttons, _card, _page
        _page(lay)
        lay.addWidget(_card(box))
        lay.addWidget(_buttons(self))


class TableEditor(theme.GlassCard):
    def __init__(self, app: App, tid: str, parent=None):
        super().__init__(pad=8, parent=parent)
        self.app, self.tid = app, tid
        self.model = TableModel(app, tid, self)
        self.view = _Grid(self)
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        self.view.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.view.verticalHeader().setDefaultSectionSize(22)
        self.view.horizontalHeader().setDefaultSectionSize(90)
        self.view.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.view.horizontalHeader().sectionDoubleClicked.connect(self._rename_header)
        self.view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.horizontalHeader().customContextMenuRequested.connect(self._header_menu)
        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        self.restyle()
        self._headers: list[str] = []
        self._fit_columns()
        self.model.modelReset.connect(lambda: self._fit_columns(True))
        self.model.headerDataChanged.connect(lambda *_: self._fit_columns())

    def _fit_columns(self, force: bool = False):
        """Column width = widest header line (or a 6-digit number); a manual width is kept until its header changes."""
        fm = QFontMetrics(self.view.horizontalHeader().font())
        heads = [str(self.model.headerData(j, Qt.Horizontal) or "") for j in range(self.model.columnCount())]
        for j, h in enumerate(heads):
            if force or j >= len(self._headers) or self._headers[j] != h:
                self.view.setColumnWidth(j, max(MIN_COL_PX, max(fm.horizontalAdvance(s) for s in h.split("\n")) + 24))
        self._headers = heads

    # ---- selection helpers
    def _cells(self):
        return [(i.row(), i.column()) for i in self.view.selectionModel().selectedIndexes() if i.row() < self.model.shape[0]]

    def _hull(self):
        sel = self.view.selectionModel().selectedIndexes()
        if not sel:
            return None
        r0, r1 = min(i.row() for i in sel), max(i.row() for i in sel)
        c0, c1 = min(i.column() for i in sel), max(i.column() for i in sel)
        return r0, r1, c0, c1

    # ---- edit actions
    def copy(self):
        h = self._hull()
        if h is None:
            return
        r0, r1, c0, c1 = h
        r1 = min(r1, self.model.shape[0] - 1)
        rows = ["\t".join(str(self.model.data(self.model.index(r, c), Qt.EditRole) or "") for c in range(c0, c1 + 1))
                for r in range(r0, r1 + 1)]
        QGuiApplication.clipboard().setText("\n".join(rows))

    def paste(self):
        h = self._hull()
        r0, c0 = (h[0], h[2]) if h else (0, 0)
        bad = self.app.paste(self.tid, r0, c0, QGuiApplication.clipboard().text())
        if bad:
            self.app.status.emit(f"{bad} pasted cell(s) that do not fit this table were left blank")

    def clear_cells(self):
        edits = []
        for r, c in self._cells():
            col = self.model.cols[c]
            edits.append((r, col, "" if col[0] == "title" else float("nan")))
        self.app.set_cells(self.tid, edits, "Clear cells")

    def toggle_exclusion(self):
        cells = [(r, self.model.cols[c][1], self.model.cols[c][2]) for r, c in self._cells() if self.model.cols[c][0] == "v"]
        self.app.toggle_exclusion(self.tid, cells)

    def add_rows(self):
        rows = sorted({r for r, _ in self._cells()})
        self.app.add_rows(self.tid, max(1, len(rows)), rows[0] if rows else None)

    def remove_rows(self):
        self.app.remove_rows(self.tid, sorted({r for r, _ in self._cells()}))

    def add_dataset(self):
        self.app.add_dataset(self.tid)

    def remove_datasets(self):
        idx = sorted({self.model.cols[c][1] for _, c in self._cells() if self.model.cols[c][0] == "v"}
                     | {self.model.cols[i.column()][1] for i in self.view.selectionModel().selectedColumns() if self.model.cols[i.column()][0] == "v"})
        self.app.remove_datasets(self.tid, idx)

    def toggle_variable_type(self):
        """MultipleVariables: switch the selected variables numeric <-> categorical."""
        t = self.model.t
        if t.kind != "MultipleVariables":
            return
        cols = {self.model.cols[c][1] for _, c in self._cells() if self.model.cols[c][0] == "v"}
        cols |= {self.model.cols[i.column()][1] for i in self.view.selectionModel().selectedColumns() if self.model.cols[i.column()][0] == "v"}
        for di in sorted(cols):
            self.app.set_variable_type(self.tid, di, t.datasets[di].categories is None)

    def _rename_header(self, section):
        c, t = self.model.cols[section], self.model.t
        if c[0] == "x":
            text, ok = QInputDialog.getText(self, "Rename X column", "Title:", text=t.x_title)
            if ok and text:
                self.app.set_cells(self.tid, [(-1, ("xtitle",), text)], "Rename X column")
            return
        if c[0] != "v":
            return
        ds = t.datasets[c[1]]
        if t.kind == "Nested":
            sub = ds.sub_titles[c[2]] if c[2] < len(ds.sub_titles) else f"Subject {c[2] + 1}"
            dlg = _TwoTitles(self, ds.title, sub)
            if dlg.exec():
                edits = [(-1, ("dtitle", c[1]), dlg.group.text()), (-1, ("stitle", c[1], c[2]), dlg.subject.text())]
                self.app.set_cells(self.tid, [e for e in edits if e[2].strip()], "Rename group and subject")
            return
        text, ok = QInputDialog.getText(self, "Rename data set", "Title:", text=ds.title)
        if ok and text:
            self.app.rename_dataset(self.tid, c[1], text)

    def _header_menu(self, pos):
        section = self.view.horizontalHeader().logicalIndexAt(pos)
        if section < 0:
            return
        m = QMenu(self)
        m.addAction("Rename...", lambda: self._rename_header(section))
        if self.model.t.kind == "MultipleVariables" and self.model.cols[section][0] == "v":
            ds = self.model.t.datasets[self.model.cols[section][1]]
            m.addAction("Switch to numeric" if ds.categories is not None else "Switch to categorical",
                        lambda: self.app.set_variable_type(self.tid, self.model.cols[section][1], ds.categories is None))
        m.exec(self.view.horizontalHeader().mapToGlobal(pos))
