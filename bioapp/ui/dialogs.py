"""Dialogs: new table (Prism-style welcome), import options, analyze / edit analysis, figure export."""
from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, QSettings, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QToolButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import analysis, io_tabular
from ..analysis import fit
from ..data import ONE_SUBCOLUMN_KINDS, TABLE_KINDS, DataSet, Table, Y_FORMATS, new_id
from . import theme
from .catalog import (CATEGORIES, EQUATION_TREE, EQUATIONS, KIND_CATEGORY, SECTION_OF, SECTION_ORDER, TABLE_KIND_LABEL,
                      X_SCALE_TEXT)

Y_FORMAT_LABELS = {"replicates": "Replicate values", "mean_sd": "Mean, SD", "mean_sem": "Mean, SEM",
                   "mean_sd_n": "Mean, SD, N", "mean_sem_n": "Mean, SEM, N"}
# per table kind: label of the data-set count, default number of rows, rows label
SET_LABEL = {"Survival": "Groups", "Nested": "Groups", "Contingency": "Outcome columns", "MultipleVariables": "Variables"}
ROWS_LABEL = {"Survival": "Subjects (rows)", "Nested": "Technical replicates (rows)", "Contingency": "Rows (groups)",
              "MultipleVariables": "Observations (rows)"}
DEFAULT_ROWS = {"Survival": 20, "Nested": 3, "Contingency": 2}


def _settings() -> QSettings:
    """Per-user settings; ``BIOAPP_SETTINGS_DIR`` redirects them to an ini file (tests, portable use)."""
    d = os.environ.get("BIOAPP_SETTINGS_DIR")
    return QSettings(str(Path(d) / "bioapp.ini"), QSettings.IniFormat) if d else QSettings("bioapp", "bioapp")


def _buttons(dlg: QDialog) -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    return box


def _card(layout, pad=14) -> theme.GlassCard:
    """A glass card (flat style: a plain container) holding ``layout``."""
    c = theme.GlassCard(pad=pad)
    c.setLayout(layout)
    return c


def _page(lay: QVBoxLayout) -> None:
    """Outer margins of a dialog page: the cards bring their own shadow band."""
    lay.setContentsMargins(10, 8, 10, 12)
    lay.setSpacing(8)


def _combo(items, current=None) -> QComboBox:
    c = QComboBox()
    c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)  # a long label must not widen the form
    c.setMinimumContentsLength(8)
    for value, label in items:
        c.addItem(label, value)
    if current is not None and c.findData(current) >= 0:
        c.setCurrentIndex(c.findData(current))
    return c


# ------------------------------------------------------------------ table kinds: schematics and sample data

KIND_INFO = {  # one-line description per table kind
    "XY": "One numeric X column and one or more Y data sets: dose-response, kinetics, standard curves.",
    "Column": "One group per column: compare groups with t tests, ANOVA and nonparametric tests.",
    "Grouped": "Two grouping variables (rows and data sets) with replicates: two-way ANOVA.",
    "Contingency": "Counts of subjects in each category: chi-square, Fisher, relative risk, odds ratio.",
    "Survival": "Time to event with censoring, one column per group: Kaplan-Meier, log-rank.",
    "Nested": "Subjects or experiments nested in groups, technical replicates in the rows.",
    "MultipleVariables": "One row per observation, one column per variable: regression, PCA, clustering.",
}
# (leading column header | None, [(group title, [subcolumn headers])])
SCHEMA = {
    "XY": ("X", [("A", ["Y1", "Y2", "Y3"]), ("B", ["Y1", "Y2", "Y3"])]),
    "Column": (None, [("A", [""]), ("B", [""]), ("C", [""])]),
    "Grouped": ("Row", [("A", ["1", "2", "3"]), ("B", ["1", "2", "3"])]),
    "Contingency": ("Row", [("Out A", [""]), ("Out B", [""])]),
    "Survival": ("Time", [("A", [""]), ("B", [""])]),
    "Nested": (None, [("A", ["S1", "S2", "S3"]), ("B", ["S1", "S2", "S3"])]),
    "MultipleVariables": ("Obs", [("Var 1", [""]), ("Var 2", [""]), ("Group", [""])]),
}
FILLS = (theme.SKY.base, theme.PEACH.base, theme.MINT.base)


def schematic(kind: str, w: int, h: int, text: bool = True) -> QPixmap:
    """Small drawing of a table's column layout (leading column, data sets, subcolumns)."""
    lead, groups = SCHEMA[kind]
    cols = ([("", theme.PANEL)] if lead else []) + [(s, FILLS[i % 3]) for i, (_, subs) in enumerate(groups) for s in subs]
    sub_row = any(s for _, subs in groups for s in subs)
    n_rows = 1 + sub_row + 4
    pm = QPixmap(w, h)
    pm.fill(QColor(theme.SURFACE))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    cw, ch = (w - 2) / len(cols), (h - 2) / n_rows
    f = QFont()
    f.setPixelSize(max(8, int(ch * 0.55)))
    p.setFont(f)
    line, ink = QPen(QColor(theme.BORDER_STRONG), 1), QPen(QColor(theme.TEXT))

    def cell(r, fill, label=""):
        p.setPen(line)
        p.setBrush(QColor(fill))
        p.drawRect(r)
        if text and label:
            p.setPen(ink)
            p.drawText(r, Qt.AlignCenter, label)
    x = 1.0
    heads = ([(lead, 1, theme.PANEL)] if lead else []) + [(t, len(subs), FILLS[i % 3]) for i, (t, subs) in enumerate(groups)]
    for title, span, fill in heads:  # header row: lead column and one merged cell per data set
        cell(QRectF(x, 1, cw * span, ch), fill, title)
        x += cw * span
    for j, (sub, fill) in enumerate(cols):
        for i in range(1, n_rows):  # sub-header row (when there is one) and data cells
            head = i == 1 and sub_row
            cell(QRectF(1 + j * cw, 1 + i * ch, cw, ch), fill if head else theme.SURFACE, sub if head else "")
    p.end()
    return pm


def _tbl(kind, title, sets, x=None, x_title="X", rows=None) -> Table:
    return Table(new_id("tbl"), title, kind, datasets=[DataSet(new_id("ds"), n, v, **kw) for n, v, kw in sets], x=x,
                 x_title=x_title, row_titles=rows or [])


def _sample_dose_response(rng):
    x = np.linspace(-9, -4, 11)
    sets = [(n, 5 + (top - 5) / (1 + 10 ** ((ec - x) * 1.1))[:, None] + rng.normal(0, 3, (11, 3)), {})
            for n, top, ec in (("Control", 100, -6.5), ("Treated", 85, -6.0))]
    return _tbl("XY", "Dose-response", sets, x, "log[agonist], M")


def _sample_standard_curve(rng):
    x = np.array([0, 1, 2, 4, 8, 16, 32], float)
    return _tbl("XY", "Standard curve", [("Signal", 0.05 + 0.03 * x[:, None] + rng.normal(0, 0.02, (7, 2)), {})], x, "Concentration")


def _sample_groups(names, means, n, rng):
    return _tbl("Column", "Column data", [(nm, rng.normal(m, 12, (n, 1)), {}) for nm, m in zip(names, means)])


def _sample_two_way(rng):
    t = np.array([0, 1, 2, 4], float)[:, None]
    sets = [(n, 20 + k * t + 3 * (j == "T") * t + rng.normal(0, 3, (4, 3)), {}) for n, j, k in (("Control", "C", 4), ("Treated", "T", 4))]
    return _tbl("Grouped", "Two-way", sets, rows=["0 h", "1 h", "2 h", "4 h"])


def _sample_contingency(rng):
    return _tbl("Contingency", "2 x 2 table", [("Disease", [[30.0], [10.0]], {}), ("No disease", [[20.0], [40.0]], {})],
                rows=["Exposed", "Unexposed"])


def _sample_survival(rng):
    n = 20
    t = np.round(np.r_[rng.exponential(10, n), rng.exponential(20, n)] + 0.5, 1)
    ev = (rng.random(2 * n) < 0.8).astype(float)
    sets = [(name, np.where(np.arange(2 * n) // n == g, ev, np.nan), {}) for g, name in enumerate(("Control", "Treated"))]
    return _tbl("Survival", "Survival", sets, t, "Days")


def _sample_nested(rng):
    subs = ["Subject 1", "Subject 2", "Subject 3"]
    sets = [(n, m + rng.normal(0, 4, (1, 3)) + rng.normal(0, 1, (3, 3)), {"sub_titles": subs}) for n, m in (("Control", 50), ("Treated", 58))]
    return _tbl("Nested", "Nested", sets)


def _sample_multiple(rng):
    n = 24
    age, dose, grp = rng.normal(50, 10, n), rng.uniform(0, 10, n), (np.arange(n) % 2).astype(float)
    resp = 5 + 0.8 * dose + 0.1 * age + 2 * grp + rng.normal(0, 1, n)
    sets = [("Age", age, {}), ("Dose", dose, {}), ("Response", resp, {}), ("Group", grp, {"categories": ["control", "drug"]})]
    return _tbl("MultipleVariables", "Multiple variables", sets, rows=[""] * n)


# kind -> [(label, builder(rng))]; a builder returns a fresh, validated Table
SAMPLES = {
    "XY": [("Dose-response, 3 replicates, two data sets", _sample_dose_response),
           ("Standard curve, duplicates", _sample_standard_curve)],
    "Column": [("Two groups (10 values each)", lambda r: _sample_groups(("Control", "Treated"), (100, 115), 10, r)),
               ("Four groups (8 values each)", lambda r: _sample_groups(("Control", "Dose 1", "Dose 2", "Dose 3"), (100, 108, 120, 135), 8, r))],
    "Grouped": [("Two-way: 2 treatments x 4 time points, triplicates", _sample_two_way)],
    "Contingency": [("2 x 2: exposure by disease", _sample_contingency)],
    "Survival": [("Two groups, 20 subjects each", _sample_survival)],
    "Nested": [("Two groups, 3 subjects, 3 technical replicates", _sample_nested)],
    "MultipleVariables": [("Four variables, one categorical", _sample_multiple)],
}


def make_sample(kind: str, index: int = 0) -> Table:
    """Sample table ``index`` of ``kind`` (deterministic values)."""
    return SAMPLES[kind][index][1](np.random.default_rng(2026))


class _KindRow(QWidget):
    """One row of the new-table kind list: small schematic, bold title, one-line description."""

    def __init__(self, kind: str):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        icon = QLabel()
        icon.setPixmap(schematic(kind, 56, 40, text=False))
        title = QLabel(f"<b>{TABLE_KIND_LABEL[kind]}</b>")
        desc = QLabel(KIND_INFO[kind])
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {theme.TEXT2}")
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(title)
        col.addWidget(desc)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(icon)
        lay.addLayout(col, 1)


class _TableOptions(QWidget):
    """Kind / y format / replicates controls shared by the new-table and import dialogs."""

    def __init__(self):
        super().__init__()
        self.kind = _combo([(k, TABLE_KIND_LABEL[k]) for k in TABLE_KINDS], "XY")
        self.y_format = _combo([(k, Y_FORMAT_LABELS[k]) for k in Y_FORMATS])
        self.replicates = QSpinBox(minimum=1, maximum=50, value=3)
        self.rep_label = QLabel("Replicates")
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        for label, w in (("Table type", self.kind), ("Y values", self.y_format), (self.rep_label, self.replicates)):
            self.form.addRow(label, w)
        self.kind.currentIndexChanged.connect(self._sync)
        self.y_format.currentIndexChanged.connect(self._sync)
        self._forced = False
        self._sync()

    def _sync(self):
        k = self.kind.currentData()
        if k in ONE_SUBCOLUMN_KINDS or k == "Nested":  # these kinds hold raw values only
            self.y_format.setCurrentIndex(self.y_format.findData("replicates"))
        self.y_format.setEnabled(k in ("XY", "Column", "Grouped"))
        raw = self.y_format.currentData() == "replicates"
        self.replicates.setEnabled(raw and k in ("XY", "Grouped", "Nested"))
        self.rep_label.setText("Subjects per group" if k == "Nested" else "Replicates")
        forced = raw and k in ("Column",) + ONE_SUBCOLUMN_KINDS
        if forced:
            self.replicates.setValue(1)
        elif self._forced and self.replicates.value() == 1:
            self.replicates.setValue(3)
        self._forced = forced


class NewTableDialog(QDialog):
    """Welcome dialog: pick a table kind (list with schematic and description), set its options (data sets / groups /
    outcome columns, rows, Nested subject titles, MultipleVariables variables) or start from sample data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New data table")
        self.resize(860, 600)
        self.opts = _TableOptions()
        self.opts.form.setRowVisible(self.opts.kind, False)  # the kind is chosen in the list; the combo keeps the value
        self.n_sets = QSpinBox(minimum=1, maximum=100, value=2)
        self.n_rows = QSpinBox(minimum=1, maximum=100_000, value=10)
        self.sets_label, self.rows_label = QLabel("Data sets"), QLabel("Rows")
        self.sub_titles = QLineEdit()
        self.sub_titles.setPlaceholderText("Subject 1, Subject 2, ... (comma-separated)")
        self.sub_label = QLabel("Subject titles")
        self.vars = QTableWidget(0, 2)
        self.vars.setHorizontalHeaderLabels(["Variable", "Type"])
        self.vars.horizontalHeader().setStretchLastSection(True)
        self.vars.setMaximumHeight(150)
        self.vars_label = QLabel("Variables")
        for label, w in ((self.sets_label, self.n_sets), (self.rows_label, self.n_rows), (self.sub_label, self.sub_titles),
                         (self.vars_label, self.vars)):
            self.opts.form.addRow(label, w)
        self.kind_list = QListWidget()
        self.kind_list.setFixedWidth(330)
        for k in TABLE_KINDS:
            it = QListWidgetItem()
            it.setSizeHint(QSize(300, 62))
            self.kind_list.addItem(it)
            self.kind_list.setItemWidget(it, _KindRow(k))
        self.picture = QLabel()
        self.picture.setFixedSize(300, 150)
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.samples = QListWidget()
        self.samples.setMinimumHeight(70)
        top = QHBoxLayout()
        top.addWidget(self.picture)
        top.addWidget(self.desc, 1, Qt.AlignTop)
        right = QVBoxLayout()
        right.addLayout(top)
        right.addWidget(QLabel("<b>Options</b>"))
        right.addWidget(self.opts)
        right.addWidget(QLabel("<b>Start with</b>"))
        right.addWidget(self.samples, 1)
        body = QHBoxLayout()
        pick = QVBoxLayout()
        pick.addWidget(self.kind_list)
        left = _card(pick, 8)
        left.set_inner_width(330)
        body.addWidget(left)
        body.addWidget(_card(right), 1)
        lay = QVBoxLayout(self)
        _page(lay)
        lay.addLayout(body)
        lay.addWidget(_buttons(self))
        self.opts.kind.currentIndexChanged.connect(self._kind_changed)
        self.kind_list.currentRowChanged.connect(self._row_picked)  # bound methods, not lambdas: a lambda still fires
        self.samples.currentRowChanged.connect(self._sample_picked)  # while the dialog is being destroyed
        self.n_sets.valueChanged.connect(self._fill_vars)
        self._kind_changed()

    def _row_picked(self, r: int):
        if r >= 0:
            self.opts.kind.setCurrentIndex(r)

    def _sample_picked(self, r: int):
        self.opts.setEnabled(r <= 0)  # a sample fixes the table shape

    @property
    def kind(self) -> str:
        return self.opts.kind.currentData()

    def _kind_changed(self):
        k = self.kind
        self.kind_list.blockSignals(True)
        self.kind_list.setCurrentRow(TABLE_KINDS.index(k))
        self.kind_list.blockSignals(False)
        self.picture.setPixmap(schematic(k, 298, 148))
        self.desc.setText(f"<b>{TABLE_KIND_LABEL[k]}</b><br>{KIND_INFO[k]}")
        self.sets_label.setText(SET_LABEL.get(k, "Data sets"))
        self.rows_label.setText(ROWS_LABEL.get(k, "Rows"))
        self.n_rows.setValue(DEFAULT_ROWS.get(k, 10))
        for w in (self.sub_label, self.sub_titles):
            w.setVisible(k == "Nested")
        for w in (self.vars_label, self.vars):
            w.setVisible(k == "MultipleVariables")
        if k == "MultipleVariables":
            self._fill_vars()
        self.samples.clear()
        self.samples.addItem("Blank table: enter or paste data")
        self.samples.addItems([label for label, _ in SAMPLES[k]])
        self.samples.setCurrentRow(0)

    def _fill_vars(self):
        """One row per variable; existing names and types survive a change of the count."""
        n, old = self.n_sets.value(), self.variables()
        self.vars.setRowCount(n)
        for i in range(n):
            self.vars.setItem(i, 0, QTableWidgetItem(old[i][0] if i < len(old) else f"Variable {i + 1}"))
            self.vars.setCellWidget(i, 1, _combo([(False, "Numeric"), (True, "Categorical")], old[i][1] if i < len(old) else False))

    def variables(self) -> list[tuple[str, bool]]:
        return [((self.vars.item(i, 0).text().strip() if self.vars.item(i, 0) else "") or f"Variable {i + 1}",
                 bool(self.vars.cellWidget(i, 1).currentData())) for i in range(self.vars.rowCount())]

    def values(self) -> dict:
        o = self.opts
        k = o.kind.currentData()
        v = dict(kind=k, y_format=o.y_format.currentData(), n_sub=o.replicates.value(), n_datasets=self.n_sets.value(),
                 n_rows=self.n_rows.value())
        if k == "Nested":
            v["sub_titles"] = [s.strip() for s in self.sub_titles.text().split(",")] if self.sub_titles.text().strip() else []
        if k == "MultipleVariables":
            v["variables"] = self.variables()
        return v

    def sample_table(self, title: str | None = None) -> Table | None:
        """The chosen sample-data table, or None when the blank table (``values()``) is wanted."""
        i = self.samples.currentRow()
        if i <= 0:
            return None
        t = make_sample(self.kind, i - 1)
        t.title = title or t.title
        return t


class ImportDialog(QDialog):
    """Options for a text / spreadsheet grid; accept only when the grid converts to a table."""

    def __init__(self, grid: list[list[str]], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import {title}")
        self.resize(620, 560)
        self.grid, self.title_, self.table, self.report = grid, title, None, []
        self.opts = _TableOptions()
        self.header_rows = QSpinBox(minimum=0, maximum=10, value=1)
        self.x_col = QSpinBox(minimum=1, maximum=1000, value=1)
        self.title_col = QSpinBox(minimum=0, maximum=1000, value=0, specialValueText="none")
        self.decimal = _combo([(".", "Point (1.5)"), (",", "Comma (1,5)")])
        for label, w in (("Header rows above the data (Nested: 2)", self.header_rows),
                         ("X / time column number (XY, Survival)", self.x_col),
                         ("Row-title column number (none = 0)", self.title_col), ("Decimal separator", self.decimal)):
            self.opts.form.addRow(label, w)
        s = _settings()  # last choices; a missing or stale value keeps the default
        for combo, key in ((self.opts.kind, "kind"), (self.opts.y_format, "y_format"), (self.decimal, "decimal")):
            i = combo.findData(s.value(f"import/{key}", "", type=str))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.opts.kind.currentIndexChanged.connect(self._sync)
        self._sync()
        prev = QTableWidget(min(len(grid), 8), len(grid[0]) if grid else 0)
        prev.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i, row in enumerate(grid[:8]):
            for j, v in enumerate(row):
                prev.setItem(i, j, QTableWidgetItem(v))
        self.error = QLabel()
        self.error.setStyleSheet(f"color: {theme.DANGER}")
        self.error.setWordWrap(True)
        lay = QVBoxLayout(self)
        _page(lay)
        lay.addWidget(QLabel(f"{len(grid)} rows x {len(grid[0]) if grid else 0} columns; first rows:"))
        lay.addWidget(prev)
        opts = QVBoxLayout()
        opts.addWidget(self.opts)
        lay.addWidget(_card(opts))
        lay.addWidget(self.error)
        lay.addWidget(_buttons(self))

    def _sync(self):
        k = self.opts.kind.currentData()
        self.x_col.setEnabled(k in ("XY", "Survival"))
        self.title_col.setEnabled(k in ("Grouped", "Contingency", "MultipleVariables"))
        if k == "Nested":  # row 1 names the groups, row 2 the subjects
            self.header_rows.setValue(max(2, self.header_rows.value()))

    def accept(self):
        o = self.opts
        try:
            self.table, self.report = io_tabular.grid_to_table(
                self.grid, o.kind.currentData(), header_rows=self.header_rows.value(), x_col=self.x_col.value() - 1,
                row_title_col=(self.title_col.value() - 1) if self.title_col.value() else None,
                replicates=o.replicates.value(), y_format=o.y_format.currentData(), decimal=self.decimal.currentData(),
                title=self.title_)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        s = _settings()
        for combo, key in ((o.kind, "kind"), (o.y_format, "y_format"), (self.decimal, "decimal")):
            s.setValue(f"import/{key}", combo.currentData())
        super().accept()


# ------------------------------------------------------------------ analyze

def _num(text: str, label: str, optional: bool):
    text = text.strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"{label}: a number is required")
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{label}: {text!r} is not a number") from None


class ConstraintsGrid(QTableWidget):
    """Per-parameter Fixed / Min / Max for the selected model."""

    def __init__(self, current: dict | None):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["Fixed", "Min", "Max"])
        self._saved: dict[str, list[str]] = {}
        for name, c in (current or {}).items():
            self._saved[name] = [str(c.get("fixed", "")), str(c.get("min", "")), str(c.get("max", ""))]
        self.names: list[str] = []

    def set_names(self, names):
        self._keep()
        self.names = list(names)
        self.setRowCount(len(self.names))
        self.setVerticalHeaderLabels(self.names)
        for i, n in enumerate(self.names):
            for j, v in enumerate(self._saved.get(n, ["", "", ""])):
                self.setItem(i, j, QTableWidgetItem(v))
        self.setFixedHeight(32 + 28 * len(self.names))

    def _keep(self):
        for i, n in enumerate(self.names):
            self._saved[n] = [(self.item(i, j).text() if self.item(i, j) else "") for j in range(3)]

    def values(self) -> dict:
        self._keep()
        out = {}
        for n in self.names:
            fx, lo, hi = self._saved[n]
            if fx.strip():
                out[n] = {"fixed": _num(fx, f"{n} fixed", False)}
            elif lo.strip() or hi.strip():
                out[n] = {k: _num(v, f"{n} {k}", False) for k, v in (("min", lo), ("max", hi)) if v.strip()}
        return out


class ConstantsGrid(QTableWidget):
    """Experimental constants of the selected model, one row per data set (params "dataset_constants")."""

    def __init__(self, table, current: dict | None):
        super().__init__(len(table.datasets), 0)
        self.ids = [d.id for d in table.datasets]
        self.setVerticalHeaderLabels([d.title for d in table.datasets])
        self._saved = {(i, n): str(v) for i, c in (current or {}).items() for n, v in c.items() if i in self.ids}
        self.names: list[str] = []
        self.setFixedHeight(min(32 + 28 * len(self.ids), 200))

    def set_names(self, names):
        self._keep()
        self.names = list(names)
        self.setColumnCount(len(self.names))
        self.setHorizontalHeaderLabels(self.names)
        for r, i in enumerate(self.ids):
            for c, n in enumerate(self.names):
                self.setItem(r, c, QTableWidgetItem(self._saved.get((i, n), "")))

    def _keep(self):
        for r, i in enumerate(self.ids):
            for c, n in enumerate(self.names):
                self._saved[(i, n)] = self.item(r, c).text() if self.item(r, c) else ""

    def values(self) -> dict:
        self._keep()
        out: dict = {}
        for (i, n), s in self._saved.items():
            if n in self.names and s.strip():
                out.setdefault(i, {})[n] = _num(s, f"{n} of data set {self.ids.index(i) + 1}", False)
        return out


class ModelPicker(QWidget):
    """Prism-style equation list: search box, tree by family, equation text and parameters of the selection.

    ``combo`` (hidden, one item per model) holds the value; the tree and the combo stay in sync.
    """

    def __init__(self, choices, current):
        super().__init__()
        self.combo = _combo(choices, current)
        self.combo.setParent(self)
        self.combo.hide()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search equations (name, parameter, family)")
        self.search.setClearButtonEnabled(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setWordWrap(True)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setMinimumHeight(260)
        self.tree.setItemDelegate(theme.CapsuleDelegate(self.tree))
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        self.info.setStyleSheet(theme.info_css())
        self._leaves: list[QTreeWidgetItem] = []
        for fam, subs in EQUATION_TREE.items():
            top = QTreeWidgetItem(self.tree, [fam])
            for sub, entries in subs.items():
                node = QTreeWidgetItem(top, [sub]) if sub else top
                for e in entries:
                    name, mid = e if isinstance(e, tuple) else (fit.MODELS[e].name, e)
                    leaf = QTreeWidgetItem(node, [name])
                    leaf.setData(0, Qt.UserRole, mid)
                    leaf.setToolTip(0, name)
                    self._leaves.append(leaf)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.search)
        lay.addWidget(self.tree)
        lay.addWidget(self.info)
        self.search.textChanged.connect(self._filter)
        self.tree.currentItemChanged.connect(self._picked)
        self.combo.currentIndexChanged.connect(self._synced)
        self._synced()

    def model_id(self) -> str:
        return self.combo.currentData()

    def _picked(self, item, _prev=None):
        mid = item.data(0, Qt.UserRole) if item is not None else None
        if mid is not None and mid != self.model_id():
            self.combo.setCurrentIndex(self.combo.findData(mid))

    def _synced(self, _=None):
        mid = self.model_id()
        cur = self.tree.currentItem()
        if cur is None or cur.data(0, Qt.UserRole) != mid:
            leaf = next(x for x in self._leaves if x.data(0, Qt.UserRole) == mid)
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(leaf)
            self.tree.blockSignals(False)
        self._expand_current()
        if self.tree.currentItem() is not None:
            self.tree.scrollToItem(self.tree.currentItem())
        self._show_info(fit.MODELS[mid])

    def _expand_current(self):
        item = self.tree.currentItem()
        while item is not None and item.parent() is not None:
            item = item.parent()
            item.setExpanded(True)

    def _filter(self, text: str):
        q = text.strip().lower()

        def visit(node) -> bool:
            if node.childCount() == 0:
                mid = node.data(0, Qt.UserRole)
                m = fit.MODELS[mid]
                path, n = [], node
                while n is not None:
                    path.append(n.text(0))
                    n = n.parent()
                hay = " ".join(path + list(m.params) + [m.name, mid]).lower()
                shown = all(w in hay for w in q.split())
            else:
                shown = any([visit(node.child(i)) for i in range(node.childCount())])
            node.setHidden(not shown)
            if node.childCount():
                node.setExpanded(bool(q) and shown)
            return shown
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
        if not q:
            self._expand_current()

    def _show_info(self, m):
        esc = lambda s: html.escape(s).replace("\n", "<br>")  # noqa: E731
        rows = [f"<b>{esc(m.name)}</b>",
                f"<span style=\"font-family:Consolas,'Courier New',monospace\">{esc(EQUATIONS[m.id])}</span>",
                f"<b>Parameters:</b> {esc(', '.join(m.params))}"]
        if m.constants:
            rows.append(f"<b>Constants (enter per data set below):</b> {esc(', '.join(m.constants))}")
        if m.title_const:
            rows.append(f"{esc(m.title_const)} is read from each data-set title (a number).")
        rows.append(esc(X_SCALE_TEXT[m.x_scale] + "."))
        if m.n_datasets:
            rows.append(f"Needs exactly {m.n_datasets} data sets, in the order of the equation.")
        elif m.min_datasets > 1:
            rows.append(f"Needs at least {m.min_datasets} data sets.")
        self.info.setText("<br>".join(rows))


class _Section(QFrame):
    """Titled, collapsible block of option rows (glass: an inner translucent card)."""

    def __init__(self, title: str):
        super().__init__()
        self.setProperty("glass", "inner")
        self.head = QToolButton(text=title, checkable=True, checked=True, autoRaise=True)
        self.head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.head.setArrowType(Qt.DownArrow)
        f = self.head.font()
        f.setBold(True)
        self.head.setFont(f)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {theme.BORDER}; border: none")
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(8, 0, 0, 6)
        self.head.toggled.connect(self._toggle)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 4)
        lay.setSpacing(0)
        top = QHBoxLayout()
        top.addWidget(self.head)
        top.addWidget(line, 1)
        lay.addLayout(top)
        lay.addWidget(self.body)

    def _toggle(self, on: bool):
        self.body.setVisible(on)
        self.head.setArrowType(Qt.DownArrow if on else Qt.RightArrow)


def _row(label: str, w: QWidget, wide: bool = False) -> QWidget:
    """Label (word-wrapped) left, widget right; ``wide``: label above a full-width widget."""
    row = QWidget()
    g = QGridLayout(row)
    g.setContentsMargins(0, 2, 0, 2)
    lab = QLabel(label)
    lab.setWordWrap(True)
    if wide:
        g.addWidget(lab, 0, 0)
        g.addWidget(w, 1, 0)
    else:
        g.addWidget(lab, 0, 0, Qt.AlignVCenter)
        g.addWidget(w, 0, 1, Qt.AlignVCenter)
        g.setColumnStretch(0, 5)
        g.setColumnStretch(1, 4)
    return row


def _plan(kind) -> list[tuple[str | None, list]]:
    """Option sections of a kind; forms with few options or a single section stay flat (title None)."""
    sec = [(SECTION_OF.get(p.name, "Fit"), p) for p in kind.params]
    if len(kind.params) < 6 or len({s for s, _ in sec}) < 2:
        return [(None, list(kind.params))]
    return [(s, [p for t, p in sec if t == s]) for s in SECTION_ORDER if any(t == s for t, _ in sec)]


class ParamForm(QWidget):
    """Form generated from the Param declarations of an AnalysisKind."""

    def __init__(self, kind, table, params: dict):
        super().__init__()
        self.kind, self._get = kind, {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 8, 0)
        constraints = model = cbody = None
        for title, plist in _plan(kind):
            body = lay
            if title:
                sec = _Section(title)
                lay.addWidget(sec)
                body = sec.body_lay
            for p in plist:
                val = params.get(p.name, p.default)
                wide = False
                if p.type == "choice" and p.name == "model" and any(q.type == "constraints" for q in kind.params):
                    w = ModelPicker(p.choices, val)
                    model, g, wide = w.combo, w.combo.currentData, True
                elif p.type == "choice":
                    w = _combo(p.choices, val)
                    g = w.currentData
                elif p.type in ("float", "float?"):
                    w = QLineEdit("" if val is None else repr(val))
                    g = (lambda w=w, p=p: _num(w.text(), p.label, p.type == "float?"))
                elif p.type == "int":
                    w = QSpinBox(minimum=-10**9, maximum=10**9, value=int(val or 0))
                    g = w.value
                elif p.type == "bool":
                    w = QCheckBox(checked=bool(val))
                    g = w.isChecked
                elif p.type == "str":
                    w = QLineEdit("" if val is None else str(val))
                    g = w.text
                elif p.type == "dataset":
                    w = _combo([(None, "(none)")] + [(d.id, d.title) for d in table.datasets], val)
                    g = w.currentData
                elif p.type == "datasets":
                    w = QListWidget()
                    for d in table.datasets:
                        it = QListWidgetItem(d.title)
                        it.setData(Qt.UserRole, d.id)
                        it.setCheckState(Qt.Checked if d.id in (val or []) else Qt.Unchecked)
                        w.addItem(it)
                    w.setMaximumHeight(120)
                    g = (lambda w=w: [w.item(i).data(Qt.UserRole) for i in range(w.count()) if w.item(i).checkState() == Qt.Checked])
                else:  # constraints
                    w = constraints = ConstraintsGrid(val)
                    g, wide, cbody = w.values, True, body
                self._get[p.name] = g
                body.addWidget(_row(p.label, w, wide))
        if constraints is not None:
            names = lambda: fit.MODELS[model.currentData()].params if model is not None else ()  # noqa: E731
            constraints.set_names(names())
            if model is not None:
                model.currentIndexChanged.connect(lambda _=None: constraints.set_names(names()))
        if kind.id == "nonlin_fit" and model is not None:  # constants of binding / enzyme / special models, per data set
            grid = ConstantsGrid(table, params.get("dataset_constants"))
            row = _row("Constants per data set", grid, True)
            (cbody or lay).addWidget(row)

            def show_constants(_=None):
                consts = fit.MODELS[model.currentData()].constants
                grid.set_names(consts)
                row.setVisible(bool(consts))
            model.currentIndexChanged.connect(show_constants)
            show_constants()
            self._get["dataset_constants"] = grid.values
        lay.addStretch(1)

    def values(self) -> dict:
        return {k: g() for k, g in self._get.items()}


class AnalyzeDialog(QDialog):
    """New analysis (``analysis_obj`` None) or edit of an existing one.

    Left: searchable analysis tree by Prism category (kinds the table cannot take are greyed with a tooltip) and the
    data-set checklist. Right: the options of the selected analysis."""

    def __init__(self, app, table_id: str | None = None, kind: str | None = None, analysis_obj=None, parent=None):
        super().__init__(parent)
        self.app, self.edit = app, analysis_obj
        self.setWindowTitle("Edit analysis" if analysis_obj else "Analyze data")
        self.resize(1000, 700)
        tables = app.project.tables
        self.table_combo = _combo([(t.id, t.title) for t in tables.values()], analysis_obj.table_id if analysis_obj else table_id)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search analyses")
        self.search.setClearButtonEnabled(True)
        self.kinds = QTreeWidget()
        self.kinds.setHeaderHidden(True)
        self.kinds.setIndentation(14)
        self.kinds.setItemDelegate(theme.CapsuleDelegate(self.kinds))
        self.ds_list = QListWidget()
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 700; font-size: 15pt")
        self.title_label.setWordWrap(True)
        self.form_area = QScrollArea(widgetResizable=True)
        self.form_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.form_area.setFrameShape(QFrame.NoFrame)
        self.error = QLabel()
        self.error.setStyleSheet(f"color: {theme.DANGER}")
        self.error.setWordWrap(True)
        self.form: ParamForm | None = None
        if analysis_obj:
            self.table_combo.setEnabled(False)
        top = QFormLayout()
        top.addRow("Data table", self.table_combo)
        left = QVBoxLayout()
        left.addWidget(self.search)
        left.addWidget(self.kinds, 3)
        left.addWidget(QLabel("Data sets to analyze"))
        left.addWidget(self.ds_list, 1)
        leftw = _card(left, 12)
        leftw.set_inner_width(300)
        right = QVBoxLayout()
        right.addWidget(self.title_label)
        right.addWidget(self.form_area, 1)
        mid = QHBoxLayout()
        mid.addWidget(leftw)
        mid.addWidget(_card(right, 12), 1)
        lay = QVBoxLayout(self)
        _page(lay)
        lay.addLayout(top)
        lay.addLayout(mid, 1)
        lay.addWidget(self.error)
        lay.addWidget(_buttons(self))
        self._want_kind = analysis_obj.kind if analysis_obj else kind
        self.table_combo.currentIndexChanged.connect(self._fill_kinds)
        self.kinds.currentItemChanged.connect(self._kind_picked)  # bound method: see NewTableDialog
        self.search.textChanged.connect(self._filter_kinds)
        self._fill_kinds()

    def _kind_picked(self, *_):
        self._fill_form()

    @property
    def table(self):
        return self.app.project.tables.get(self.table_combo.currentData())

    def _fill_kinds(self):
        t = self.table
        self.kinds.blockSignals(True)
        self.kinds.clear()
        self.ds_list.clear()
        if t is None:
            self.kinds.blockSignals(False)
            return
        cats = {c: QTreeWidgetItem(self.kinds, [c]) for c in CATEGORIES}
        for c in cats.values():
            c.setFlags(Qt.ItemIsEnabled)
            f = c.font(0)
            f.setBold(True)
            c.setFont(0, f)
        first = want = None
        for k in analysis.all_kinds().values():
            it = QTreeWidgetItem(cats[KIND_CATEGORY.get(k.id, CATEGORIES[1])], [k.label])
            it.setData(0, Qt.UserRole, k.id)
            if t.kind in k.table_kinds:
                first = first or it
                want = it if k.id == self._want_kind else want
            else:
                it.setFlags(Qt.NoItemFlags)
                it.setToolTip(0, f"Needs a {' or '.join(TABLE_KIND_LABEL[x] for x in k.table_kinds)} table; "
                                 f"'{t.title}' is a {TABLE_KIND_LABEL[t.kind]} table.")
        for c in cats.values():
            enabled = any(c.child(i).flags() & Qt.ItemIsEnabled for i in range(c.childCount()))
            c.setExpanded(enabled)
            if not enabled:
                c.setForeground(0, self.palette().placeholderText())
                c.setToolTip(0, f"No {c.text(0).lower()} are available for a {TABLE_KIND_LABEL[t.kind]} table.")
        self.kinds.setEnabled(self.edit is None)
        self.kinds.blockSignals(False)
        want_ids = set(self.edit.dataset_ids) if self.edit else {d.id for d in t.datasets if d.categories is None}
        for d in t.datasets:
            it = QListWidgetItem(d.title)
            it.setData(Qt.UserRole, d.id)
            it.setCheckState(Qt.Checked if d.id in want_ids else Qt.Unchecked)
            self.ds_list.addItem(it)
        self._filter_kinds(self.search.text())
        pick = want or first
        if pick is not None:
            self.kinds.setCurrentItem(pick)
            self._fill_form()

    def _filter_kinds(self, text: str = ""):
        q = text.strip().lower()
        for i in range(self.kinds.topLevelItemCount()):
            c = self.kinds.topLevelItem(i)
            shown = 0
            for j in range(c.childCount()):
                ch = c.child(j)
                hit = not q or all(w in f"{ch.text(0)} {ch.data(0, Qt.UserRole)}".lower() for w in q.split())
                ch.setHidden(not hit)
                shown += hit
            c.setHidden(shown == 0)
            if q and shown:
                c.setExpanded(True)

    def _fill_form(self):
        it = self.kinds.currentItem()
        if it is None or self.table is None or it.data(0, Qt.UserRole) is None:
            return
        self._want_kind = it.data(0, Qt.UserRole)
        kind = analysis.all_kinds()[self._want_kind]
        params = self.edit.params if self.edit and self.edit.kind == kind.id else {}
        self.title_label.setText(f"{kind.label}   <span style='font-size:10.5pt;color:{theme.TEXT2};font-weight:normal'>"
                                 f"{KIND_CATEGORY.get(kind.id, '')}</span>")
        self.form = ParamForm(kind, self.table, params)
        self.form_area.setWidget(self.form)

    def selected_datasets(self) -> list[str]:
        return [self.ds_list.item(i).data(Qt.UserRole) for i in range(self.ds_list.count())
                if self.ds_list.item(i).checkState() == Qt.Checked]

    def accept(self):
        if self.form is None:
            return
        try:
            params = self.form.values()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        ids = self.selected_datasets()
        if not ids:
            self.error.setText("Select at least one data set.")
            return
        if self.edit:
            self.app.edit_analysis(self.edit.id, params, ids)
        else:
            self.app.add_analysis(self.table.id, self._want_kind, ids, params)
        super().accept()


# ------------------------------------------------------------------ export

EXPORT_FORMATS = ("svg", "pdf", "png", "tiff", "eps")
EXPORT_LABELS = {"svg": "SVG (vector)", "pdf": "PDF (vector)", "png": "PNG (raster)", "tiff": "TIFF (raster)",
                 "eps": "EPS (vector)"}
RASTER = ("png", "tiff")


class ExportDialog(QDialog):
    """Single-file or batch (folder) figure export options; remembers the last folder, format and resolution."""

    def __init__(self, parent=None, batch: bool = False, default_name: str = "figure"):
        super().__init__(parent)
        self.batch, self.default_name = batch, default_name
        self.setWindowTitle("Export all graphs" if batch else "Export figure")
        self.setMinimumWidth(520)
        s = _settings()
        self.fmt = _combo([(f, EXPORT_LABELS[f]) for f in EXPORT_FORMATS], s.value("export/format", "svg", type=str))
        self.dpi = QSpinBox(minimum=72, maximum=1200, value=s.value("export/dpi", 600, type=int), suffix=" dpi")
        self.keep_text = QCheckBox("Keep text as editable text (SVG, PDF)")
        self.keep_text.setChecked(s.value("export/keep_text", True, type=bool))
        self.target = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.target, 1)
        row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Save to folder" if batch else "Save as", row)
        form.addRow("Format", self.fmt)
        form.addRow("Resolution (PNG, TIFF)", self.dpi)
        form.addRow("", self.keep_text)
        lay = QVBoxLayout(self)
        _page(lay)
        lay.addWidget(_card(form))
        lay.addWidget(_buttons(self))
        docs = Path.home() / "Documents"
        folder = s.value("export/folder", "", type=str) or str(docs if docs.is_dir() else Path.home())
        self.target.setText(folder if batch else str(Path(folder) / self._file_name()))
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        self._fmt_changed()

    def _file_name(self) -> str:
        return f"{Path(self.default_name).stem}.{self.fmt.currentData()}"

    def _fmt_changed(self):
        self.dpi.setEnabled(self.fmt.currentData() in RASTER)
        self.keep_text.setEnabled(self.fmt.currentData() in ("svg", "pdf", "eps"))
        t = self.target.text().strip()
        if t and not self.batch:  # keep the folder and name, follow the format
            self.target.setText(str(Path(t).with_suffix("." + self.fmt.currentData())))

    def _browse(self):
        if self.batch:
            d = QFileDialog.getExistingDirectory(self, "Export folder", self.target.text())
            if d:
                self.target.setText(d)
        else:
            f, _ = QFileDialog.getSaveFileName(self, "Export figure", self.target.text() or self._file_name(),
                                               "Figures (*.svg *.pdf *.png *.tif *.tiff *.eps)")
            if f:
                self.target.setText(f)
                ext = Path(f).suffix.lower().lstrip(".")
                if self.fmt.findData(ext) >= 0:
                    self.fmt.setCurrentIndex(self.fmt.findData(ext))

    def accept(self):
        t = self.target.text().strip()
        if not t:
            QMessageBox.warning(self, "Export", "Choose where to save.")
            return
        s = _settings()
        s.setValue("export/folder", t if self.batch else str(Path(t).parent))
        s.setValue("export/format", self.fmt.currentData())
        s.setValue("export/dpi", self.dpi.value())
        s.setValue("export/keep_text", self.keep_text.isChecked())
        super().accept()

    def values(self) -> dict:
        t = self.target.text().strip()
        if not self.batch:
            ext = Path(t).suffix.lower().lstrip(".")
            if ext not in ("svg", "pdf", "png", "tif", "tiff", "eps"):
                t += "." + self.fmt.currentData()
        return dict(target=t, fmt=self.fmt.currentData(), dpi=self.dpi.value(), text_as_paths=not self.keep_text.isChecked())
