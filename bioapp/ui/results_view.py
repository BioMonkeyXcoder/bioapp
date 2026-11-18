"""Results view: all display tables of one analysis stacked on one sheet (Prism's "Tabular results"), a status banner
and a parameters button.

Display contract (DESIGN.md) plus one optional key per table: `"sheet"`.
  - absent / None: the table is a section of the main sheet, unless it has more than LONG_TABLE rows (then its own tab);
  - a string: the table gets its own tab with that name;  True: own tab named after the table;  False: always on the sheet.
"""
from __future__ import annotations

import csv
import math
import re
from typing import NamedTuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import plot
from . import fmt, theme
from .dialogs import AnalyzeDialog

LONG_TABLE = 40       # rows; longer tables get their own tab
MAIN_TAB = "Tabular results"
WIDE_TEXT = 24        # text longer than this is left-aligned
MAX_COL_PX = 300
WRAP_COL_PX = 190     # width of a column that holds wrapped long text
TITLE_BG, HEAD_BG, STRIPE_BG = (QColor(c) for c in (theme.pale(theme.LAVENDER, 0.4), theme.pale(theme.LAVENDER, 0.7),
                                                    theme.mix(theme.BG, theme.PANEL, 0.6)))
# glass: the same lavender / white as translucent fills over the card
GLASS_ALPHA = {"title": 0.50, "head": 0.28}


def sheet_colors() -> tuple[QColor, QColor, QColor]:
    """(title, head, stripe) fills of the sheet in the current style."""
    if theme.style() != "glass":
        return TITLE_BG, HEAD_BG, STRIPE_BG
    t, h, s = QColor(theme.LAVENDER.base), QColor(theme.LAVENDER.base), QColor("#FFFFFF")
    t.setAlphaF(GLASS_ALPHA["title"])
    h.setAlphaF(GLASS_ALPHA["head"])
    s.setAlphaF(theme.STRIPE_ALPHA)
    return t, h, s


class Cell(NamedTuple):
    text: str
    raw: object = ""     # full-precision value for "copy raw" and export
    role: str = "text"   # title | head | text | num


def rawtext(c: Cell) -> str:
    """Full-precision text of a cell (what "Copy all as full precision" and the CSV export write)."""
    r = c.raw
    if isinstance(r, bool):
        return c.text
    if isinstance(r, float):
        return fmt.full(r) if math.isfinite(r) else fmt.fmt_num(r)
    return str(r)


# ---------------------------------------------------------------- building the sheet (no Qt widgets)

def _cell(v, col: str, label) -> Cell:
    if v is None:
        return Cell("")
    if isinstance(v, bool):
        return Cell("Yes" if v else "No", v)
    if isinstance(v, int):
        return Cell(str(v), v, "num")
    if isinstance(v, float):
        if fmt.is_p_label(col) or fmt.is_p_label(label):
            return Cell(fmt.fmt_p(v), fmt.full(v), "num")
        if v.is_integer() and (fmt.is_int_label(col) or fmt.is_int_label(label)):
            return Cell(str(int(v)), v, "num")
        return Cell(fmt.fmt_num(v), v, "num")
    s = str(v)
    return Cell(fmt.fmt_range(s), s, "text")


_MODES = (("best-fit", "value"), ("std. error", "se"), ("confidence interval", "ci"))


def _marked(v, e: dict, mode: str, is_param: bool) -> Cell | None:
    """Prism's inline flags for a fit table cell: "~" ambiguous, "???" undefined, from the raw entry of the data set."""
    if e.get("fixed") or e.get("shared"):
        return None
    amb = bool(e.get("ambiguous"))
    if mode == "value" and isinstance(v, float) and amb:
        return Cell("~" + fmt.fmt_num(v), "~" + fmt.full(v), "num")
    if mode == "se":
        if isinstance(v, float):
            return Cell("~" + fmt.fmt_num(v), "~" + fmt.full(v), "num") if amb else None
        return Cell("???", "", "num") if v is None and is_param else None
    if mode == "ci" and e.get("value") is not None:
        lo, hi = e.get("ci_low"), e.get("ci_high")
        if lo is None and hi is None:
            return Cell("Very wide" if amb else "???", "", "num")
        f = lambda a, inf: inf if a is None else fmt.fmt_num(a)  # noqa: E731
        r = lambda a, inf: inf if a is None else fmt.full(a)  # noqa: E731
        return Cell(("~" if amb else "") + f"{f(lo, '-infinity')} to {f(hi, '+infinity')}",
                    f"{r(lo, '-infinity')} to {r(hi, '+infinity')}", "num")
    return None


def build_sections(res: dict, title_to_id: dict | None = None) -> list[dict]:
    """One dict per display table: title, head (Cells), rows (lists of Cells), sheet (the optional key)."""
    ds, ids, out = res.get("datasets") or {}, title_to_id or {}, []
    for d in res.get("display") or []:
        title, cols, rows = str(d.get("title", "Results")), [str(c) for c in d.get("columns") or []], d.get("rows") or []
        mode = next((m for k, m in _MODES if k in title.lower()), None)
        mark = mode and cols[:1] == ["Parameter"] and ds
        body = []
        for r in rows:
            label = r[0] if r and isinstance(r[0], str) else None
            line = []
            for j, v in enumerate(r):
                c = None
                if mark and j > 0 and label is not None:
                    e = ds.get(ids.get(cols[j] if j < len(cols) else None)) or {}
                    ent = (e.get("params") or {}).get(label) or (e.get("derived") or {}).get(label)
                    if ent:
                        c = _marked(v, ent, mode, label in (e.get("params") or {}))
                line.append(c or _cell(v, cols[j] if j < len(cols) else "", label if j else None))
            body.append(line)
        # a one-column table named like its title (e.g. "R squared") needs no column header
        head = [] if len(cols) == 1 and cols[0].strip().lower() == title.strip().lower() else [Cell(c, c, "head") for c in cols]
        out.append({"title": title, "head": head, "rows": body, "sheet": d.get("sheet")})
    return out


def split_sections(sections: list[dict]) -> tuple[list[dict], list[tuple[str, dict]]]:
    """(sections on the main sheet, [(tab name, section)] for the tables with their own tab)."""
    main, tabs = [], []
    for s in sections:
        want = s["sheet"]
        if want is True or isinstance(want, str) and want or (want is None and len(s["rows"]) > LONG_TABLE):
            tabs.append((want if isinstance(want, str) and want else s["title"], s))
        else:
            main.append(s)
    return main, tabs


def grid_rows(sections: list[dict]) -> list[list[Cell]]:
    """Stack sections: title row, column-header row, data rows, one blank row between sections."""
    g: list[list[Cell]] = []
    for s in sections:
        if g:
            g.append([])
        g.append([Cell(s["title"], s["title"], "title")])
        if s["head"]:
            g.append(s["head"])
        g.extend(s["rows"])
    return g


# ---------------------------------------------------------------- the sheet widget

class _Sheet(QTableWidget):
    export_requested = Signal()

    def __init__(self, grid: list[list[Cell]]):
        super().__init__(len(grid), max([len(r) for r in grid] + [1]))
        self.grid = grid
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        bold = QFont(self.font())
        bold.setBold(True)
        title_font = QFont(bold)
        title_font.setPointSizeF(bold.pointSizeF() * 1.1)
        fm, bfm = QFontMetrics(self.font()), QFontMetrics(bold)
        widths = [60] * self.columnCount()
        stripe, wrapped, wide_cols = 0, [], set()
        self.setWordWrap(True)
        title_bg, head_bg, stripe_bg = sheet_colors()
        for i, row in enumerate(grid):
            if not row:
                self.setRowHeight(i, 10)
                continue
            head, title = row[0].role == "head", row[0].role == "title"
            self.setRowHeight(i, 26 if title else 22)
            stripe = 0 if head or title else stripe + 1
            for j, c in enumerate(row):
                it = QTableWidgetItem(c.text)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                left = j == 0 or title or (c.role == "text" and len(c.text) > WIDE_TEXT)
                it.setTextAlignment((Qt.AlignLeft if left else Qt.AlignHCenter) | Qt.AlignVCenter)
                if title:
                    it.setFont(title_font)
                    it.setBackground(title_bg)
                elif head:
                    it.setFont(bold)
                    it.setBackground(head_bg)
                elif stripe % 2 == 0:
                    it.setBackground(stripe_bg)
                if c.role == "num" and not isinstance(c.raw, str) and c.text != fmt.full(c.raw):
                    it.setToolTip(fmt.full(c.raw))
                self.setItem(i, j, it)
                if title:
                    if self.columnCount() > 1:  # a one-column sheet needs no span (Qt warns about a single-cell span)
                        self.setSpan(i, 0, 1, self.columnCount())
                    break
                if j > 0 and c.role == "text" and len(c.text) > WIDE_TEXT:  # long text: spans the free columns or wraps
                    wrapped.append(i)
                    if j == len(row) - 1 and j < self.columnCount() - 1:
                        self.setSpan(i, j, 1, self.columnCount() - j)
                    else:
                        wide_cols.add(j)
                else:
                    widths[j] = max(widths[j], (bfm if head else fm).horizontalAdvance(c.text) + 20)
        for j, w in enumerate(widths):
            self.setColumnWidth(j, max(w, WRAP_COL_PX) if j in wide_cols else min(w, MAX_COL_PX))
        for i in wrapped:
            self.resizeRowToContents(i)

    def sizeHint(self):
        return QSize(700, 400)

    # ---- copy / export
    def _text(self, raw: bool) -> str:
        """Tab-separated text of the selection (the whole sheet when nothing is selected)."""
        sel = {(i.row(), i.column()) for i in self.selectedIndexes()}
        if not sel:
            sel = {(r, c) for r, row in enumerate(self.grid) for c in range(len(row))}
        rows = range(min(r for r, _ in sel), max(r for r, _ in sel) + 1)
        cols = range(min(c for _, c in sel), max(c for _, c in sel) + 1)
        return "\n".join("\t".join((rawtext(self.grid[r][c]) if raw else self.grid[r][c].text)
                                   if (r, c) in sel and c < len(self.grid[r]) else "" for c in cols) for r in rows)

    def copy(self, raw: bool = False):
        QGuiApplication.clipboard().setText(self._text(raw))

    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):
            self.copy()
        else:
            super().keyPressEvent(e)

    def _menu(self, pos):
        m = QMenu(self)
        m.addAction("Copy", self.copy)
        m.addAction("Copy all as full precision", lambda: self.copy(True))
        m.addSeparator()
        m.addAction("Export results...", self.export_requested.emit)
        m.exec(self.viewport().mapToGlobal(pos))


# ---------------------------------------------------------------- export (CSV / XLSX)

def _xlsx_name(name: str, used: set) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "Sheet"
    n, out = 1, base
    while out.lower() in used:
        n += 1
        out = f"{base[:28]}_{n}"
    used.add(out.lower())
    return out


def _xl(c: Cell):
    """Spreadsheet value: a real number for finite numeric cells, else the text."""
    if c.role in ("title", "head") or isinstance(c.raw, bool):
        return c.text
    return c.raw if isinstance(c.raw, (int, float)) and math.isfinite(c.raw) else rawtext(c)


def export_pages(path: str, pages: list[tuple[str, list[list[Cell]]]]) -> None:
    """Write the sheet (and the extra tabs) with full-precision values: .csv = pages one below the other, .xlsx = one worksheet each."""
    if str(path).lower().endswith(".xlsx"):
        from openpyxl import Workbook
        from openpyxl.styles import Font
        wb, used = Workbook(), set()
        wb.remove(wb.active)
        for name, grid in pages:
            ws = wb.create_sheet(_xlsx_name(name, used))
            for row in grid:
                if not row:
                    ws.append([])
                    continue
                ws.append([_xl(c) for c in row])
                if row[0].role in ("title", "head"):
                    for c in ws[ws.max_row]:
                        c.font = Font(bold=True)
            ws.column_dimensions["A"].width = 34
        wb.save(path)
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for k, (_, grid) in enumerate(pages):
            if k:
                w.writerow([])
            for row in grid:
                w.writerow([c.text if c.role in ("title", "head") else rawtext(c) for c in row])


# ---------------------------------------------------------------- the view

class ResultsView(theme.GlassCard):
    graph_requested = Signal(str)  # analysis id: the main window creates the graphs and shows the first

    def __init__(self, app, aid: str, parent=None):
        super().__init__(pad=(16, 12, 16, 12), flat_pad=(16, 12, 16, 12), parent=parent)
        self.app, self.aid, self._shown, self._digits = app, aid, None, None
        self.pages: list[tuple[str, list[list[Cell]]]] = []
        self.title = QLabel()
        self.title.setStyleSheet("font-weight: 700; font-size: 15pt")
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.tabs = QTabWidget()
        self.tabs.setTabBarAutoHide(True)  # no tab bar while everything fits on the main sheet
        edit = QPushButton("Edit parameters...")
        edit.clicked.connect(self._edit)
        rerun = QPushButton("Recalculate")
        rerun.clicked.connect(lambda: app.schedule(aid))
        export = QPushButton("Export results...")
        export.clicked.connect(self.export_dialog)
        self.graph_btn = QPushButton("Graph this result")
        self.graph_btn.clicked.connect(lambda: self.graph_requested.emit(aid))
        head = QHBoxLayout()
        head.addWidget(self.title, 1)
        head.addWidget(self.graph_btn)
        head.addWidget(export)
        head.addWidget(edit)
        head.addWidget(rerun)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        self.restyle()
        lay.addLayout(head)
        lay.addWidget(self.banner)
        lay.addWidget(self.tabs, 1)
        app.analysis_changed.connect(self._changed)
        theme.notifier.changed.connect(self._restyled)
        self.refresh()

    def _restyled(self, *_):
        self.reformat()

    def _changed(self, aid):
        if aid == self.aid and aid in self.app.project.analyses:
            self.refresh()

    def _edit(self):
        AnalyzeDialog(self.app, analysis_obj=self.app.project.analyses[self.aid], parent=self).exec()

    def _set_banner(self, kind, lines):
        self.banner.setStyleSheet(theme.banner_css(kind))
        self.banner.setText("<br>".join(lines))
        self.banner.setVisible(bool(lines))

    def refresh(self):
        a = self.app.project.analyses[self.aid]
        self.title.setText(a.title)
        res = a.result or {}
        stale = self.app.is_stale(self.aid) and a.status != "failed"
        msgs = list(res.get("messages") or [])
        if stale:
            self._set_banner("stale", ["Out of date - recalculating..." if a.result else "Calculating..."] + msgs)
        elif res.get("status") == "failed":
            self._set_banner("failed", ["Analysis failed:"] + msgs)
        elif res.get("status") == "warning":
            self._set_banner("warning", msgs)
        else:
            self._set_banner("info", msgs)
        if res is not self._shown:  # rebuild the tables only when a new result arrives
            self._shown = res
            t = self.app.project.tables.get(a.table_id)
            self.graph_btn.setEnabled(bool(t is not None and plot.specs_for_result(t, a.id, a.kind, res, str)))
            self._build(res, {d.title: d.id for d in t.datasets} if t is not None else {})

    def _build(self, res, title_to_id):
        self._digits = fmt.digits()
        main, extra = split_sections(build_sections(res, title_to_id))
        self.pages = ([(MAIN_TAB, grid_rows(main))] if main else []) + [(n, grid_rows([s])) for n, s in extra]
        self.tabs.clear()
        for name, grid in self.pages:
            sheet = _Sheet(grid)
            sheet.export_requested.connect(self.export_dialog)
            self.tabs.addTab(sheet, name)

    def reformat(self):
        """Re-render the numbers after `fmt.set_digits` (the controller calls this on the open views)."""
        self._shown = None
        self.refresh()

    def showEvent(self, e):
        if self._digits is not None and self._digits != fmt.digits():
            self.reformat()
        super().showEvent(e)

    # ---- copy / export
    def export_to(self, path: str) -> None:
        export_pages(path, self.pages)

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export results", f"{self.title.text()}.xlsx",
                                              "Excel workbook (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            self.export_to(path)
        except OSError as e:
            QMessageBox.warning(self, "Export results", f"Could not write the file:\n{e}")
