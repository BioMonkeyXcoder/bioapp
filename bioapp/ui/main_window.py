"""Main window: navigator tree, stacked views, menus, toolbar, status bar, file handling."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog, QInputDialog, QLabel, QMainWindow, QMenu,
                               QMessageBox, QPushButton, QSplitter, QStackedWidget, QStyle, QStyledItemDelegate,
                               QStyleOptionViewItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from .. import analysis, io_prism, io_tabular
from ..app import App
from . import theme
from .dialogs import AnalyzeDialog, ImportDialog, NewTableDialog
from .graph_view import GraphView, LayoutView, batch_export, export_item
from .results_view import ResultsView
from .table_view import TableEditor

SECTIONS = (("tables", "Data Tables"), ("analyses", "Results"), ("graphs", "Graphs"), ("layouts", "Layouts"))
SOURCE_ROLE = Qt.UserRole + 1   # text shown in grey at the right of a navigator row (the source table)
FAMILY_BG = {s: QBrush(QColor(theme.pale(t))) for s, t in theme.SECTION.items()}  # rows of the current table's family
GLASS_FAMILY_ALPHA, GLASS_SELECTED_ALPHA = 0.42, 0.9  # navigator capsules: section tone (base / deep) at this opacity
GLASS_FAMILY_BG = {s: QBrush(QColor(t.base)) for s, t in theme.SECTION.items()}
for _b in GLASS_FAMILY_BG.values():
    _c = _b.color()
    _c.setAlphaF(GLASS_FAMILY_ALPHA)
    _b.setColor(_c)
MARKS = {"failed": (theme.ROSE.ink, "Failed"), "stale": (theme.SKY.ink, "Out of date - recalculating"),
         "warning": (theme.BUTTER.ink, "Completed with warnings")}
PROJECT_FILTER = "bioapp projects (*.biofig)"
IMPORT_FILTER = "Data files (*.csv *.tsv *.txt *.xlsx *.pzfx *.xml);;All files (*)"


_ICONS: dict[tuple, QIcon] = {}


def nav_icon(section: str, kind: str = "", mark: str | None = None, derived: bool = False) -> QIcon:
    """Small painted icon of a navigator row: section / kind glyph plus a status dot (``mark`` in MARKS)."""
    key = (section, kind, mark, derived)
    if key not in _ICONS:
        pm = QPixmap(16, 16)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        tone = QColor(theme.SECTION[section].ink)
        deep = QColor(theme.SECTION[section].deep)
        if section == "tables":
            p.fillRect(QRectF(2, 2, 12, 3.5), QColor(theme.LAVENDER.deep) if derived else deep)
            p.setPen(QPen(tone, 1))
            p.drawRect(QRectF(2.5, 2.5, 11, 11))
            for v in (6, 10):
                p.drawLine(QPointF(2.5, v + 1), QPointF(13.5, v + 1))
                p.drawLine(QPointF(v + 1, 2.5), QPointF(v + 1, 13.5))
        elif section == "analyses":
            p.setPen(QPen(tone, 1.6))
            if kind in ("nonlin_fit", "linear_regression", "deming_regression"):  # sigmoid
                p.drawPolyline(QPolygonF([QPointF(x, 13 - 10 / (1 + 2.718 ** (-(x - 8) * 0.7))) for x in range(2, 15)]))
            else:  # sigma
                p.drawPolyline(QPolygonF([QPointF(12.5, 3), QPointF(4, 3), QPointF(8.5, 8), QPointF(4, 13), QPointF(12.5, 13)]))
        elif section == "graphs":
            p.setPen(QPen(deep, 1.2))
            p.drawPolyline(QPolygonF([QPointF(2, 2), QPointF(2, 14), QPointF(14, 14)]))
            p.setPen(QPen(tone, 1.4))
            if kind in ("grouped", "contingency", "histogram"):
                for x, h in ((4, 6), (7, 9), (10.5, 5)):
                    p.fillRect(QRectF(x, 13 - h, 2.5, h), tone)
            elif kind in ("column", "nested"):
                p.setBrush(tone)
                for x, y in ((5, 9), (5, 6), (10, 5), (10, 8), (10, 4)):
                    p.drawEllipse(QPointF(x, y), 1.3, 1.3)
            else:
                pts = [QPointF(4, 11), QPointF(7, 9), QPointF(10, 5), QPointF(13, 4)]
                p.drawPolyline(QPolygonF(pts))
                p.setBrush(tone)
                for q in pts:
                    p.drawEllipse(q, 1.3, 1.3)
        else:  # layouts
            p.setBrush(deep)
            p.setPen(Qt.NoPen)
            for x, y in ((2, 2), (9, 2), (2, 9), (9, 9)):
                p.drawRoundedRect(QRectF(x, y, 5, 5), 1, 1)
        if mark:
            p.setBrush(QColor(MARKS[mark][0]))
            p.setPen(QPen(QColor(theme.PANEL), 1.2))
            p.drawEllipse(QPointF(12, 12), 3.4, 3.4)
        p.end()
        _ICONS[key] = QIcon(pm)
    return _ICONS[key]


SOURCE_GAP = 14  # px between a navigator name and its grey source text
TOOL_ICON = 22   # toolbar icon size (px), in step with the 12 pt interface font


class _NavDelegate(QStyledItemDelegate):
    """Default row plus the source table in small grey text at the right edge when the name leaves room."""

    def initStyleOption(self, opt, idx):
        """Row backgrounds (selected, current table's family) are painted in ``paint``: rounded, in the section colour."""
        super().initStyleOption(opt, idx)
        opt.state &= ~QStyle.State_Selected
        opt.backgroundBrush = QBrush()

    def sizeHint(self, opt, idx):
        s = super().sizeHint(opt, idx)
        return QSize(s.width(), s.height() + 6)

    def paint(self, p, opt, idx):
        data = idx.data(Qt.UserRole)
        fam = idx.data(Qt.BackgroundRole)
        fam = fam if isinstance(fam, QBrush) and fam.style() != Qt.NoBrush else None
        if data and (opt.state & QStyle.State_Selected or fam):
            p.save()
            p.setRenderHint(QPainter.Antialiasing)
            glass = theme.style() == "glass"
            tone = theme.SECTION[data[0]]
            sel = QColor(tone.deep)
            if glass:
                sel.setAlphaF(GLASS_SELECTED_ALPHA)
            on = bool(opt.state & QStyle.State_Selected)
            p.setPen(QPen(QColor(tone.ink), theme.SEL_EDGE) if on else Qt.NoPen)  # selected: an ink outline
            p.setBrush(sel if on else fam)
            r = QRectF(opt.rect).adjusted(2.5, 1.5, -2.5, -1.5)
            rad = r.height() / 2 if glass else 5  # glass: a capsule
            p.drawRoundedRect(r, rad, rad)
            p.restore()
        super().paint(p, opt, idx)
        src, name = idx.data(SOURCE_ROLE), idx.data(Qt.DisplayRole) or ""
        if not src or src.lower() in name.lower():  # "Graph of <table>" already says it
            return
        o = QStyleOptionViewItem(opt)
        self.initStyleOption(o, idx)
        st = opt.widget.style() if opt.widget else QApplication.style()
        text = st.subElementRect(QStyle.SE_ItemViewItemText, o, opt.widget)  # where the name really starts
        used = text.left() + o.fontMetrics.horizontalAdvance(name) + SOURCE_GAP
        room = opt.rect.right() - 8 - used
        if room < 48:  # too little left: the source is left out, never squeezed against the name
            return
        f = QFont(opt.font)
        f.setPointSizeF(max(6.5, f.pointSizeF() * 0.85))
        p.save()
        p.setFont(f)
        p.setPen(QColor(theme.TEXT if opt.state & QStyle.State_Selected else theme.TEXT2))
        p.drawText(QRect(used, opt.rect.top(), room, opt.rect.height()), Qt.AlignRight | Qt.AlignVCenter,
                   QFontMetrics(f).elidedText(src, Qt.ElideRight, room))
        p.restore()


class MainWindow(QMainWindow):
    def __init__(self, app: App | None = None):
        super().__init__()
        self.app = app or App()
        self.views: dict[tuple[str, str], object] = {}
        self.current: tuple[str, str] | None = None
        self._project_ref = self.app.project
        self.resize(1280, 800)

        self.nav = QTreeWidget()
        self.nav.setObjectName("nav")
        self.nav.setHeaderHidden(True)
        self.nav.setMinimumWidth(220)
        self.nav.setIconSize(QSize(16, 16))
        self.nav.setIndentation(14)
        self.nav.setItemDelegate(_NavDelegate(self.nav))
        self.nav.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)  # double-click renames
        self.nav.setContextMenuPolicy(Qt.CustomContextMenu)
        self.nav.customContextMenuRequested.connect(self._nav_menu)
        self.nav.currentItemChanged.connect(self._nav_selected)
        self.nav.itemChanged.connect(self._nav_edited)
        self._items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self.welcome_label = QLabel("Create a data table (File > New data table) or import data to begin.")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setStyleSheet(f"color: {theme.TEXT2}; font-size: 13pt")
        self.welcome = theme.GlassCard(pad=20)
        lay = QVBoxLayout()
        lay.addWidget(self.welcome_label)
        self.welcome.setLayout(lay)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.welcome)
        nav_card = theme.GlassCard(pad=8)  # the navigator floats as a glass card; views are cards or hold cards
        lay = QVBoxLayout()
        lay.addWidget(self.nav)
        nav_card.setLayout(lay)
        self.split = split = QSplitter()
        split.addWidget(nav_card)
        split.addWidget(self.stack)
        split.setStretchFactor(1, 1)
        split.setSizes([250, 1030])
        self.setCentralWidget(split)
        theme.notifier.changed.connect(self._restyle)
        self._restyle()

        self.job_label = QLabel("Idle")
        self.cancel_btn = QPushButton("Cancel jobs")
        self.cancel_btn.setFlat(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.app.cancel_jobs)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.job_label, 1)
        self.statusBar().addPermanentWidget(self.cancel_btn)

        self._build_actions()
        self.app.project_changed.connect(self.rebuild)
        self.app.dirty_changed.connect(lambda _: self._update_title())
        self.app.status.connect(self._status)
        self.app.analysis_changed.connect(self._refresh_result_row)
        self.app.table_changed.connect(self._refresh_table_row)
        self.app.file_changed_on_disk.connect(self._on_file_changed_on_disk)
        self.rebuild()
        self._update_title()

    def _on_file_changed_on_disk(self, path: str):
        """Another program (e.g. an agent through the MCP server) rewrote the open project: reload it, keeping the
        current view when the item still exists; ask first when there are unsaved changes here."""
        if self.app.dirty and QMessageBox.question(
                self, "Project changed on disk",
                "The project file was changed by another program.\n\nReload it and discard your unsaved changes?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            self._status("Kept your version; saving will overwrite the other program's changes")
            return
        keep = self.current
        self._guard(lambda: self.app.open_project(path))
        if keep and keep[1] in getattr(self.app.project, keep[0], {}):
            self.select_item(*keep)
        self._status("Reloaded: the project file was changed by another program")

    # ------------------------------------------------------------ actions / menus
    def _act(self, text, slot, shortcut=None, menu=None):
        a = QAction(text, self)
        a.triggered.connect(lambda *_: self._guard(slot))
        if shortcut:
            a.setShortcut(shortcut)
        if menu is not None:
            menu.addAction(a)
        return a

    def _guard(self, fn):
        try:
            fn()
        except Exception as exc:
            QMessageBox.warning(self, "bioapp", str(exc))

    def _build_actions(self):
        mb = self.menuBar()
        tb = theme.PillToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(TOOL_ICON, TOOL_ICON))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        f = mb.addMenu("&File")
        self._act("&New project", self.new_project, "Ctrl+N", f)
        new_table = self._act("New data &table...", self.new_table, "Ctrl+T", f)
        self._act("&Open...", self.open_dialog, "Ctrl+O", f)
        self._act("&Save", self.save, "Ctrl+S", f)
        self._act("Save &as...", self.save_as, "Ctrl+Shift+S", f)
        f.addSeparator()
        import_data = self._act("&Import data...", self.import_dialog, "Ctrl+I", f)
        import_traces = self._act("Import &traces (UV-volume, chromatograms)...", self.import_traces, "Ctrl+Shift+I", f)
        self._act("Export table as CSV...", self.export_csv, None, f)
        self._act("Export table as Prism file (.pzfx)...", self.export_pzfx, None, f)
        export = self._act("Export current graph or layout...", self.export_current, None, f)
        self._act("Export all graphs...", lambda: batch_export(self, self.app), None, f)
        f.addSeparator()
        self._act("E&xit", self.close, "Ctrl+Q", f)

        e = mb.addMenu("&Edit")
        self.undo_action = self.app.undo.createUndoAction(self, "&Undo")
        self.redo_action = self.app.undo.createRedoAction(self, "&Redo")
        self.undo_action.setShortcut("Ctrl+Z")
        self.redo_action.setShortcut("Ctrl+Y")
        for a in (self.undo_action, self.redo_action):
            e.addAction(a)
        e.addSeparator()
        for text, name, key in (("Copy", "copy", None), ("Paste", "paste", None), ("Clear cells", "clear_cells", None),
                                ("Exclude / include selected values", "toggle_exclusion", "Ctrl+E"),
                                ("Add row(s)", "add_rows", None), ("Remove selected rows", "remove_rows", None),
                                ("Add data set", "add_dataset", None), ("Remove selected data sets", "remove_datasets", None),
                                ("Switch variable numeric / categorical", "toggle_variable_type", None)):
            self._act(text, lambda n=name: self._editor_call(n), key, e)
        e.addSeparator()
        self._act("Rename", self.rename_current, "F2", e)
        self._act("Delete current item", self.delete_current, None, e)

        self.analyze_menu = mb.addMenu("&Analyze")
        self.analyze_menu.aboutToShow.connect(self._fill_analyze_menu)
        analyze = self._act("Analyze...", lambda: self.analyze(None), "Ctrl+Shift+A", None)

        g = mb.addMenu("&Graph")
        new_graph = self._act("New graph from current table...", self.new_graph, None, g)
        g.addAction(import_traces)
        self._act("Graph this result", self.graph_current_result, None, g)
        self._act("Export graph...", self.export_current, None, g)

        v = mb.addMenu("&View")
        self.digits_action = self._act("Result digits...", self.choose_digits, None, v)
        self.digits_action.setEnabled(importlib.util.find_spec("bioapp.ui.fmt") is not None)
        self.style_menu = v.addMenu("Interface style")
        self.style_group = QActionGroup(self)
        for key, label in (("glass", "Liquid glass"), ("flat", "Flat")):
            a = self.style_menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(theme.style() == key)
            a.triggered.connect(lambda _=False, k=key: self.set_style(k))
            self.style_group.addAction(a)
            a.setData(key)

        lm = mb.addMenu("&Layout")
        self._act("New layout", self.new_layout, None, lm)
        self._act("Export layout...", self.export_current, None, lm)

        # the few frequent actions as a toolbar: (action, icon, short label); None = separator
        for item in ((new_table, "table", "New table"), (import_data, "import", "Import"), (import_traces, "graph", "Traces"), None,
                     (analyze, "analyze", "Analyze"), (new_graph, "graph", "New graph"), (export, "export", "Export"), None,
                     (self.undo_action, "undo", "Undo"), (self.redo_action, "redo", "Redo")):
            if item is None:
                tb.addSeparator()
                continue
            a, name, short = item
            a.setIcon(theme.icon(name, TOOL_ICON))
            a.setIconText(short)
            tb.addAction(a)

    def _fill_analyze_menu(self):
        self.analyze_menu.clear()
        t = self.app.project.tables.get(self.current_table_id() or "")
        a = self.analyze_menu.addAction("Analyze...")
        a.triggered.connect(lambda: self._guard(lambda: self.analyze(None)))
        self.analyze_menu.addSeparator()
        for k in analysis.all_kinds().values():
            act = self.analyze_menu.addAction(k.label)
            act.setEnabled(t is not None and t.kind in k.table_kinds)
            act.triggered.connect(lambda _=False, kid=k.id: self._guard(lambda: self.analyze(kid)))

    # ------------------------------------------------------------ look
    def paintEvent(self, e):
        """Glass: the cached paper backdrop under everything (menu, toolbar and status bar are transparent)."""
        if theme.style() != "glass":
            return super().paintEvent(e)
        p = QPainter(self)
        p.drawPixmap(0, 0, theme.backdrop(self.width(), self.height(), self.devicePixelRatioF()))
        p.end()

    def set_style(self, key: str) -> None:
        """Switch the interface style for the whole application and remember it."""
        theme.apply(QApplication.instance(), key)
        theme.save_style(key)

    def _restyle(self, *_):
        glass = theme.style() == "glass"
        m = 12 - theme.SHADOW if glass else 0  # cards are 12 px from the window edge, 12 px apart (shadow bands included)
        self.split.setContentsMargins(m, 0, m, m)
        self.split.setHandleWidth(2 if glass else 7)
        if hasattr(self, "style_group"):
            for a in self.style_group.actions():
                a.setChecked(a.data() == theme.style())
        for w in self.views.values():
            self._style_view(w)
        if self._items:
            self._highlight_family()
        self.update()

    @staticmethod
    def _style_view(w) -> None:
        if isinstance(w, (GraphView, LayoutView)):  # graph / layout view: the surround of the opaque white figure
            w.scroll.setStyleSheet(theme.surround_css())

    @staticmethod
    def _cardify(w) -> None:
        """A graph or layout view holds two panes (figure, properties): each becomes its own glass card (flat style: no
        margins, nothing painted)."""
        sp = w.findChild(QSplitter, options=Qt.FindDirectChildrenOnly)
        if sp is None:
            return
        keep = [(sp.widget(i), sp.isCollapsible(i)) for i in range(sp.count())]
        for i, (child, coll) in enumerate(keep):
            card = theme.GlassCard(pad=12)
            card.setSizePolicy(child.sizePolicy())  # the splitter stretch factors live in the size policy
            lay = QVBoxLayout()
            lay.addWidget(child)
            card.setLayout(lay)
            sp.insertWidget(i, card)
            sp.setCollapsible(i, coll)
        sp.setHandleWidth(2 if theme.style() == "glass" else 7)

    # ------------------------------------------------------------ navigator and views
    def rebuild(self):
        if self._project_ref is not self.app.project:  # project replaced: drop every view
            self._project_ref = self.app.project
            self._drop_views(list(self.views))
        p = self.app.project
        live = {"tables": p.tables, "analyses": p.analyses, "graphs": p.graphs, "layouts": p.layouts}
        self._drop_views([k for k in self.views if k[1] not in live[k[0]]])
        self.nav.blockSignals(True)
        self.nav.clear()
        self._items = {}
        want = None
        for key, label in SECTIONS:
            top = QTreeWidgetItem([label])
            top.setFlags(Qt.ItemIsEnabled)
            top.setIcon(0, nav_icon(key))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self.nav.addTopLevelItem(top)
            for iid in live[key]:
                it = QTreeWidgetItem([self._label(key, iid)])
                it.setData(0, Qt.UserRole, (key, iid))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                top.addChild(it)
                self._items[(key, iid)] = it
                self._decorate((key, iid))
                if (key, iid) == self.current:
                    want = it
            top.setExpanded(True)
        self.nav.blockSignals(False)
        if want is not None:
            self.nav.setCurrentItem(want)
        elif self.current not in self.views or self.current is None:
            self.current = None
            self.stack.setCurrentWidget(self.welcome)
        self._highlight_family()
        self._update_title()

    def _label(self, section, iid) -> str:
        if section == "graphs":
            return self.app.graph_title(iid)
        return getattr(self.app.project, section)[iid].title

    def result_mark(self, aid: str) -> str | None:
        """Status marker of a result row: "failed", "stale" (out of date or still computing), "warning" or None."""
        a = self.app.project.analyses[aid]
        if a.status == "failed" or (a.result or {}).get("status") == "failed":
            return "failed"
        if self.app.is_stale(aid) or a.status in ("running", "not_run"):
            return "stale"
        return "warning" if a.status == "warning" else None

    def _table_of(self, section: str, iid: str) -> str | None:
        p = self.app.project
        if section == "tables":
            return iid
        return getattr(p, section)[iid].table_id if section in ("analyses", "graphs") and iid in getattr(p, section) else None

    def _decorate(self, key) -> None:
        """Icon, tooltip and source text of one navigator row (no signals: setting item data would count as a rename)."""
        it = self._items.get(key)
        if it is None:
            return
        sec, iid = key
        p = self.app.project
        self.nav.blockSignals(True)
        try:
            src = tip = mark = None
            if sec == "tables":
                t = p.tables[iid]
                src = None
                tip = f"{t.kind} table: {len(t.datasets)} data sets, {t.n_rows} rows"
                if t.source in p.analyses:
                    tip += f"\nOutput of {p.analyses[t.source].title}"
                it.setIcon(0, nav_icon(sec, derived=bool(t.source)))
            elif sec == "analyses":
                a = p.analyses[iid]
                mark = self.result_mark(iid)
                tip = f"{a.title}\nTable: {p.tables[a.table_id].title}" if a.table_id in p.tables else a.title
                if mark:
                    tip += f"\n{MARKS[mark][1]}"
                tip += "".join(f"\n{m}" for m in ((a.result or {}).get("messages") or [])[:3])
                it.setIcon(0, nav_icon(sec, a.kind, mark))
                src = p.tables[a.table_id].title if a.table_id in p.tables else None
            elif sec == "graphs":
                s = p.graphs[iid]
                src = p.tables[s.table_id].title if s.table_id in p.tables else None
                tip = f"{s.kind} graph of table: {src}"
                it.setIcon(0, nav_icon(sec, s.kind))
            else:
                tip = f"{len(p.layouts[iid].panels)} panels"
                it.setIcon(0, nav_icon(sec))
            it.setToolTip(0, tip)
            it.setData(0, SOURCE_ROLE, src)
        finally:
            self.nav.blockSignals(False)

    def _refresh_result_row(self, aid: str) -> None:
        self._decorate(("analyses", aid))

    def _refresh_table_row(self, tid: str) -> None:
        self._decorate(("tables", tid))

    def _highlight_family(self) -> None:
        """Subtle background on the rows that belong to the current table: it, its results and graphs."""
        tid = self._table_of(*self.current) if self.current else None
        p = self.app.project
        fam = {("tables", tid)} if tid else set()
        for sec in ("analyses", "graphs"):
            fam |= {(sec, i) for i in getattr(p, sec) if tid and self._table_of(sec, i) == tid}
        fam |= {("tables", a.output_table_id) for a in p.analyses.values() if a.table_id == tid and a.output_table_id}
        self.nav.blockSignals(True)
        for k, it in self._items.items():
            it.setBackground(0, (GLASS_FAMILY_BG if theme.style() == "glass" else FAMILY_BG)[k[0]] if k in fam else QBrush())
        self.nav.blockSignals(False)

    def _nav_edited(self, item, _col=0) -> None:
        """Inline rename (double-click / F2) ends here; applied after the editor has closed."""
        data = item.data(0, Qt.UserRole)
        if data and item.text(0).strip():
            text = item.text(0).strip()
            QTimer.singleShot(0, lambda: self.apply_rename(data[0], data[1], text))
        elif data:
            QTimer.singleShot(0, self.rebuild)  # blank name: put the old one back

    def apply_rename(self, section: str, iid: str, text: str) -> None:
        if iid in getattr(self.app.project, section):
            self.app.rename(section, iid, text)
            self.rebuild()  # also restores the label when nothing changed

    def choose_digits(self) -> None:
        from . import fmt
        n, ok = QInputDialog.getInt(self, "Result digits", "Significant digits shown in results:", fmt.digits(),
                                    fmt.MIN_DIGITS, fmt.MAX_DIGITS)
        if ok:
            self.set_result_digits(n)

    def set_result_digits(self, n: int) -> None:
        """Apply the number of significant digits and refresh the open result views."""
        from . import fmt
        fmt.set_digits(n)
        for (sec, _), w in self.views.items():
            if sec == "analyses":
                if hasattr(w, "reformat"):
                    w.reformat()
                else:  # a view without reformat(): rebuild its tables
                    w._shown = None
                    w.refresh()

    def _drop_views(self, keys):
        for k in keys:
            w = self.views.pop(k, None)
            if w is not None:
                self.stack.removeWidget(w)
                w.deleteLater()

    def _nav_selected(self, item, _prev=None):
        data = item.data(0, Qt.UserRole) if item else None
        if data:
            self.show_item(*data)
            self._highlight_family()

    def show_item(self, section: str, iid: str):
        key = (section, iid)
        if key not in self.views:
            make = {"tables": TableEditor, "analyses": ResultsView, "graphs": GraphView, "layouts": LayoutView}[section]
            w = make(self.app, iid)
            if section == "analyses":
                w.graph_requested.connect(lambda aid: self._guard(lambda: self.graph_result(aid)))
            if section in ("graphs", "layouts"):
                self._cardify(w)
            self._style_view(w)
            self.views[key] = w
            self.stack.addWidget(w)
        self.current = key
        self.stack.setCurrentWidget(self.views[key])

    def select_item(self, section: str, iid: str):
        """Select an item in the navigator (and show it)."""
        for i in range(self.nav.topLevelItemCount()):
            top = self.nav.topLevelItem(i)
            for j in range(top.childCount()):
                if top.child(j).data(0, Qt.UserRole) == (section, iid):
                    self.nav.setCurrentItem(top.child(j))
                    return
        raise KeyError(iid)

    def current_table_id(self) -> str | None:
        p = self.app.project
        if self.current:
            sec, iid = self.current
            if sec == "tables":
                return iid
            if sec == "analyses" and iid in p.analyses:
                return p.analyses[iid].table_id
            if sec == "graphs" and iid in p.graphs:
                return p.graphs[iid].table_id
        return next(iter(p.tables), None)

    def _editor_call(self, name):
        w = self.views.get(self.current)
        if isinstance(w, TableEditor):
            getattr(w, name)()

    def _nav_menu(self, pos):
        item = self.nav.itemAt(pos)
        data = item.data(0, Qt.UserRole) if item else None
        if not data:
            return
        m = QMenu(self)
        m.addAction("Rename", self.rename_current)
        if data[0] == "tables":
            m.addAction("Analyze...", lambda: self._guard(lambda: self.analyze(None)))
            m.addAction("New graph...", lambda: self._guard(self.new_graph))
        if data[0] == "analyses":
            m.addAction("Graph this result", lambda: self._guard(lambda: self.graph_result(data[1])))
        m.addAction("Delete", self.delete_current)
        m.exec(self.nav.viewport().mapToGlobal(pos))

    def rename_current(self):
        it = self._items.get(self.current)
        if it is not None:
            self.nav.editItem(it, 0)

    def delete_current(self):
        if not self.current:
            return
        sec, iid = self.current
        if QMessageBox.question(self, "Delete", f"Delete '{self._label(sec, iid)}'? You can undo this.") == QMessageBox.Yes:
            self.app.delete_item(sec, iid)

    # ------------------------------------------------------------ creating things
    def new_table(self):
        dlg = NewTableDialog(self)
        if dlg.exec():
            sample = dlg.sample_table(f"Data {len(self.app.project.tables) + 1}")
            if sample is not None:
                self.app.add_table(sample, "New table from sample data")
                tid = sample.id
            else:
                tid = self.app.new_table(**dlg.values())
            self.select_item("tables", tid)

    def analyze(self, kind: str | None):
        tid = self.current_table_id()
        if tid is None:
            raise ValueError("Create or import a data table first.")
        AnalyzeDialog(self.app, tid, kind, parent=self).exec()

    def new_graph(self):
        tid = self.current_table_id()
        if tid is None:
            raise ValueError("Create or import a data table first.")
        choices = self.app.graph_choices(tid)
        if not choices:
            raise ValueError("This table has no graph yet: run an analysis on it first (e.g. Kaplan-Meier for a "
                             "survival table), then graph its result.")
        pick = 0
        if len(choices) > 1:
            label, ok = QInputDialog.getItem(self, "New graph", "Graph:", [c[0] for c in choices], 0, False)
            if not ok:
                return
            pick = [c[0] for c in choices].index(label)
        _, kind, aid = choices[pick]
        self.select_item("graphs", self.app.new_graph(tid, kind, aid))

    def graph_result(self, aid: str):
        ids = self.app.graph_result(aid)
        if not ids:
            raise ValueError("This result has no graph.")
        self.select_item("graphs", ids[0])

    def graph_current_result(self):
        if not self.current or self.current[0] != "analyses":
            raise ValueError("Open a result first.")
        self.graph_result(self.current[1])

    def new_layout(self):
        self.select_item("layouts", self.app.new_layout())

    # ------------------------------------------------------------ files
    def _update_title(self):
        name = Path(self.app.path).name if self.app.path else "Untitled"
        self.setWindowTitle(f"{'*' if self.app.dirty else ''}{name} - bioapp")

    def _status(self, text):
        self.job_label.setText(text)
        self.cancel_btn.setEnabled(text.startswith("Computing"))

    def maybe_save(self) -> bool:
        """True if it is fine to discard the current project (saved, or the user chose to discard)."""
        if not self.app.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to the current project?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        return r == QMessageBox.Discard or self.save()

    def new_project(self):
        if self.maybe_save():
            self.app.new_project()
            self.app.discard_recovery()

    def open_dialog(self):
        if not self.maybe_save():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        self.app.open_project(path)

    def save(self) -> bool:
        if not self.app.path:
            return self.save_as()
        self.app.save()
        return True

    def save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.app.path or "", PROJECT_FILTER)
        if not path:
            return False
        self.app.save(path if path.lower().endswith(".biofig") else path + ".biofig")
        return True

    def import_traces(self):
        """Quick trace mode: CSV/TXT instrument exports -> one table per trace + a trace graph (selected)."""
        from .trace_dialog import open_trace_import
        self._guard(lambda: open_trace_import(self))

    def import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import data", "", IMPORT_FILTER)
        if path:
            self.import_path(path)

    def import_path(self, path: str, options: dict | None = None, quiet: bool = False) -> list[str]:
        """Import a data file. ``options`` (grid_to_table keywords) and ``quiet`` skip the dialogs (tests, scripts)."""
        p = Path(path)
        if p.suffix.lower() in (".pzfx", ".xml"):
            tables, report = io_prism.read_pzfx(p)
            if tables:
                self.app.add_tables(tables, "Import Prism file")
            if not (options or quiet):
                QMessageBox.information(self, "Import report", "\n".join(report) or "Nothing was imported.")
            return report
        grid = io_tabular.read_grid(p)
        if options:
            table, report = io_tabular.grid_to_table(grid, title=p.stem, **options)
        else:
            dlg = ImportDialog(grid, p.stem, self)
            if not dlg.exec():
                return []
            table, report = dlg.table, dlg.report
        self.app.add_tables([table])
        return report

    def _current_table(self):
        tid = self.current[1] if self.current and self.current[0] == "tables" else None
        if tid is None:
            raise ValueError("Select a data table first.")
        return self.app.project.tables[tid]

    def export_csv(self):
        t = self._current_table()
        path, _ = QFileDialog.getSaveFileName(self, "Export table", t.title + ".csv", "CSV (*.csv)")
        if path:
            io_tabular.write_csv(t, path)

    def export_pzfx(self):
        t = self._current_table()
        path, _ = QFileDialog.getSaveFileName(self, "Export table", t.title + ".pzfx", "Prism files (*.pzfx)")
        if path:
            io_prism.write_pzfx([t], path)

    def export_current(self):
        if not self.current or self.current[0] not in ("graphs", "layouts"):
            raise ValueError("Open a graph or a layout first.")
        export_item(self, self.app, self.current[0][:-1], self.current[1])

    def closeEvent(self, e):
        if not self.maybe_save():
            e.ignore()
            return
        self.app.discard_recovery()
        self.app.shutdown()
        e.accept()

    def offer_recovery(self):
        if self.app.has_recovery() and QMessageBox.question(
                self, "Recover work", "bioapp did not close normally. Recover the autosaved project?") == QMessageBox.Yes:
            self.app.recover()
        elif self.app.has_recovery():
            self.app.discard_recovery()

