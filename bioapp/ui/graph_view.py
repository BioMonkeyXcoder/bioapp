"""Graph view (canvas + property panel), layout view and figure export helper."""
from __future__ import annotations

import re
import string

import numpy as np
from matplotlib.axis import Axis
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform
from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

from .. import plot
from .dialogs import ExportDialog

BASE_DPI = 100.0  # logical pixels per inch at 100 % zoom
ZOOMS = ("fit", 75, 100, 150, 200, 300)  # "fit": scale the figure to the preview area
DEFAULT_ZOOM = "fit"  # tests pin 200 so pixel-exact canvas interactions stay reproducible
FONTS = ("Arial", "Helvetica", "Times New Roman", "DejaVu Sans", "Microsoft YaHei")
PRIORITY = ("bracket", "text", "annotation", "legend_entry", "legend", "fit", "series", "band", "title", "xtitle", "ytitle",
            "xaxis", "yaxis")  # what a click selects when several objects lie under it
COLORMAPS = ("viridis", "RdBu_r", "coolwarm", "magma", "plasma", "cividis", "Blues", "Greys")
CATEGORY_X = ("column", "grouped", "contingency", "nested", "heatmap", "correlation_matrix")  # no numeric X axis
# what the property panel shows for the internal ids of PlotSpec choices
LABELS = {
    "mean_error": "Mean with error bars", "replicates": "Individual replicates", "scatter": "Scatter dot plot",
    "bar": "Bars with scatter dots", "box": "Box and whiskers", "violin": "Violin plot", "bars": "Bars",
    "grouped": "Side by side", "stacked": "Stacked", "overlay": "Overlaid bars", "points": "Points and lines",
    "subject_markers": "Subjects as markers", "subject_colors": "Subjects as colours",
    "sd": "Standard deviation (SD)", "sem": "Standard error (SEM)", "ci95": "95% confidence interval", "none": "None",
    "ci": "95% confidence band", "pi": "95% prediction band", "both": "Confidence and prediction bands",
    "fill": "Shaded", "dotted": "Dotted lines",
    "counts": "Counts", "relative": "Relative frequency", "cumulative": "Cumulative", "percent": "Percent of row total",
    "raw": "Raw values", "zscore": "Z-scores", "eigenvalues": "Eigenvalues", "explained": "Percent variance explained",
    "off": "No", "values": "Yes, plain numbers", "power": "Yes, as powers of 10",
    "outside": "Outside, right of the graph", "best": "Automatic (inside, away from data)", "manual": "Where I dragged it",
    "upper left": "Inside, upper left", "upper right": "Inside, upper right",
    "lower left": "Inside, lower left", "lower right": "Inside, lower right",
    "o": "Circle", "s": "Square", "^": "Triangle up", "v": "Triangle down", "D": "Diamond", "<": "Triangle left",
    ">": "Triangle right", "p": "Pentagon",
    "filled": "Filled", "hollow": "Hollow", "custom_fill": "Custom colour...",
    "solid": "Solid", "dashed": "Dashed", "dotted": "Dotted", "dashdot": "Dash-dot", "custom": "Custom pattern",
    "both": "Both sides", "above": "Above only", "below": "Below only",
    "white": "White", "transparent": "Transparent",
    "line": "Line from the samples", "dots": "Small dots (no line)", "min": "Subtract the minimum", "max": "To the maximum",
    "window": "Inside a window",
}
HATCH_LABELS = {"": "None", "/": "Diagonal", "//": "Dense diagonal", "\\": "Backward diagonal",
                "\\\\": "Dense backward diagonal", "|": "Vertical", "-": "Horizontal", "+": "Grid", "x": "Cross-hatch",
                "o": "Circles", ".": "Dots", "*": "Stars"}
# which controls of the Series section a graph kind uses
NO_SERIES = ("heatmap", "correlation_matrix", "scree", "pca_scores", "pca_loadings", "biplot")
MARKER_KINDS = ("xy", "column", "grouped", "nested", "qq", "residuals", "histogram", "trace")
LINE_KINDS = ("xy", "column", "grouped", "nested", "survival", "roc", "histogram", "trace")
ERROR_KINDS = ("xy", "column", "grouped", "nested")
FILL_KINDS = ("column", "grouped", "histogram", "contingency")
BAND_KINDS = ("xy", "survival")
DEBOUNCE_MS = 80  # redraw delay after the last slider move
_style_clipboard: dict = {}  # "template": the copied style, shared by all graphs of this session


def _surround_css() -> str:
    """Background around the white figure canvas: the theme's panel colour (grey fallback without the theme)."""
    try:
        from . import theme
        return f"QScrollArea {{ background: {theme.PANEL}; }}"
    except (ImportError, AttributeError):
        return "QScrollArea { background: #e9ecef; }"


LOOK_LABELS = {"default": "Default", "bold": "Bold publication"}


def human(v) -> str:
    """Readable text for an internal choice id."""
    return LABELS[v] if v in LABELS else str(v).replace("_", " ").capitalize()


def swatch_icon(color: str) -> QIcon:
    pm = QPixmap(16, 16)
    pm.fill(QColor(color))
    p = QPainter(pm)
    p.setPen(QApplication.palette().color(QPalette.Mid))
    p.drawRect(0, 0, 15, 15)
    p.end()
    return QIcon(pm)


def strip_icon(colors) -> QIcon:
    """A row of colour chips (palette preview)."""
    pm = QPixmap(8 * len(colors), 12)
    p = QPainter(pm)
    for i, c in enumerate(colors):
        p.fillRect(8 * i, 0, 8, 12, QColor(c))
    p.end()
    return QIcon(pm)


def line_icon(style: str, dash: str = "") -> QIcon:
    """The look of a line style, drawn with Qt's own dash patterns (the pen uses the text colour)."""
    pm = QPixmap(56, 14)
    pm.fill(Qt.transparent)
    pen = QPen(QApplication.palette().color(QPalette.Text), 2)
    pen.setCapStyle(Qt.FlatCap)
    if style == "custom":
        try:
            pen.setDashPattern(list(plot.linestyle("custom", dash)[1]))  # in pen widths, like Matplotlib's
        except ValueError:
            pen.setStyle(Qt.SolidLine)
    else:
        pen.setStyle({"solid": Qt.SolidLine, "dashed": Qt.DashLine, "dotted": Qt.DotLine,
                      "dashdot": Qt.DashDotLine}[style])
    p = QPainter(pm)
    p.setPen(pen)
    p.drawLine(2, 7, 54, 7)
    p.end()
    return QIcon(pm)


class Section(QGroupBox):
    """Collapsible group box: unticking the title hides the rows (``form``) under it."""

    def __init__(self, title: str):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(True)
        self.setStyleSheet("QGroupBox { font-weight: 700 } QGroupBox > QWidget { font-weight: 700 }")
        body = QWidget()
        self.form = QFormLayout(body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setRowWrapPolicy(QFormLayout.WrapLongRows)  # a narrow panel puts a long label above its field
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 8)
        lay.addWidget(body)
        self.toggled.connect(body.setVisible)


def make_canvas(fig, zoom: int) -> FigureCanvasQTAgg:
    """Canvas showing ``fig`` at its physical proportions (mm sizes) scaled by ``zoom`` percent."""
    dpi = BASE_DPI * zoom / 100
    fig.set_dpi(dpi)
    fig._original_dpi = dpi  # the Qt backend multiplies this by the screen scale factor
    c = FigureCanvasQTAgg(fig)
    c.setFixedSize(round(fig.get_figwidth() * dpi), round(fig.get_figheight() * dpi))
    return c


def export_item(parent, app, kind: str, item_id: str) -> None:
    """Ask for a file name and options, then export one graph or layout."""
    dlg = ExportDialog(parent)
    if not dlg.exec():
        return
    v = dlg.values()
    try:
        app.export_figure(kind, item_id, v["target"], v["dpi"], v["text_as_paths"])
        QMessageBox.information(parent, "Export", f"Saved {v['target']}")
    except Exception as exc:
        QMessageBox.warning(parent, "Export failed", str(exc))


def batch_export(parent, app) -> None:
    dlg = ExportDialog(parent, batch=True)
    if not dlg.exec():
        return
    v = dlg.values()
    rep = app.batch_export(v["target"], v["fmt"], v["dpi"], v["text_as_paths"])
    bad = [r for r in rep if not r[1]]
    text = f"{len(rep) - len(bad)} of {len(rep)} graphs exported to {v['target']}."
    box = QMessageBox(QMessageBox.Warning if bad else QMessageBox.Information, "Batch export", text, parent=parent)
    box.setDetailedText("\n".join(f"{'ok    ' if ok else 'FAILED'} {name}  {msg if not ok else ''}" for name, ok, msg in rep))
    box.exec()


# ---------------------------------------------------------------------------- direct editing on the canvas
SERIES_SEL = ("series", "legend_entry", "band")  # selections that edit one series
MOVABLE = ("legend", "legend_entry", "title", "axtitle", "text", "bracket")
TEXTUAL = ("title", "axtitle", "legend_entry", "bracket", "text", "annotation")  # double-click opens the inline editor
DRAG_PX = 3  # a press must move this far before it is a drag
NUDGE_MS = 400  # arrow-key nudges are committed (one undo step) this long after the last key
_SELECT_NAMES = {"series": "Data series", "bracket": "Significance bracket", "axis": "Axis", "axtitle": "Axis title",
                 "title": "Graph title", "legend": "Legend", "legend_entry": "Legend entry", "band": "Confidence band",
                 "text": "Text", "annotation": "Annotation", "frame": "Plot area"}


def sel_gids(sel) -> tuple[str, ...]:
    """The gids of the artists that make up a selection."""
    k, i = sel[0], (sel[1] if len(sel) > 1 else "")
    return {"series": (f"series:{i}", f"fit:{i}"), "band": (f"band:{i}",), "legend_entry": (f"legend_entry:{i}",),
            "bracket": (f"bracket:{i}",), "legend": ("legend",), "title": ("title",), "annotation": ("annotation",),
            "text": (f"text:{i}",), "axis": (f"{i}axis",), "axtitle": (f"{i}title",)}.get(k, ())


def move_kind(sel) -> str | None:
    """What dragging this selection moves (a legend entry drags its legend)."""
    return None if not sel else "legend" if sel[0] == "legend_entry" else sel[0] if sel[0] in MOVABLE else None


def quick_fields(view, sel) -> dict:
    """The quick-format controls of a selection: ``cap -> (getter(spec), setter(spec, value))``. A ``hide`` setter
    removes or blanks the object."""
    k, i, kind = sel[0], sel[1], view.spec.kind
    ser = lambda s: next((x for x in s.series if x.dataset_id == i), plot.SeriesStyle(""))  # noqa: E731
    axs = lambda s: s.x_axis if i == "x" else s.y_axis  # noqa: E731
    txt = lambda s: s.texts[i] if isinstance(i, int) and 0 <= i < len(s.texts) else plot.new_text(0, 0)  # noqa: E731
    attrs = lambda f, *names: (lambda s, v: [setattr(f(s), n, v) for n in names])  # noqa: E731
    key = lambda f, name: (lambda s, v: f(s).__setitem__(name, v))  # noqa: E731
    top = lambda s: s  # noqa: E731
    if k in SERIES_SEL and k != "band":
        alphas = ("marker_alpha", "curve_alpha", "err_alpha") + (("fill_alpha",) if kind in FILL_KINDS else ())
        f = {"color": (lambda s: ser(s).color, lambda s, v: plot.recolor(ser(s), v)),
             "alpha": (lambda s: ser(s).marker_alpha, attrs(ser, *alphas)),
             "width": (lambda s: ser(s).line_width, attrs(ser, "line_width")),
             "hide": (None, lambda s, _: setattr(ser(s), "visible", False))}
        if kind in LINE_KINDS:
            f["style"] = (lambda s: ser(s).curve_style, attrs(ser, "curve_style"))
        if kind in MARKER_KINDS:
            f["marker"] = (lambda s: ser(s).marker, attrs(ser, "marker"))
            f["msize"] = (lambda s: ser(s).marker_size, attrs(ser, "marker_size"))
        return f
    if k == "band":
        return {"color": (lambda s: ser(s).band_color or ser(s).color, attrs(ser, "band_color")),
                "alpha": (lambda s: 0.18 if ser(s).band_alpha is None else ser(s).band_alpha, attrs(ser, "band_alpha"))}
    if k == "axis":
        return {"color": (lambda s: s.axis_color, attrs(top, "axis_color")),
                "width": (lambda s: s.axis_width or s.line_width_pt, attrs(top, "axis_width")),
                "size": (lambda s: axs(s).font_size or s.font_size_pt, attrs(axs, "font_size"))}
    if k == "axtitle":
        return {"size": (lambda s: axs(s).font_size or s.font_size_pt, attrs(axs, "font_size")),
                "bold": (lambda s: axs(s).title_bold, attrs(axs, "title_bold")),
                "italic": (lambda s: axs(s).title_italic, attrs(axs, "title_italic"))}
    if k == "title":
        return {"size": (lambda s: s.title_size_pt or 1.1 * s.font_size_pt, attrs(top, "title_size_pt")),
                "bold": (lambda s: s.title_bold, attrs(top, "title_bold")),
                "italic": (lambda s: s.title_italic, attrs(top, "title_italic")),
                "hide": (None, lambda s, _: setattr(s, "title", ""))}
    if k == "legend":
        return {"hide": (None, lambda s, _: setattr(s, "legend", False))}
    if k == "bracket":
        br = lambda s: next((b for b in s.brackets if b.comparison_id == i), plot.BracketSpec(""))  # noqa: E731
        return {"color": (lambda s: s.bracket_color, attrs(top, "bracket_color")),
                "width": (lambda s: s.bracket_width or 0.8 * s.line_width_pt, attrs(top, "bracket_width")),
                "size": (lambda s: s.star_size_pt or s.font_size_pt, attrs(top, "star_size_pt")),
                "hide": (None, lambda s, _: s.brackets.__setitem__(slice(None), [b for b in s.brackets if b is not br(s)]))}
    if k == "text":
        return {"color": (lambda s: txt(s)["color"], key(txt, "color")),
                "size": (lambda s: txt(s)["size_pt"], key(txt, "size_pt")),
                "bold": (lambda s: txt(s)["bold"], key(txt, "bold")),
                "italic": (lambda s: txt(s)["italic"], key(txt, "italic")),
                "hide": (None, lambda s, _: s.texts.__delitem__(i))}
    if k == "annotation":
        return {"hide": (None, lambda s, _: setattr(s, "annotation", ""))}
    return {}


def swatch_colors(spec) -> list[str]:
    """Quick-format swatches: the graph's own series colours, the Macaron palette, black and greys."""
    out: list[str] = []
    for c in [s.color for s in spec.series] + plot.PALETTES["Macaron"] + ["#000000", "#404040", "#7F7F7F", "#BFBFBF"]:
        if c.upper() not in out:
            out.append(c.upper())
    return out[:24]


class QuickFormat(QFrame):
    """Small popup next to a selected object: colour swatches, opacity, line, marker and font controls for it.
    Discrete choices apply at once; sliders and spin boxes after DEBOUNCE_MS. Each application is one undo step."""

    def __init__(self, view, sel):
        super().__init__(view, Qt.Popup)
        from . import theme
        self.view, self.sel, self.pending = view, sel, {}
        self.fields = quick_fields(view, sel)
        self.widgets: dict[str, QWidget] = {}
        self.swatches: dict[str, QToolButton] = {}
        self.setObjectName("quickformat")
        self.setStyleSheet(f"QFrame#quickformat {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER_STRONG}; "
                           f"border-radius: 6px }} QLabel {{ color: {theme.TEXT2} }}")
        self.timer = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self.timer.timeout.connect(self.flush)
        spec, f = view.spec, self.fields
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        title = QLabel(view.sel_label.text())
        title.setStyleSheet(f"font-weight: 700; color: {theme.TEXT}")
        lay.addWidget(title)
        get = lambda cap: f[cap][0](spec)  # noqa: E731
        if "color" in f:
            grid, cur = QGridLayout(), get("color").upper()
            grid.setSpacing(3)
            for n, c in enumerate(swatch_colors(spec)):
                b = QToolButton()
                b.setFixedSize(18, 18)
                b.setStyleSheet(f"QToolButton {{ background: {c}; border: {'2px solid ' + theme.TEXT if c == cur else '1px solid ' + theme.BORDER_STRONG}; "
                                f"border-radius: 3px }}")
                b.clicked.connect(lambda _=False, c=c: self.queue("color", c, now=True))
                grid.addWidget(b, n // 8, n % 8)
                self.swatches[c] = b
            more = QPushButton("More colours...")
            more.clicked.connect(self._more)
            self.widgets["More colours"] = more
            lay.addLayout(grid)
            lay.addWidget(more)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        if "alpha" in f:
            w, num, row = QSlider(Qt.Horizontal), QLabel(), QWidget()
            w.setRange(0, 100)
            w.setValue(round(100 * get("alpha")))
            num.setText(f"{w.value()} %")
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(w, 1)
            h.addWidget(num)
            w.valueChanged.connect(lambda v: (num.setText(f"{v} %"), self.queue("alpha", v / 100)))
            self.widgets["Opacity"] = w
            form.addRow("Opacity", row)
        if "width" in f:
            self._spin(form, "Line width", "width", 0, 8, 0.25)
        if "style" in f:
            w = QComboBox()
            w.setIconSize(QSize(56, 14))
            for v in plot.LINE_STYLES[:-1]:
                w.addItem(line_icon(v), human(v), v)
            w.setCurrentIndex(max(w.findData(get("style")), 0))
            w.activated.connect(lambda _, w=w: self.queue("style", w.currentData(), now=True))
            self.widgets["Line style"] = w
            form.addRow("Line style", w)
        if "marker" in f:
            w = QComboBox()
            for v in plot.MARKERS:
                w.addItem(human(v), v)
            w.setCurrentIndex(max(w.findData(get("marker")), 0))
            w.activated.connect(lambda _, w=w: self.queue("marker", w.currentData(), now=True))
            self.widgets["Marker"] = w
            form.addRow("Marker", w)
            self._spin(form, "Marker size", "msize", 1, 30, 0.5)
        if "size" in f:
            self._spin(form, "Font size", "size", 3, 60, 0.5)
        lay.addLayout(form)
        if "bold" in f:
            row, h = QWidget(), None
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            for cap, caption in (("bold", "B"), ("italic", "I")):
                b = QToolButton()
                b.setText(caption)
                b.setCheckable(True)
                b.setChecked(bool(get(cap)))
                b.setStyleSheet(f"QToolButton {{ min-width: 22px; font-weight: {'700' if cap == 'bold' else '400'}; "
                                f"font-style: {'italic' if cap == 'italic' else 'normal'} }} "
                                f"QToolButton:checked {{ background: {theme.SKY.base} }}")
                b.toggled.connect(lambda on, cap=cap: self.queue(cap, on, now=True))
                self.widgets[cap.capitalize()] = b
                h.addWidget(b)
            h.addStretch(1)
            lay.addWidget(row)
        row = QHBoxLayout()
        if "hide" in f:
            hide = QPushButton("Hide")
            hide.clicked.connect(self._hide)
            self.widgets["Hide"] = hide
            row.addWidget(hide)
        fmt = QPushButton("Format...")
        fmt.clicked.connect(self._format)
        self.widgets["Format"] = fmt
        row.addWidget(fmt)
        lay.addLayout(row)
        self.setFixedWidth(max(230, self.sizeHint().width()))

    def _spin(self, form, label, cap, lo, hi, step) -> None:
        w = QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=2)
        w.setValue(float(self.fields[cap][0](self.view.spec)))
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self.queue(cap, v))
        self.widgets[label] = w
        form.addRow(label, w)

    def queue(self, cap, value, now=False) -> None:
        self.pending[cap] = value
        if now:
            self.flush()
        else:
            self.timer.start()

    def flush(self) -> None:
        pend, self.pending = self.pending, {}
        if pend:
            self.view._change(lambda s: [self.fields[c][1](s, v) for c, v in pend.items()], "Quick format")

    def _more(self) -> None:
        self.hide()
        c = QColorDialog.getColor(QColor(self.fields["color"][0](self.view.spec)), self.view, "Colour")
        if c.isValid():
            self.queue("color", c.name().upper(), now=True)

    def _hide(self) -> None:
        self.timer.stop()
        self.close()
        self.view._change(lambda s: self.fields["hide"][1](s, None), f"Hide {_SELECT_NAMES[self.sel[0]].lower()}")
        self.view.select(None)

    def _format(self) -> None:
        self.flush()
        self.close()
        self.view.focus_panel()


class _InlineEdit(QLineEdit):
    """One-line editor laid over a text of the canvas: Enter / focus-out commit, Esc cancels."""
    cancelled = Signal()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(e)


class CanvasEditor:
    """Prism-like direct editing of one graph canvas (a new one is made with every rebuilt figure).

    Hover and selection feedback, drags and the corner resize only restore the cached background and draw the few
    artists involved (blitting); the figure is rebuilt once, by the app, when an edit is committed."""

    def __init__(self, view, canvas):
        from . import theme
        self.view, self.canvas, self.fig = view, canvas, canvas.figure
        self.ax = self.fig.axes[0]
        self.accent, self.glow_color = theme.SKY.ink, theme.SKY.deep
        self.bg = None  # the drawn figure, without the overlay (and without the artist being dragged)
        self._cand: list | None = None
        self.hover = self.pend = self.drag = self.field = None
        self.hover_corner = False  # the pointer is on the resize corner of the plot area
        self.nudge = QTimer(canvas, singleShot=True, interval=NUDGE_MS)
        self.nudge.timeout.connect(lambda: self._end(True))
        for name, fn in (("draw_event", self._on_draw), ("button_press_event", self._press),
                         ("button_release_event", self._release), ("motion_notify_event", self._motion),
                         ("key_press_event", self._key), ("figure_leave_event", self._leave)):
            canvas.mpl_connect(name, fn)
        canvas.setFocusPolicy(Qt.StrongFocus)
        self.tool = QToolButton(canvas)  # the small floating toolbar: opens the quick-format popup
        self.tool.setText("...")
        self.tool.setToolTip("Quick format")
        self.tool.setStyleSheet(f"QToolButton {{ background: {theme.SURFACE}; border: 1px solid {self.accent}; "
                                f"border-radius: 4px; padding: 0 5px; color: {theme.TEXT} }}")
        self.tool.clicked.connect(lambda: view.open_quick(view.sel))
        self.tool.hide()

    # ---------------------------------------------------------------- geometry and hit testing
    @property
    def dpr(self) -> float:
        return self.canvas.devicePixelRatio()

    def candidates(self) -> list:
        if self._cand is None:
            self._cand = self.fig.findobj(lambda a: bool(a.get_gid()) and a.get_gid().partition(":")[0] in _GID_KINDS)
        return self._cand

    def group(self, sel) -> list:
        gids = sel_gids(sel)
        return [a for a in self.candidates() if a.get_gid() in gids and a.get_visible() and not isinstance(a, Axis)]

    def sel_bbox(self, sel):
        """Display-space box around the artists of ``sel`` (None when it has none)."""
        r = self.canvas.get_renderer()
        if sel[0] == "frame":
            return self.ax.get_window_extent(r)
        boxes = []
        for a in self.group(sel):
            try:
                bb = a.get_window_extent(r)
            except Exception:  # noqa: BLE001  an artist without a usable extent is skipped
                continue
            if np.isfinite(bb.extents).all() and (bb.width > 0 or bb.height > 0):
                boxes.append(bb)
        return Bbox.union(boxes) if boxes else None

    def _inside(self, a, ev, r) -> bool:
        if isinstance(a, (Text, Legend)):
            bb = a.get_window_extent(r)
            return bb.width > 0 and bb.x0 - 2 <= ev.x <= bb.x1 + 2 and bb.y0 - 2 <= ev.y <= bb.y1 + 2
        try:
            return bool(a.contains(ev)[0])
        except Exception:  # noqa: BLE001
            return False

    def resolve_at(self, ev):
        """(selection, artists under the pointer). Empty plot area = the frame; outside = None."""
        r = self.canvas.get_renderer()
        hits = [a for a in self.candidates() if a.get_visible() and self._inside(a, ev, r)][::-1]  # topmost first
        sel = GraphView.resolve([a.get_gid() for a in hits])
        if sel is None and self.ax.get_window_extent(r).contains(ev.x, ev.y):
            sel = ("frame", "")
        return sel, hits

    def near_corner(self, ev) -> bool:
        bb, tol = self.ax.get_window_extent(self.canvas.get_renderer()), 8 * self.dpr
        return abs(ev.x - bb.x1) <= tol and abs(ev.y - bb.y0) <= tol

    # ---------------------------------------------------------------- drawing the overlay (blitting)
    def _on_draw(self, _event) -> None:
        self._cand = None
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)
        if self.drag is None or self.drag.get("armed"):
            self._overlay()

    def _paint(self) -> None:
        """Restore the background, draw the overlay and the dragged artists, blit; one cheap step per mouse move."""
        if self.bg is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self.bg)
        self._overlay()
        self.canvas.blit(self.fig.bbox)

    def _draw(self, art, r) -> None:
        art.set_figure(self.fig)
        art.draw(r)

    def _rect(self, bb, r, ec=None, lw=0.8, ls="-", fc="none", alpha=1.0, pad=0.0) -> None:
        self._draw(Rectangle((bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad, transform=IdentityTransform(),
                             fill=fc != "none", fc=fc, ec=ec or self.accent, lw=lw, ls=ls, alpha=alpha), r)

    def _handle(self, x, y, r) -> None:
        d = self.dpr
        self._rect(Bbox.from_bounds(x - 3 * d, y - 3 * d, 6 * d, 6 * d), r, fc="white")

    def _overlay(self) -> None:
        r, d, sel = self.canvas.get_renderer(), self.dpr, self.view.sel
        if self.drag:  # the dragged artists, hidden while the background was taken
            for a in self.drag["arts"]:
                a.draw(r)
        if self.hover and self.hover != sel:
            self._glow(self.hover, r)
        if sel:
            bb = self.sel_bbox(sel)
            if bb is not None:
                self._rect(bb, r, ls=(0, (4, 2)), pad=3 * d)
                if move_kind(sel):  # corner handles: the object can be moved
                    for x in (bb.x0 - 3 * d, bb.x1 + 3 * d):
                        for y in (bb.y0 - 3 * d, bb.y1 + 3 * d):
                            self._handle(x, y, r)
            if sel[0] == "series":  # a handle on every data point of the series
                for a in self.group(sel):
                    if isinstance(a, Line2D) and a.get_marker() not in ("None", None, "", "_", "|") and len(a.get_xdata()):
                        pts = a.get_transform().transform(a.get_xydata())
                        for x, y in pts[::len(pts) // 40 + 1]:
                            self._handle(x, y, r)
        if sel == ("frame", "") or self.hover_corner:
            ab = self.ax.get_window_extent(r)
            self._rect(Bbox.from_bounds(ab.x1 - 4 * d, ab.y0 - 4 * d, 8 * d, 8 * d), r, fc=self.accent)
        if self.drag and self.drag["kind"] == "resize":
            self._rect(self.drag["ghost"], r, ls=(0, (4, 2)), lw=1.0)

    def _glow(self, sel, r) -> None:
        """Hover feedback: a translucent halo along lines and error bars, an outline round the other artists."""
        d, texts = self.dpr, []
        for a in self.group(sel):
            if isinstance(a, Line2D):
                ln = a.get_linestyle() not in ("None", "none", " ", "")
                g = Line2D(*a.get_data(orig=False), transform=a.get_transform(), color=self.glow_color, alpha=0.55,
                           lw=a.get_linewidth() + 4 if ln else 0, ls="-" if ln else "None", drawstyle=a.get_drawstyle(),
                           marker=a.get_marker() if a.get_marker() not in ("None", None, "") else None,
                           ms=a.get_markersize() + 4, mfc=self.glow_color, mec=self.glow_color, mew=0)
                if a.get_clip_on():
                    g.set_clip_path(self.ax.patch)
                self._draw(g, r)
            elif isinstance(a, LineCollection):
                g = LineCollection(a.get_segments(), colors=self.glow_color, linewidths=[w + 4 for w in a.get_linewidths()],
                                   alpha=0.55, transform=a.get_transform())
                g.set_clip_path(self.ax.patch)
                self._draw(g, r)
            elif isinstance(a, Text):
                texts.append(a.get_window_extent(r))
            else:
                self._rect(a.get_window_extent(r), r, ec=self.glow_color, lw=1.6, pad=2 * d)
        if texts:
            self._rect(Bbox.union(texts), r, ec=self.glow_color, lw=1.6, pad=2 * d)

    def sync(self) -> None:
        """The view selected something (a click here or in the panel): repaint the overlay, place the toolbar."""
        sel = self.view.sel
        bb = self.sel_bbox(sel) if sel and sel[0] != "frame" else None
        self.tool.setVisible(bb is not None and self.drag is None)
        if bb is not None:
            self.tool.adjustSize()
            x, y = bb.x1 / self.dpr + 6, (self.fig.bbox.height - bb.y1) / self.dpr - 2
            self.tool.move(int(min(max(x, 0), self.canvas.width() - self.tool.width())), int(max(y, 0)))
        self._paint()

    def _set_cursor(self, shape) -> None:
        self.canvas.setCursor(shape)

    # ---------------------------------------------------------------- mouse
    def _leave(self, _ev) -> None:
        if self.hover is not None or self.hover_corner:
            self.hover, self.hover_corner = None, False
            self._paint()

    def _motion(self, ev) -> None:
        if ev.x is None:
            return
        if self.pend or self.drag:
            return self._drag_to(ev)
        corner = self.near_corner(ev)
        sel, _ = self.resolve_at(ev)
        hv = None if corner or sel is None or sel[0] == "frame" else sel
        self._set_cursor(Qt.SizeFDiagCursor if corner else Qt.SizeVerCursor if hv and hv[0] == "bracket"
                         else Qt.SizeAllCursor if move_kind(hv) else Qt.ArrowCursor)
        if hv != self.hover or corner != self.hover_corner:
            self.hover, self.hover_corner = hv, corner
            self._paint()

    def _press(self, ev) -> None:
        if ev.x is None:
            return
        self.canvas.setFocus()
        self.hover = None
        if ev.button == 1 and not ev.dblclick and self.near_corner(ev):
            self.view.select(("frame", ""))
            self.pend = {"kind": "resize", "sel": ("frame", ""), "x0": ev.x, "y0": ev.y}
            return
        sel, hits = self.resolve_at(ev)
        self.view.select(sel)
        if ev.button == 3:
            if sel and sel[0] != "frame":
                self.view.open_quick(sel, ev)
        elif ev.dblclick:
            self._double_click(ev, sel, hits)
        elif ev.button == 1 and move_kind(sel):
            self.pend = {"kind": move_kind(sel), "sel": sel, "x0": ev.x, "y0": ev.y}

    def _double_click(self, ev, sel, hits) -> None:
        if sel and sel[0] in TEXTUAL and any(isinstance(a, Text) and a.get_gid() in sel_gids(sel) for a in hits):
            self.edit(sel)
        elif sel == ("frame", ""):  # empty plot area: a new free text, opened for typing
            x, y = self.ax.transAxes.inverted().transform((ev.x, ev.y))
            n = len(self.view.spec.texts)
            self.view._change(lambda s: s.texts.append(plot.new_text(round(float(x), 4), round(float(y), 4),
                                                                      size_pt=s.font_size_pt)), "Add text")
            self.view.select(("text", n))
            self.view.editor.edit(("text", n))

    def _release(self, _ev) -> None:
        if self.drag and not self.drag.get("nudge"):
            self._end(True)
        self.pend = None

    # ---------------------------------------------------------------- drags
    def _drag_to(self, ev) -> None:
        p = self.drag or self.pend
        dx, dy = ev.x - p["x0"], ev.y - p["y0"]
        if self.drag is None:
            if max(abs(dx), abs(dy)) < DRAG_PX * self.dpr:
                return
            self._begin(self.pend)
            self.pend = None
        self._move(dx, dy)

    def _artists(self, kind, sel) -> list:
        if kind == "legend":
            return [self.ax.get_legend()]
        if kind == "title":
            return [self.ax.title]
        if kind == "axtitle":
            return [(self.ax.xaxis if sel[1] == "x" else self.ax.yaxis).label]
        return self.group(sel) if kind in ("text", "bracket") else []

    def _begin(self, p) -> None:
        """Start a drag: the artists leave the cached background (one full draw of the existing figure, no rebuild)."""
        kind, sel, r = p["kind"], p["sel"], self.canvas.get_renderer()
        if kind == "legend" and sel != ("legend", ""):  # dragging an entry moves, and so selects, the whole legend
            self.view.select(sel := ("legend", ""))
        arts =[a for a in self._artists(kind, sel) if a is not None]
        spec, ab = self.view.spec, self.ax.get_window_extent(r)
        d = dict(p, arts=arts, dx=0.0, dy=0.0, armed=False, ab=ab)
        if kind == "legend" and arts:
            bb = arts[0].get_window_extent(r)
            d["fx"], d["fy"] = bb.x0, bb.y1  # upper-left corner in display pixels
        elif kind in ("title", "axtitle"):
            o = spec if kind == "title" else (spec.x_axis if sel[1] == "x" else spec.y_axis)
            d["o"] = (o.title_dx, o.title_dy)
        elif kind == "text" and arts:
            d["o"] = tuple(arts[0].get_position())
        elif kind == "bracket":
            d["o"] = {id(a): (a.get_ydata().copy() if isinstance(a, Line2D) else np.array(a.xy)) for a in arts}
        elif kind == "resize":
            d["ghost"] = Bbox(ab)
        self.drag = d
        self.tool.hide()
        for a in arts:
            a.set_visible(False)
        self.canvas.draw()  # the background without them; the draw event stores it
        for a in arts:
            a.set_visible(True)
        d["armed"] = True

    def _move(self, dx, dy) -> None:
        d, ax = self.drag, self.ax
        kind, pt = d["kind"], 72 / self.fig.dpi  # pixels -> points
        d["dx"], d["dy"] = dx, dy
        if kind == "legend" and d["arts"]:
            leg, ab = d["arts"][0], d["ab"]
            leg.set_loc("upper left")
            leg.borderaxespad = 0
            leg.set_bbox_to_anchor(((d["fx"] + dx - ab.x0) / ab.width, (d["fy"] + dy - ab.y0) / ab.height), ax.transAxes)
        elif kind in ("title", "axtitle") and d["arts"]:
            plot.set_offset(d["arts"][0], d["o"][0] + dx * pt, d["o"][1] + dy * pt)
        elif kind == "text" and d["arts"]:
            d["arts"][0].set_position((d["o"][0] + dx / d["ab"].width, d["o"][1] + dy / d["ab"].height))
        elif kind == "bracket":
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            shift = dy * span / d["ab"].height  # data units
            for a in d["arts"]:
                o = d["o"][id(a)]
                if isinstance(a, Line2D):
                    a.set_ydata(o + shift)
                else:
                    a.xy = (o[0], o[1] + shift)
        elif kind == "resize":
            ab = d["ab"]
            d["ghost"] = Bbox.from_extents(ab.x0, ab.y0 + dy, ab.x1 + dx, ab.y1)
        self._paint()

    def _end(self, commit: bool) -> None:
        d, self.drag = self.drag, None
        self.nudge.stop()
        if d is None:
            return
        kind, v, mm = d["kind"], self.view, 25.4 / self.fig.dpi
        dx, dy, pt, ab = d["dx"], d["dy"], 72 / self.fig.dpi, d["ab"]
        if not commit:
            return v.refresh()
        if kind == "legend":
            v.commit_legend(d["fx"] + dx, d["fy"] + dy, self.fig.bbox)
        elif kind == "title":
            v._change(lambda s: (setattr(s, "title_dx", round(d["o"][0] + dx * pt, 2)),
                                 setattr(s, "title_dy", round(d["o"][1] + dy * pt, 2))), "Move title")
        elif kind == "axtitle":
            a = lambda s: s.x_axis if d["sel"][1] == "x" else s.y_axis  # noqa: E731
            v._change(lambda s: (setattr(a(s), "title_dx", round(d["o"][0] + dx * pt, 2)),
                                 setattr(a(s), "title_dy", round(d["o"][1] + dy * pt, 2))), "Move axis title")
        elif kind == "text":
            i = d["sel"][1]
            v._change(lambda s: s.texts[i].update(x=round(d["o"][0] + dx / ab.width, 4), y=round(d["o"][1] + dy / ab.height, 4)),
                      "Move text")
        elif kind == "bracket":
            cid = d["sel"][1]

            def mutate(s):
                b = next((b for b in s.brackets if b.comparison_id == cid), None)
                if b is not None:
                    b.dy_pt = round(b.dy_pt + dy * pt, 2)
            v._change(mutate, "Move bracket")
        elif kind == "resize":
            v._change(lambda s: (setattr(s, "width_mm", max(20.0, round(s.width_mm + dx * mm, 1))),
                                 setattr(s, "height_mm", max(20.0, round(s.height_mm - dy * mm, 1)))), "Resize graph")
        if v.editor is not self:  # the figure was rebuilt; this canvas is gone
            return
        v.refresh()  # the edit changed nothing: put the artists back

    # ---------------------------------------------------------------- keyboard
    def _key(self, ev) -> None:
        k = ev.key or ""
        sel = self.view.sel
        if k == "escape":
            if self.drag:
                self._end(False)
            self.pend = None
            self.view.select(None)
        elif k in ("delete", "backspace") and sel:
            self.view.delete_selected()
        elif k.split("+")[-1] in ("left", "right", "up", "down") and move_kind(sel):
            step = (10 if "shift" in k else 1) * self.dpr
            ux, uy = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}[k.split("+")[-1]]
            if self.drag is None:
                self._begin({"kind": move_kind(sel), "sel": sel, "x0": 0, "y0": 0})
                self.drag["nudge"] = True
            self._move(self.drag["dx"] + ux * step, self.drag["dy"] + uy * step)
            self.nudge.start()

    # ---------------------------------------------------------------- inline text editing
    def _text_artist(self, sel):
        found = [a for a in self.group(sel) if isinstance(a, Text)]
        return found[0] if found else None

    def edit(self, sel) -> bool:
        """Open the inline editor over the text of ``sel``; Enter / focus-out commit one undo step, Esc cancels."""
        a = self._text_artist(sel)
        if a is None:
            return False
        self.close_field(True)
        bb, d = a.get_window_extent(self.canvas.get_renderer()), self.dpr
        ed = _InlineEdit(a.get_text(), self.canvas)
        font = QFont(a.get_fontname())
        font.setPixelSize(max(6, round(a.get_fontsize() * self.fig.dpi / d / 72)))
        font.setBold(a.get_fontweight() in ("bold", 700))
        font.setItalic(a.get_fontstyle() == "italic")
        ed.setFont(font)
        ed.setStyleSheet(f"QLineEdit {{ background: white; border: 1px solid {self.accent}; padding: 0 3px; color: black }}")
        ed.setGeometry(round(bb.x0 / d) - 5, round((self.fig.bbox.height - bb.y1) / d) - 4,
                       round(max(bb.width / d + 24, 110)), round(bb.height / d) + 8)
        self.field = (ed, sel, a.get_text())
        self.tool.hide()
        width = ed.width()
        ed.textChanged.connect(lambda t: ed.resize(max(width, ed.fontMetrics().horizontalAdvance(t) + 28), ed.height()))
        ed.returnPressed.connect(lambda: self.close_field(True))
        ed.editingFinished.connect(lambda: self.close_field(True))  # focus-out
        ed.cancelled.connect(lambda: self.close_field(False))
        ed.show()
        ed.setFocus()
        ed.selectAll()
        return True

    def close_field(self, commit: bool) -> None:
        if self.field is None:
            return
        (ed, sel, old), self.field = self.field, None
        txt = ed.text()
        ed.hide()
        ed.deleteLater()
        if commit and txt != old:
            self.view.set_text(sel, txt)
        if self.view.editor is self:  # (an edit rebuilds the canvas, and with it the editor)
            self.sync()


# gid prefixes of the artists the editor can hit
_GID_KINDS = {"series", "fit", "band", "bracket", "legend", "legend_entry", "title", "xtitle", "ytitle", "xaxis", "yaxis",
              "text", "annotation"}


class GraphView(QWidget):
    selection_changed = Signal(object)

    def __init__(self, app, gid: str, parent=None):
        super().__init__(parent)
        self.app, self.gid, self.sel, self.zoom = app, gid, None, DEFAULT_ZOOM
        self._fit_timer = QTimer(self, singleShot=True, interval=150)  # re-fit after the window stops resizing
        self._fit_timer.timeout.connect(self._maybe_refit)
        self._shown_zoom = 0
        self.editor: CanvasEditor | None = None
        self.quick: QuickFormat | None = None
        self._loaders: list = []
        self.widgets: dict[str, QWidget] = {}  # label -> input widget (graph-level fields and the selection pages)
        self._busy = False
        self.canvas: FigureCanvasQTAgg | None = None
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet(_surround_css())
        self.error = QLabel()
        self.error.setStyleSheet("background: #fde7e9; color: #7a0010; padding: 6px")
        self.error.setWordWrap(True)
        self.error.hide()
        zoom = QComboBox()
        for z in ZOOMS:
            zoom.addItem("Fit" if z == "fit" else f"{z}%", z)
        zoom.setCurrentIndex(ZOOMS.index(self.zoom))
        zoom.activated.connect(lambda _: (setattr(self, "zoom", zoom.currentData()), self.refresh()))
        export = QPushButton("Export...")
        export.clicked.connect(lambda: export_item(self, app, "graph", gid))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Preview"))
        bar.addWidget(zoom)
        bar.addStretch(1)
        bar.addWidget(export)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(bar)
        ll.addWidget(self.error)
        ll.addWidget(self.scroll, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self._build_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setCollapsible(1, False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        for sig in (app.graph_changed, app.analysis_changed, app.table_changed):
            sig.connect(self._maybe_refresh)
        if self.zoom != "fit":
            self.refresh()  # "fit": the first draw waits for showEvent, when the preview area has its size (one draw, not two)

    def showEvent(self, event):
        super().showEvent(event)
        if self.canvas is None:
            self.refresh()

    # ------------------------------------------------------------ properties panel
    @property
    def spec(self):
        return self.app.project.graphs[self.gid]

    def _change(self, fn, text="Edit graph"):
        if not self._busy:
            self.app.edit_graph(self.gid, fn, text)

    def _bind(self, form, label, widget, get, set_, text, signal, read, write, key=None):
        """Connect a widget to a spec field: ``get(spec)`` loads it, ``set_(spec, value)`` stores it."""
        def load(s):
            widget.blockSignals(True)
            write(widget, get(s))
            widget.blockSignals(False)
        signal.connect(lambda *_: self._change(lambda s: set_(s, read(widget)), text))
        self._loaders.append(load)
        self.widgets[key or label] = widget
        form.addRow(label, widget)
        return widget

    def _text(self, form, label, get, set_, text, key=None):
        w = QLineEdit()
        return self._bind(form, label, w, get, set_, text, w.editingFinished, lambda w: w.text(), lambda w, v: w.setText(v or ""), key)

    def _opt_num(self, form, label, get, set_, text, key=None):
        w = QLineEdit()
        w.setPlaceholderText("auto")

        def read(w):
            t = w.text().strip()
            try:
                return float(t) if t else None
            except ValueError:
                return "bad"

        def store(s, v):
            if v != "bad":
                set_(s, v)
        return self._bind(form, label, w, get, store, text, w.editingFinished, read,
                          lambda w, v: w.setText("" if v is None else format(v, ".10g")), key)

    def _spin(self, form, label, get, set_, text, lo, hi, step=0.5, auto=False, key=None):
        """``auto``: the minimum value reads "auto" (the spec's 0 / -1 for "automatic")."""
        w = QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=2)
        w.setKeyboardTracking(False)
        if auto:
            w.setSpecialValueText("auto")
        return self._bind(form, label, w, get, set_, text, w.valueChanged, lambda w: w.value(), lambda w, v: w.setValue(v), key)

    def _slider(self, form, label, get, set_, text, auto_pct=100):
        """Alpha 0-100 %. The preview and the undo step follow DEBOUNCE_MS after the last move."""
        w, num, row, timer = QSlider(Qt.Horizontal), QLabel(), QWidget(), QTimer(self)
        w.setRange(0, 100)
        timer.setSingleShot(True)
        timer.setInterval(DEBOUNCE_MS)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(w, 1)
        h.addWidget(num)
        num.setMinimumWidth(40)

        def load(s):
            v = get(s)
            w.blockSignals(True)
            w.setValue(auto_pct if v is None else round(100 * v))
            w.blockSignals(False)
            num.setText(f"{w.value()} %")
        w.valueChanged.connect(lambda v: (num.setText(f"{v} %"), timer.start()))
        timer.timeout.connect(lambda: self._change(lambda s: set_(s, w.value() / 100), text))
        self._loaders.append(load)
        self.widgets[label] = w
        form.addRow(label, row)
        return w

    def _check(self, form, label, get, set_, text, key=None):
        w = QCheckBox()
        return self._bind(form, label, w, get, set_, text, w.toggled, lambda w: w.isChecked(), lambda w, v: w.setChecked(bool(v)), key)

    def _choice(self, form, label, items, get, set_, text, editable=False, key=None):
        w = QComboBox()
        w.setEditable(editable)
        for v in items:
            w.addItem(human(v), v)
        return self._bind(form, label, w, get, set_, text, w.activated,
                          lambda w: w.currentText() if editable else w.currentData(),
                          lambda w, v: w.setCurrentText(v) if editable else w.setCurrentIndex(max(w.findData(v), 0)), key)

    def _color_row(self, form, label, get, set_, text, auto=None, quick=()):
        """Swatch button that opens the colour dialog. ``auto``: function giving the colour shown while the field is
        "" (automatic); it adds an "Auto" reset button. ``quick``: extra (caption, colour) buttons."""
        btn, row = QPushButton(), QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(btn, 1)
        shown = lambda s: get(s) or (auto(s) if auto else "#000000")

        def paint(s):
            v = get(s)
            btn.setIcon(swatch_icon(shown(s)))
            btn.setText(v.upper() if v else "Auto")

        def pick():
            c = QColorDialog.getColor(QColor(shown(self.spec)), self, label)
            if c.isValid():
                self._change(lambda s: set_(s, c.name().upper()), text)
        btn.clicked.connect(pick)
        for caption, value in ([("Auto", "")] if auto else []) + list(quick):
            b = QPushButton(caption)
            b.clicked.connect(lambda _=False, v=value: self._change(lambda s: set_(s, v), text))
            h.addWidget(b)
            self.widgets[f"{label} {caption.lower()}"] = b
        self._loaders.append(paint)
        self.widgets[label] = btn
        form.addRow(label, row)
        return btn

    def _line_style_rows(self, form, ser) -> None:
        """Line style combo with drawn previews and the custom dash pattern field."""
        combo, dash = QComboBox(), QLineEdit()
        combo.setIconSize(QSize(56, 14))
        for v in plot.LINE_STYLES:
            combo.addItem(line_icon(v, "4,2"), human(v), v)
        dash.setPlaceholderText("e.g. 6,2,1,2")
        dash.setToolTip("Custom dash pattern: lengths of ink and gap in line widths")
        get_s, set_s = ser("curve_style")
        get_d, set_d = ser("curve_dash")

        def pick(_):
            v = combo.currentData()
            self._change(lambda s: (set_s(s, v), v == "custom" and not get_d(s) and set_d(s, "4,2")), "Line style")

        def typed():
            try:
                plot.linestyle("custom", dash.text())
            except ValueError as exc:
                dash.setToolTip(str(exc))
                dash.setText(get_d(self.spec))
                return
            self._change(lambda s: (set_d(s, dash.text().strip()), set_s(s, "custom")), "Dash pattern")
        combo.activated.connect(pick)
        dash.editingFinished.connect(typed)

        def load(s):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(get_s(s)), 0))
            combo.setItemIcon(plot.LINE_STYLES.index("custom"), line_icon("custom", get_d(s) or "4,2"))
            combo.blockSignals(False)
            dash.setText(get_d(s))
        self._loaders.append(load)
        self.widgets["Line style"], self.widgets["Dash pattern"] = combo, dash
        form.addRow("Line style", combo)
        form.addRow("Dash pattern", dash)

    def _ser(self, s):
        ds = self.sel[1] if self.sel and self.sel[0] in SERIES_SEL else None
        return next((x for x in s.series if x.dataset_id == ds), plot.SeriesStyle(""))

    def _axis_rows(self, form, which: str, category: bool) -> None:
        """Title, range, ticks and scale of the X or Y axis (a category X axis only has a title)."""
        name = "Right Y axis" if which == "y2" else f"{which.upper()} axis"
        get = {"x": lambda s: s.x_axis, "y": lambda s: s.y_axis, "y2": lambda s: s.trace.y2_axis}[which]
        ax = lambda attr: (lambda s, v: setattr(get(s), attr, v))
        head = QLabel(name)
        head.setStyleSheet("font-weight: 700; margin-top: 4px")
        form.addRow(head)
        self._text(form, "Title", lambda s: get(s).title, ax("title"), f"{name} title", f"{name} title")
        if which != "y2":
            self._spin(form, "Tick label rotation (deg)", lambda s: get(s).tick_rotation, ax("tick_rotation"),
                       f"{name} tick rotation", -90, 90, 5, key=f"{name} tick rotation")
        if category:
            return
        self._opt_num(form, "Minimum", lambda s: get(s).min, ax("min"), f"{name} minimum", f"{name} minimum")
        self._opt_num(form, "Maximum", lambda s: get(s).max, ax("max"), f"{name} maximum", f"{name} maximum")
        self._opt_num(form, "Tick interval", lambda s: get(s).major_tick, ax("major_tick"), f"{name} tick interval", f"{name} tick interval")
        self._check(form, "Minor ticks", lambda s: get(s).minor_ticks, ax("minor_ticks"), f"{name} minor ticks", f"{name} minor ticks")
        if which != "y2":
            self._check(form, "Logarithmic scale", lambda s: get(s).log, ax("log"), f"{name} log scale", f"{name} log scale")
        if which == "x":
            self._choice(form, "Values are log10", ("off", "values", "power"), lambda s: get(s).log10_values,
                         ax("log10_values"), "X axis log10 labels", key="X axis log10")
        if which != "y2":
            self._opt_num(form, "Tick decimals", lambda s: get(s).tick_decimals,
                          lambda s, v: setattr(get(s), "tick_decimals", None if v is None else max(0, round(v))),
                          f"{name} tick decimals", f"{name} tick decimals")
        self._opt_num(form, "Font size (pt)", lambda s: get(s).font_size, ax("font_size"), f"{name} font size", f"{name} font size")
        self._opt_num(form, "Title size (pt)", lambda s: get(s).title_size_pt, ax("title_size_pt"), f"{name} title size",
                      f"{name} title size")

    def _build_panel(self) -> QWidget:
        none = QLabel("Click a data series, axis, title, legend or bracket in the graph to edit it. Double-click a text "
                      "to retype it, double-click empty space to add a text, drag to move, right-click for quick format.")
        none.setWordWrap(True)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.sel_label = QLabel()
        self.sel_label.setStyleSheet("font-weight: 700")
        pages = [none]

        def page():
            w = QWidget()
            f = QFormLayout(w)
            pages.append(w)
            return f

        f = page()  # 1: bracket
        cid = lambda s: next((b for b in s.brackets if self.sel and b.comparison_id == self.sel[1]), plot.BracketSpec(""))
        self._text(f, "Label (blank = P value stars)", lambda s: cid(s).label,
                   lambda s, v: setattr(cid(s), "label", v or None), "Bracket label", "Bracket label")
        hide = QPushButton("Hide bracket")
        hide.clicked.connect(lambda: self._change(lambda s: s.brackets.__setitem__(
            slice(None), [b for b in s.brackets if b.comparison_id != self.sel[1]]), "Hide bracket"))
        f.addRow(hide)

        pages.append(self.hint)  # 2: series / axis / title / legend: edited in a section below

        f = page()  # 3: free text
        tx = lambda s: s.texts[self.sel[1]] if self.sel and self.sel[0] == "text" and self.sel[1] < len(s.texts) \
            else plot.new_text(0, 0)  # a stand-in while no text is selected
        item = lambda name: (lambda s: tx(s)[name], lambda s, v: tx(s).__setitem__(name, v))
        self._text(f, "Text", *item("text"), "Text", "Text content")
        self._spin(f, "Text size (pt)", *item("size_pt"), "Text size", 3, 80)
        self._spin(f, "Rotation (deg)", *item("rotation"), "Text rotation", -180, 180, 5)
        self._color_row(f, "Text colour", *item("color"), "Text colour")
        self._check(f, "Bold", *item("bold"), "Text bold", "Text bold")
        self._check(f, "Italic", *item("italic"), "Text italic", "Text italic")
        delete = QPushButton("Delete text")
        delete.clicked.connect(self.delete_selected)
        f.addRow(delete)

        for p in (none, self.hint):  # word-wrapped labels need their own height
            p.setMinimumHeight(96 if p is none else 48)
            p.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stack = QStackedWidget()
        for p in pages:
            self.stack.addWidget(p)
        self._fit_stack(0)
        sel_box = self.sel_box = QGroupBox("Selected element")
        sl = QVBoxLayout(sel_box)
        sl.addWidget(self.sel_label)
        sl.addWidget(self.stack)

        kind = self.spec.kind
        self.sections = {k: Section(t) for k, t in (("series", "Series"), ("trace", "Trace"),
                                                    ("data", "Data && error bars"), ("axes", "Axes"),
                                                    ("curves", "Curves && bands"), ("brackets", "Brackets"),
                                                    ("text", "Legend && text"), ("style", "Style"),
                                                    ("size", "Size && font"))}
        _, trace, data, axes, curves, brk, text, style, size = (self.sections[k].form for k in self.sections)
        self.fit_combo = self.stats_combo = self.result_combo = self.brackets = self.color_combo = None
        self.series_body = self.series_combo = None
        if kind not in NO_SERIES:
            self._series_rows(self.sections["series"].form, kind)
        self._style_rows(style, kind)

        if kind in plot.STYLES:
            self._choice(data, "Style", plot.STYLES[kind], lambda s: s.style, lambda s, v: setattr(s, "style", v), "Graph style")
        if kind in ("xy", "column", "grouped", "nested"):
            self._choice(data, "Error bars", plot.ERRORS, lambda s: s.error, lambda s, v: setattr(s, "error", v), "Error bars")
        cat = kind in CATEGORY_X
        self._axis_rows(axes, "x", cat)
        self._axis_rows(axes, "y", False)
        self.right_list = self.event_run_combo = self.event_kind_combo = self.peak_combo = self.metric_list = self.kin_table = None
        if kind == "trace":
            self._axis_rows(axes, "y2", False)
            self._trace_rows(trace)

        if kind == "xy":
            self.fit_combo = self._analysis_combo(curves, "Curve from fit", "fit_analysis_id")
        if kind in ("column", "grouped", "nested"):
            self.stats_combo = self._analysis_combo(brk, "Comparisons from", "stats_analysis_id")
            self.brackets = QListWidget()
            self.brackets.setMaximumHeight(140)
            self.brackets.itemChanged.connect(self._bracket_toggled)
            brk.addRow("Show brackets", self.brackets)
            self._color_row(brk, "Bracket colour", lambda s: s.bracket_color, lambda s, v: setattr(s, "bracket_color", v), "Bracket colour")
            self._spin(brk, "Bracket width (pt)", lambda s: s.bracket_width, lambda s, v: setattr(s, "bracket_width", v),
                       "Bracket width", 0, 6, 0.25, auto=True)
            self._spin(brk, "Bracket label size (pt)", lambda s: s.star_size_pt, lambda s, v: setattr(s, "star_size_pt", v),
                       "Bracket label size", 0, 40, auto=True)

        self._check(text, "Show legend", lambda s: s.legend, lambda s, v: setattr(s, "legend", v), "Legend", "Legend")
        self._check(text, "Legend frame", lambda s: s.legend_frame, lambda s, v: setattr(s, "legend_frame", v), "Legend frame")
        self._choice(text, "Legend position", plot.LEGEND_LOCS, lambda s: s.legend_loc, lambda s, v: setattr(s, "legend_loc", v), "Legend position")
        self._text(text, "Graph title", lambda s: s.title, lambda s, v: setattr(s, "title", v), "Graph title", "Title")
        self._kind_options(data, curves, text, kind)

        self._spin(size, "Width (mm)", lambda s: s.width_mm, lambda s, v: setattr(s, "width_mm", v), "Graph width", 20, 500, 1)
        self._spin(size, "Height (mm)", lambda s: s.height_mm, lambda s, v: setattr(s, "height_mm", v), "Graph height", 20, 500, 1)
        self._choice(size, "Font", FONTS, lambda s: s.font_family, lambda s, v: setattr(s, "font_family", v), "Font", editable=True)
        self._spin(size, "Font size (pt)", lambda s: s.font_size_pt, lambda s, v: setattr(s, "font_size_pt", v), "Font size", 4, 40)
        self._spin(size, "Axis line (pt)", lambda s: s.line_width_pt, lambda s, v: setattr(s, "line_width_pt", v), "Axis line width", 0.25, 6, 0.25)
        self._check(size, "Bold tick labels", lambda s: s.tick_bold, lambda s, v: setattr(s, "tick_bold", v), "Bold tick labels")
        for label, attr, what, hi in (("Tick length (pt)", "tick_length_pt", "Tick length", 20), ("Tick width (pt)", "tick_width_pt", "Tick width", 6),
                                      ("Legend text size (pt)", "legend_size_pt", "Legend text size", 40),
                                      ("Legend handle length", "legend_handle_len", "Legend handle length", 10),
                                      ("Legend entry spacing", "legend_spacing", "Legend entry spacing", 5)):
            self._opt_num(size, label, lambda s, a=attr: getattr(s, a), lambda s, v, a=attr: setattr(s, a, v), what, what)

        inner = QWidget()
        il = QVBoxLayout(inner)
        il.addWidget(sel_box)
        for sec in self.sections.values():
            il.addWidget(sec)  # parent first: setVisible on a parentless widget opens it as a window
            sec.setHidden(sec.form.rowCount() == 0)
        il.addStretch(1)
        self.area = QScrollArea(widgetResizable=True)
        self.area.setWidget(inner)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # wide enough for the widest row, so the right edge is never clipped
        self.area.setMinimumWidth(max(320, inner.minimumSizeHint().width() + self.area.verticalScrollBar().sizeHint().width()
                                      + 2 * self.area.frameWidth() + 4))
        return self.area

    def _window(self, form, label, attr):
        """Text field "x0, x1" for a trace window (empty = none)."""
        w = QLineEdit()
        w.setPlaceholderText("x0, x1")

        def read(w):
            t = w.text().strip()
            try:
                a, b = (float(v) for v in re.split(r"[,; ]+", t)) if t else (None, None)
            except ValueError:
                return "bad"
            return [] if a is None else [a, b] if a < b else "bad"
        return self._bind(form, label, w, lambda s: getattr(s.trace, attr),
                          lambda s, v: None if v == "bad" else setattr(s.trace, attr, v), label, w.editingFinished, read,
                          lambda w, v: w.setText(", ".join(format(a, ".10g") for a in v)))

    def _pick_combo(self, form, label, attr, empty: str, text: str) -> QComboBox:
        """Combo of the trace option ``attr`` (a text or a one-item list): "(empty)" = unset; filled by ``_fill_trace``."""
        w = QComboBox()

        def mutate(s):
            v = w.currentData()
            setattr(s.trace, attr, ([v] if v else []) if isinstance(getattr(s.trace, attr), list) else v or "")
        w.activated.connect(lambda _: self._change(mutate, text))
        w.setProperty("empty", empty)
        form.addRow(label, w)
        self.widgets[label] = w
        return w

    def _trace_rows(self, f) -> None:
        """Trace section: right axis, baseline, normalisation, stacking, event marks and peak labels."""
        tr = lambda attr: (lambda s: getattr(s.trace, attr), lambda s, v: setattr(s.trace, attr, v))  # noqa: E731
        self.right_list = QListWidget()
        self.right_list.setMaximumHeight(110)
        self.right_list.itemChanged.connect(self._right_toggled)
        self.widgets["Series on the right axis"] = self.right_list
        f.addRow("Series on the right axis", self.right_list)
        self._choice(f, "Baseline", plot.TRACE_BASELINE, *tr("baseline"), "Baseline", key="Trace baseline")
        self._window(f, "Baseline window", "baseline_window")
        self._choice(f, "Normalise", plot.TRACE_NORMALIZE, *tr("normalize"), "Normalise", key="Trace normalise")
        self._window(f, "Normalise window", "norm_window")
        self._text(f, "Normalised axis title", *tr("norm_title"), "Normalised axis title")
        self._spin(f, "Stack offset", *tr("stack_offset"), "Stack offset", -1e7, 1e7, 0.1)
        self._check(f, "Event marks", *tr("show_events"), "Event marks", key="Trace event marks")
        self.event_run_combo = self._pick_combo(f, "Events of run", "event_run", "(all runs)", "Event run")
        self.event_kind_combo = self._pick_combo(f, "Kind of event", "event_kinds", "(all kinds)", "Event kind")
        self._check(f, "Label peaks", *tr("peaks"), "Peak labels", key="Trace peak labels")
        self._spin(f, "Peak prominence (x Y span)", *tr("peak_prominence"), "Peak prominence", 0.01, 1.0, 0.01)
        self._int(f, "Peak label decimals", *tr("peak_decimals"), "Peak decimals", 0, 4)
        self.peak_combo = self._pick_combo(f, "Peaks of", "peak_series", "(first left trace)", "Peak series")
        if self.spec.trace.bli.sensors:
            self._bli_rows(f)

    def _bli_rows(self, f) -> None:
        """BLI sensorgram options: curves shown, concentration palette, rebased time, phase line / labels, fit style and
        the file metrics printed in the legend."""
        b = lambda attr: (lambda s: getattr(s.trace.bli, attr), lambda s, v: setattr(s.trace.bli, attr, v))  # noqa: E731
        self._choice(f, "BLI curves", plot.BLI_SHOW, *b("show"), "BLI curves", key="BLI show")
        self._choice(f, "BLI palette", plot.BLI_PALETTES, lambda s: s.trace.bli.palette, plot.apply_bli_palette, "BLI palette")
        self._check(f, "Time from association start", *b("rebase"), "BLI rebase", key="BLI rebase")
        self._check(f, "Phase line", *b("phase_line"), "BLI phase line", key="BLI phase line")
        self._check(f, "Phase labels", *b("phase_labels"), "BLI phase labels", key="BLI phase labels")
        self._spin(f, "BLI phase width (pt)", lambda s: s.trace.bli.phase_width or 0.0,
                   lambda s, v: setattr(s.trace.bli, "phase_width", v or None), "BLI phase width", 0.0, 10, 0.25, auto=True)
        self._color_row(f, "BLI phase colour", *b("phase_color"), "BLI phase colour")
        # all raw curves at once (colours stay per concentration); 0 / "Each curve's own" = per-series settings
        self._spin(f, "BLI raw width (pt)", lambda s: s.trace.bli.raw_width or 0.0,
                   lambda s, v: setattr(s.trace.bli, "raw_width", v or None), "BLI raw width", 0.0, 10, 0.25, auto=True)
        self._slider(f, "BLI raw alpha", *b("raw_alpha"), "BLI raw alpha")
        styles = [v for v in plot.LINE_STYLES if v != "custom"]
        w = self._choice(f, "BLI raw line style", [""] + styles, *b("raw_style"), "BLI raw style", key="BLI raw style")
        if isinstance(w, QComboBox):
            w.setItemText(0, "Each curve's own")
        self._color_row(f, "BLI fit colour", *b("fit_color"), "BLI fit colour", lambda s: "#808080")
        self._slider(f, "BLI fit alpha", *b("fit_alpha"), "BLI fit alpha")
        self._spin(f, "BLI fit width (pt)", *b("fit_width"), "BLI fit width", 0.1, 4, 0.1)
        self._choice(f, "BLI fit line style", [v for v in plot.LINE_STYLES if v != "custom"], *b("fit_style"), "BLI fit style",
                     key="BLI fit style")
        self.metric_list = self.widgets["BLI metrics"] = QListWidget()
        self.metric_list.setMaximumHeight(90)
        self.metric_list.itemChanged.connect(self._metric_toggled)
        f.addRow("Metrics in the legend", self.metric_list)
        self._check(f, "Kinetics block", *b("kinetics_show"), "BLI kinetics block", key="BLI kinetics show")
        self.kin_table = self.widgets["BLI kinetics"] = QTableWidget(0, 3)
        self.kin_table.setHorizontalHeaderLabels(["Name", "Value", "Unit"])
        self.kin_table.setMaximumHeight(130)
        self.kin_table.setToolTip("Typed-in values (KD in molar, shown as nM etc.; kon and koff in E notation) or the file's own "
                                  "KD / kon / koff; empty values are not printed. Nothing is computed from the data.")
        self.kin_table.itemChanged.connect(self._kinetics_edited)
        f.addRow("Kinetics values", self.kin_table)
        add, drop = QPushButton("Add row"), QPushButton("Remove row")
        add.clicked.connect(lambda: self._set_kinetics(lambda rows: rows + [{"name": "", "value": None, "unit": ""}], "BLI kinetics row"))
        drop.clicked.connect(lambda: self._set_kinetics(
            lambda rows: [r for i, r in enumerate(rows) if i != (self.kin_table.currentRow() if self.kin_table.currentRow() >= 0 else len(rows) - 1)],
            "BLI kinetics row"))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(add)
        h.addWidget(drop)
        f.addRow(row)
        self.widgets.update({"BLI kinetics add": add, "BLI kinetics remove": drop})

    def _set_kinetics(self, fn, text: str) -> None:
        """Store ``fn(rows)`` as the kinetics rows (the file's own KD / kon / koff while none are stored count as rows)."""
        self._change(lambda s: setattr(s.trace.bli, "kinetics", fn(plot.bli_kinetics(s.trace.bli))), text)

    def _kinetics_edited(self, _item) -> None:
        t, rows = self.kin_table, []
        try:
            for i in range(t.rowCount()):
                cell = lambda c: (t.item(i, c).text().strip() if t.item(i, c) else "")  # noqa: E731
                rows.append({"name": cell(0), "value": float(cell(1)) if cell(1) else None, "unit": cell(2)})
        except ValueError:  # not a number: put the stored value back
            self._fill_trace()
            return
        self._set_kinetics(lambda _: [r for r in rows if r["name"]], "BLI kinetics")

    def _metric_toggled(self, item) -> None:
        name, on = item.data(Qt.UserRole), item.checkState() == Qt.Checked
        self._change(lambda s: setattr(s.trace.bli, "metrics", [m for m in s.trace.bli.metrics if m != name] + ([name] if on else [])),
                     "BLI metrics")

    def _right_toggled(self, item) -> None:
        ds, on = item.data(Qt.UserRole), item.checkState() == Qt.Checked
        self._change(lambda s: setattr(s.trace, "right", [d for d in s.trace.right if d != ds] + ([ds] if on else [])),
                     "Right axis")

    def _fill_trace(self) -> None:
        """Series / run / kind lists of the Trace section follow the spec."""
        if self.right_list is None:
            return
        s = self.spec
        self.right_list.blockSignals(True)
        self.right_list.clear()
        for x in s.series:
            it = QListWidgetItem(x.label or x.dataset_id)
            it.setData(Qt.UserRole, x.dataset_id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if x.dataset_id in s.trace.right else Qt.Unchecked)
            self.right_list.addItem(it)
        self.right_list.blockSignals(False)
        runs = list(dict.fromkeys(e.get("run", "") for e in s.trace.events))
        kinds = list(dict.fromkeys(e.get("kind", "") for e in s.trace.events))
        for w, items, cur in ((self.event_run_combo, [(r, r) for r in runs if r], s.trace.event_run),
                              (self.event_kind_combo, [(k, k) for k in kinds if k], (s.trace.event_kinds or [""])[0]),
                              (self.peak_combo, [(x.label or x.dataset_id, x.dataset_id) for x in s.series],
                               (s.trace.peak_series or [""])[0])):
            w.blockSignals(True)
            w.clear()
            w.addItem(w.property("empty"), "")
            for lab, data in items:
                w.addItem(lab, data)
            w.setCurrentIndex(max(w.findData(cur), 0))
            w.blockSignals(False)
        if self.kin_table is not None:
            self.kin_table.blockSignals(True)
            rows = plot.bli_kinetics(s.trace.bli)
            self.kin_table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                for c, v in enumerate((r["name"], "" if r["value"] is None else format(r["value"], ".6g"), r["unit"])):
                    self.kin_table.setItem(i, c, QTableWidgetItem(v))
            self.kin_table.blockSignals(False)
        if self.metric_list is not None:
            self.metric_list.blockSignals(True)
            self.metric_list.clear()
            for name in plot.bli_metric_names(s.trace.bli):
                it = QListWidgetItem(name)
                it.setData(Qt.UserRole, name)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if name in s.trace.bli.metrics else Qt.Unchecked)
                self.metric_list.addItem(it)
            self.metric_list.blockSignals(False)

    def _series_rows(self, form, kind: str) -> None:
        """Series section: a series picker plus the style of the selected series, grouped by what it styles."""
        body = self.series_body = QWidget()
        self.series_groups: dict[str, Section] = {}  # collapsible groups of the section; a collapsed one is disabled
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QFormLayout()
        top.setRowWrapPolicy(QFormLayout.WrapLongRows)
        lay.addLayout(top)
        self.series_combo = self.widgets["Series"] = QComboBox()
        self.series_combo.activated.connect(
            lambda i: self.select(("series", self.series_combo.itemData(i)) if self.series_combo.itemData(i) else None))
        top.addRow("Series", self.series_combo)
        ser = lambda attr: (lambda s: getattr(self._ser(s), attr), lambda s, v: setattr(self._ser(s), attr, v))
        auto = lambda s: self._ser(s).color
        pastel = lambda s: plot.mpl.colors.to_hex(plot._lighten(self._ser(s).color))

        def group(title, opened=True):
            sec = Section(title)
            lay.addWidget(sec)
            sec.setChecked(opened)
            self.series_groups[title] = sec
            return sec.form
        self._text(top, "Label", *ser("label"), "Series label")
        self._check(top, "Visible", *ser("visible"), "Series visibility")
        self._color_row(top, "Colour", *ser("color"), "Series colour")
        self._spin(top, "Base width (pt)", *ser("line_width"), "Series line width", 0.1, 8, 0.25)
        if kind in MARKER_KINDS:
            f = group("Markers")
            self._choice(f, "Marker", plot.MARKERS, *ser("marker"), "Series marker")
            self._spin(f, "Marker size", *ser("marker_size"), "Marker size", 1, 30)
            self._marker_fill_row(f)
            self._slider(f, "Marker alpha", *ser("marker_alpha"), "Marker alpha")
            self._color_row(f, "Marker edge colour", *ser("marker_edge_color"), "Marker edge colour", auto)
            self._spin(f, "Marker edge width (pt)", *ser("marker_edge_width"), "Marker edge width", 0, 6, 0.25, auto=True)
        if kind in LINE_KINDS:
            f = group("Line and fitted curve")
            self._color_row(f, "Line colour", *ser("curve_color"), "Line colour", auto)
            self._slider(f, "Line alpha", *ser("curve_alpha"), "Line alpha")
            self._spin(f, "Line width (pt)", *ser("curve_width"), "Line width", 0, 8, 0.25, auto=True)
            self._line_style_rows(f, ser)
        if kind in ERROR_KINDS:
            f = group("Error bars", False)
            self._color_row(f, "Error colour", *ser("err_color"), "Error colour", auto, (("Black", "#000000"),))
            self._slider(f, "Error alpha", *ser("err_alpha"), "Error alpha")
            self._spin(f, "Error width (pt)", *ser("err_width"), "Error width", 0, 6, 0.25, auto=True)
            self._spin(f, "Cap length (pt)", *ser("err_cap"), "Cap length", -1, 20, 1, auto=True)
            self._choice(f, "Error direction", plot.ERR_DIRS, *ser("err_dir"), "Error direction")
        if kind in FILL_KINDS:
            f = group("Bars, boxes and fills", False)
            self._check(f, "Filled", *ser("fill"), "Series fill")
            self._color_row(f, "Fill colour", *ser("fill_color"), "Fill colour", pastel)
            self._slider(f, "Fill alpha", *ser("fill_alpha"), "Fill alpha")
            self._color_row(f, "Edge colour", *ser("edge_color"), "Edge colour", auto)
            self._spin(f, "Edge width (pt)", *ser("edge_width"), "Edge width", 0, 6, 0.25, auto=True)
            hatch = QComboBox()
            for v in plot.HATCHES:
                hatch.addItem(HATCH_LABELS[v], v)
            self._bind(f, "Hatch", hatch, *ser("hatch"), "Hatch", hatch.activated, lambda w: w.currentData(),
                       lambda w, v: w.setCurrentIndex(max(w.findData(v), 0)))
        if kind in BAND_KINDS:
            f = group("Confidence band", False)
            self._color_row(f, "Band colour", *ser("band_color"), "Band colour", auto)
            self._slider(f, "Band alpha", *ser("band_alpha"), "Band alpha", 18)
        form.addRow(body)

    def _marker_fill_row(self, form) -> None:
        """Filled / hollow / custom fill colour (the colour dialog opens when "Custom colour" is chosen)."""
        w = QComboBox()
        for v in ("filled", "hollow", "custom_fill"):
            w.addItem(human(v), v)

        def mode(s):
            x = self._ser(s)
            f = x.marker_fill or ("filled" if x.fill else "hollow")
            return f if f in ("filled", "hollow") else "custom_fill"

        def pick(i):
            v, cur = w.itemData(i), self._ser(self.spec)
            if v == "custom_fill":
                c = QColorDialog.getColor(QColor(cur.marker_fill if cur.marker_fill.startswith("#") else cur.color),
                                          self, "Marker fill colour")
                if not c.isValid():
                    self._load()
                    return
                v = c.name().upper()
            self._change(lambda s: setattr(self._ser(s), "marker_fill", v), "Marker fill")

        def load(s):
            w.blockSignals(True)
            w.setCurrentIndex(w.findData(mode(s)))
            w.blockSignals(False)
        w.activated.connect(pick)
        self._loaders.append(load)
        self.widgets["Marker fill"] = w
        form.addRow("Marker fill", w)

    def _style_rows(self, form, kind: str) -> None:
        """Graph-level style: palette, axis colour and width, background, style copy / paste / files."""
        put = lambda attr: (lambda s: getattr(s, attr), lambda s, v: setattr(s, attr, v))
        pal = self.widgets["Palette"] = QComboBox()
        pal.addItem("Choose a palette...", None)
        for name, cols in plot.PALETTES.items():
            pal.addItem(strip_icon(cols[:6]), name, name)
        pal.activated.connect(self._palette_picked)
        form.addRow("Palette (all series)", pal)
        look = self.widgets["Look"] = QComboBox()
        look.addItem("Choose a look...", None)
        for name in plot.LOOKS:
            look.addItem(LOOK_LABELS[name], name)
        look.activated.connect(self._look_picked)
        form.addRow("Look", look)
        self._color_row(form, "Axis colour", *put("axis_color"), "Axis colour")
        self._spin(form, "Axis width (pt)", *put("axis_width"), "Axis width", 0, 6, 0.25, auto=True)
        self._choice(form, "Background", plot.BACKGROUNDS, *put("background"), "Background")
        self.apply_all = QPushButton("Apply to all series")
        self.keep_colors = QCheckBox("Keep colours and markers")
        self.keep_colors.setChecked(True)
        self.apply_all.setToolTip("Copy the selected series' style to every series")
        self.apply_all.clicked.connect(lambda: self._change(lambda s: plot.apply_style(
            s, plot.style_template(s, self.sel[1]), graph=False, keep_color=self.keep_colors.isChecked()), "Style to all series"))
        form.addRow(self.apply_all)
        form.addRow(self.keep_colors)
        copy, paste = QPushButton("Copy style"), QPushButton("Paste style")
        copy.clicked.connect(lambda: _style_clipboard.update(template=plot.style_template(self.spec)))
        paste.clicked.connect(self._paste_style)
        save, load = QPushButton("Save style..."), QPushButton("Load style...")
        save.clicked.connect(self._save_style)
        load.clicked.connect(self._load_style)
        for a, b in ((copy, paste), (save, load)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(a)
            h.addWidget(b)
            form.addRow(row)
        self.widgets.update({"Copy style": copy, "Paste style": paste, "Save style": save, "Load style": load,
                             "Apply to all": self.apply_all})

    def _look_picked(self, i: int) -> None:
        name = self.widgets["Look"].itemData(i)
        self.widgets["Look"].setCurrentIndex(0)  # an action, like the palette: one undo step
        if name:
            self._change(lambda s: plot.apply_look(s, name), f"Look: {LOOK_LABELS[name]}")

    def _palette_picked(self, i: int) -> None:
        name = self.widgets["Palette"].itemData(i)
        self.widgets["Palette"].setCurrentIndex(0)
        if name:
            self._change(lambda s: plot.apply_palette(s, name), f"Palette: {name}")

    def _paste_style(self) -> None:
        tmpl = _style_clipboard.get("template")
        if tmpl is None:
            QMessageBox.information(self, "Paste style", "Copy the style of a graph first.")
        else:
            self._change(lambda s: plot.apply_style(s, tmpl), "Paste style")

    def _save_style(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save style", "style.json", "Style files (*.json)")
        if path:
            try:
                plot.save_style(plot.style_template(self.spec), path)
            except OSError as exc:
                QMessageBox.warning(self, "Save style", str(exc))

    def _load_style(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load style", "", "Style files (*.json)")
        if not path:
            return
        try:
            tmpl = plot.load_style(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load style", str(exc))
            return
        _style_clipboard["template"] = tmpl
        self._change(lambda s: plot.apply_style(s, tmpl), "Load style")

    def _fit_stack(self, page: int) -> None:
        """Let the stack be as tall as its current page only."""
        for i in range(self.stack.count()):
            self.stack.widget(i).setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred if i == page else QSizePolicy.Ignored)
        self.stack.updateGeometry()

    def _analysis_combo(self, form, label, attr) -> QComboBox:
        w = QComboBox()

        def mutate(s):
            setattr(s, attr, w.currentData())
            if attr == "stats_analysis_id":  # a new test: its significant comparisons get brackets
                s.brackets = plot.default_brackets(self.app.results().get(w.currentData()))
        w.activated.connect(lambda _: self._change(mutate, label))
        form.addRow(label, w)
        return w

    def _int(self, form, label, get, set_, text, lo, hi):
        w = QSpinBox(minimum=lo, maximum=hi)
        w.setKeyboardTracking(False)
        return self._bind(form, label, w, get, set_, text, w.valueChanged, lambda w: w.value(), lambda w, v: w.setValue(int(v)))

    def _kind_options(self, f, curves, text, kind: str) -> None:
        """Rows of the PlotSpec options that matter for this graph kind (data / curve / text sections)."""
        put = lambda attr: (lambda s: getattr(s, attr), lambda s, v: setattr(s, attr, v))
        mv = self.app.project.tables[self.spec.table_id].kind == "MultipleVariables"
        if kind in plot.RESULT_HAS:
            self.result_combo = self._analysis_combo(f, "Result", "analysis_id")
        if kind in plot.VALUES:
            opts = plot.VALUES[kind]
            self._choice(f, "Values", opts, lambda s: s.value or opts[0], lambda s, v: setattr(s, "value", v), "Graph values")
        if kind == "xy" and not mv:
            self._choice(curves, "Band", plot.BANDS, *put("band"), "Confidence / prediction band")
            self._choice(curves, "Band style", plot.BAND_STYLES, *put("band_style"), "Band style")
            self._check(curves, "Interpolated points", *put("show_interpolated"), "Interpolated points")
        if kind == "survival":
            for label, attr in (("Percent survival", "percent"), ("Confidence band", "show_ci"),
                                ("Censor marks", "show_censor"), ("Number at risk", "show_at_risk")):
                self._check(f, label, *put(attr), label)
        if kind in ("pca_scores", "pca_loadings", "biplot"):
            self._int(f, "PC on X", *put("pc_x"), "PC on X", 1, 50)
            self._int(f, "PC on Y", *put("pc_y"), "PC on Y", 1, 50)
        if kind in ("pca_scores", "biplot"):
            self.color_combo = QComboBox()
            self.color_combo.activated.connect(lambda _: self._change(lambda s: setattr(s, "color_by", self.color_combo.currentData()), "Colour by"))
            f.addRow("Colour by", self.color_combo)
        if kind in ("heatmap", "correlation_matrix"):
            self._choice(f, "Colour map", COLORMAPS, *put("colormap"), "Colormap", editable=True, key="Colormap")
        if kind == "heatmap":
            self._check(f, "Dendrograms", *put("show_dendrograms"), "Dendrograms")
        if kind == "correlation_matrix":
            self._check(f, "Show r values", *put("show_values"), "Show r values")
        note = self._text(text, "Annotation", *put("annotation"), "Annotation")
        note.setPlaceholderText("auto" if kind == "survival" else "")

    def _fill_color_by(self):
        if self.color_combo is None:
            return
        t = self.app.project.tables.get(self.spec.table_id)
        w = self.color_combo
        w.blockSignals(True)
        w.clear()
        w.addItem("(none)", None)
        for d in (t.datasets if t else []):
            if d.categories is not None:
                w.addItem(d.title, d.id)
        w.setCurrentIndex(max(w.findData(self.spec.color_by), 0))
        w.blockSignals(False)

    def _fill_analyses(self):
        s, an = self.spec, self.app.project.analyses
        import bioapp.analysis as analysis
        kinds = analysis.all_kinds()
        mine = [a for a in an.values() if a.table_id == s.table_id and a.kind in kinds and kinds[a.kind].output == "result"]
        for w, attr in ((self.fit_combo, "fit_analysis_id"), (self.stats_combo, "stats_analysis_id"),
                        (self.result_combo, "analysis_id")):
            if w is None:
                continue
            w.blockSignals(True)
            w.clear()
            w.addItem("(none)", None)
            for a in mine:
                if attr != "analysis_id" or plot.result_ok(s.kind, a.result):
                    w.addItem(a.title, a.id)
            w.setCurrentIndex(max(w.findData(getattr(s, attr)), 0))
            w.blockSignals(False)
        self._fill_color_by()
        self._fill_brackets()
        self._fill_series_list()
        self._fill_trace()

    def _fill_brackets(self):
        if self.brackets is None:
            return
        s = self.spec
        res = self.app.results().get(s.stats_analysis_id) or {}
        have = {b.comparison_id for b in s.brackets}
        t = self.app.project.tables.get(s.table_id)

        def name(ds):
            try:
                return t.dataset(ds).title
            except Exception:
                return str(ds)

        def pval(c):
            p = c.get("p_adj", c.get("p"))
            return "" if not isinstance(p, (int, float)) or p != p else f", P {'< 0.0001' if p < 1e-4 else f'= {p:.4f}'}"
        self.brackets.blockSignals(True)
        self.brackets.clear()
        for c in res.get("comparisons") or []:
            row = f" (row {c['row'] + 1})" if c.get("row") is not None else ""
            it = QListWidgetItem(f"{name(c['a'])} vs {name(c['b'])}{row}: {c.get('summary', '')}{pval(c)}")
            it.setData(Qt.UserRole, c["id"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if c["id"] in have else Qt.Unchecked)
            self.brackets.addItem(it)
        self.brackets.blockSignals(False)

    def _bracket_toggled(self, item):
        cid, on = item.data(Qt.UserRole), item.checkState() == Qt.Checked

        def mutate(s):
            s.brackets[:] = [b for b in s.brackets if b.comparison_id != cid]
            if on:  # ticked by hand: shown even when not significant
                s.brackets.append(plot.BracketSpec(cid, show_ns=True))
        self._change(mutate, "Bracket")

    # ------------------------------------------------------------ selection and drawing
    def select(self, sel):
        self.sel = sel
        page = 0 if not sel else 1 if sel[0] == "bracket" else 3 if sel[0] == "text" else 2
        label = _SELECT_NAMES.get(sel[0], "Data series") if sel else "Nothing selected"
        where = {"axis": "axes", "axtitle": "axes", "title": "text", "legend": "text", "annotation": "text",
                 "legend_entry": "series", "series": "series", "band": "series", "frame": "size"}
        if sel and sel[0] in SERIES_SEL:
            label += f": {self._ser(self.spec).label}"
        if sel and sel[0] in where and not (where[sel[0]] == "series" and self.series_body is None):
            sec = self.sections[where[sel[0]]]
            self.hint.setText(f"Edit it in the \"{sec.title().replace('&&', '&')}\" section below." if sel[0] != "frame" else
                              "Drag the corner handle at the lower right of the plot area to resize the graph.")
            sec.setChecked(True)
            self.area.ensureWidgetVisible(sec)
        elif page == 3:
            self.area.ensureWidgetVisible(self.sel_box)
        if sel and sel[0] in ("axis", "axtitle") and sel[1] in ("x", "y"):  # the clicked axis's widgets answer to the short names
            name = f"{sel[1].upper()} axis"
            for short, long in (("Minimum", "minimum"), ("Maximum", "maximum"), ("Major tick", "tick interval"),
                                ("Log scale", "log scale"), ("Axis title", "title")):
                if f"{name} {long}" in self.widgets:
                    self.widgets[short] = self.widgets[f"{name} {long}"]
        self.stack.setCurrentIndex(page)
        self._fit_stack(page)
        self.sel_label.setText(label)
        self._load()
        self._sync_series_controls()
        if self.editor is not None:
            self.editor.sync()
        self.selection_changed.emit(sel)

    def focus_panel(self) -> None:
        """"Format...": jump to the full section of the selected object."""
        self.select(self.sel)

    def _sync_series_controls(self) -> None:
        """The series picker follows the selection; the style controls need a selected series."""
        on = bool(self.sel and self.sel[0] in SERIES_SEL)
        if self.series_body is not None:
            self.series_body.setEnabled(on)
            self.series_combo.setCurrentIndex(max(self.series_combo.findData(self.sel[1] if on else None), 0))
        self.apply_all.setEnabled(on)

    def _fill_series_list(self) -> None:
        w = self.series_combo
        if w is None:
            return
        w.blockSignals(True)
        w.clear()
        w.addItem("(choose a series)", None)
        for s in self.spec.series:
            w.addItem(s.label or s.dataset_id, s.dataset_id)
        w.blockSignals(False)
        self._sync_series_controls()

    def _load(self):
        self._busy = True
        for fn in self._loaders:
            fn(self.spec)
        self._busy = False

    def _maybe_refresh(self, item_id=None):
        if self.gid not in self.app.project.graphs:
            return
        s = self.spec
        if item_id in (self.gid, s.table_id, s.fit_analysis_id, s.stats_analysis_id, s.analysis_id) \
                or item_id in s.trace.sources.values() or item_id in {
                a.id for a in self.app.project.analyses.values() if a.table_id == s.table_id}:
            self.refresh()

    def _fit_zoom(self, w_in: float, h_in: float) -> int:
        """Zoom percent that shows a w x h inch figure in the preview area (200 % while the view is not on screen)."""
        vp = self.scroll.viewport()
        if not vp.isVisible() or vp.width() < 100 or vp.height() < 100:
            return 200
        z = 0.94 * min(vp.width() / (w_in * BASE_DPI), vp.height() / (h_in * BASE_DPI)) * 100
        return int(max(50, min(200, z)))  # above 200 % the preview only gets slower to redraw, not clearer

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom == "fit" and self.canvas is not None:
            self._fit_timer.start()

    def _maybe_refit(self):
        """After a resize: rebuild only if the fitted zoom really changed (>= 3 %), so open canvases stay put."""
        spec = self.app.project.graphs.get(self.gid)
        if spec is None or self.zoom != "fit" or self.canvas is None:
            return
        if abs(self._fit_zoom(spec.width_mm / 25.4, spec.height_mm / 25.4) - self._shown_zoom) >= 3:
            self.refresh()

    def refresh(self):
        if self.gid not in self.app.project.graphs:
            return
        self._load()
        self._fill_analyses()
        try:
            fig = self.app.graph_figure(self.gid)
        except Exception as exc:
            self.error.setText(f"Cannot draw this graph: {exc}")
            self.error.show()
            return
        self.error.hide()
        old = self.scroll.takeWidget()  # an edit commits from inside an event of the old canvas: delete it later
        self._shown_zoom = self._fit_zoom(*fig.get_size_inches()) if self.zoom == "fit" else self.zoom
        self.canvas = make_canvas(fig, self._shown_zoom)
        self.editor = CanvasEditor(self, self.canvas)
        self.scroll.setWidget(self.canvas)
        if old is not None:
            old.deleteLater()
        if self.quick is not None and not self.quick.isVisible():
            self.quick = None
        if self.sel and self.editor.sel_bbox(self.sel) is None:  # the selected object is gone (hidden, deleted, undone)
            self.select(None)

    @staticmethod
    def resolve(gids: list[str]):
        """Map the gids under a click to one selection, most specific artist first."""
        special = {"xaxis": ("axis", "x"), "yaxis": ("axis", "y"), "xtitle": ("axtitle", "x"), "ytitle": ("axtitle", "y")}
        for key in PRIORITY:
            for g in gids:
                kind, _, ident = g.partition(":")
                if kind == key:
                    return special.get(kind) or (("series", ident) if kind in ("series", "fit") else
                                                 (kind, int(ident)) if kind == "text" else (kind, ident))
        return None

    # ------------------------------------------------------------ edits made on the canvas
    def open_quick(self, sel, ev=None) -> None:
        """Show the quick-format popup of ``sel`` at the pointer (right click) or below the floating toolbar."""
        if not sel or not quick_fields(self, sel):
            return
        self.quick = q = QuickFormat(self, sel)
        if ev is None:
            pos = self.editor.tool.mapToGlobal(QPoint(0, self.editor.tool.height() + 2))
        else:
            d = self.editor.dpr
            pos = self.canvas.mapToGlobal(QPoint(round(ev.x / d), round((self.canvas.figure.bbox.height - ev.y) / d)))
        room = (self.screen().availableGeometry() if self.screen() else None)
        if room is not None:
            pos = QPoint(min(pos.x(), room.right() - q.width()), min(pos.y(), room.bottom() - q.sizeHint().height()))
        q.move(pos)
        q.show()

    def set_text(self, sel, txt: str) -> None:
        """Commit an inline edit: the text of ``sel`` becomes ``txt`` (one undo step)."""
        k, i = sel

        def mutate(s):
            if k == "title":
                s.title = txt
            elif k == "axtitle":
                (s.x_axis if i == "x" else s.y_axis).title = txt
            elif k == "legend_entry":
                next(x for x in s.series if x.dataset_id == i).label = txt
            elif k == "bracket":
                next(b for b in s.brackets if b.comparison_id == i).label = txt or None  # blank = the P value stars
            elif k == "text":
                s.texts[i]["text"] = txt
            elif k == "annotation":
                s.annotation = txt
        self._change(mutate, f"Edit {_SELECT_NAMES[k].lower()}")

    def commit_legend(self, px: float, py: float, fig_box) -> None:
        """Store a dragged legend's upper-left corner (display pixels) as axes fractions of the rebuilt graph, so it
        stays where it was dropped even when the right margin made for an outside legend appears or goes away."""
        fx, fy = px / fig_box.width, py / fig_box.height
        t, res = self.app.project.tables[self.spec.table_id], self.app.results()

        def axes_frac(s):
            W, H = s.width_mm, s.height_mm
            x0, y0, w, h = plot._axes_rect((0, 0, W, H), plot._margins(plot._resolve(s, res), t, res), (W, H))
            return (fx - x0) / w, (fy - y0) / h

        def mutate(s):
            s.legend_loc, s.legend_x, s.legend_y = "manual", 0.0, 0.0
            s.legend_x, s.legend_y = axes_frac(s)
            if s.legend_x >= 1.0:  # right of the axes: that reserves the margin, which narrows them
                s.legend_x, s.legend_y = axes_frac(s)
            s.legend_x, s.legend_y = round(s.legend_x, 4), round(s.legend_y, 4)
        self._change(mutate, "Move legend")

    def delete_selected(self) -> None:
        """Delete key: remove the selected bracket, free text or annotation."""
        sel = self.sel
        if sel and sel[0] in ("bracket", "text", "annotation"):
            hide = quick_fields(self, sel)["hide"][1]
            self._change(lambda s: hide(s, None), f"Delete {_SELECT_NAMES[sel[0]].lower()}")
            self.select(None)


# ---------------------------------------------------------------------------- layout view
class LayoutView(QWidget):
    GAP, MARGIN = 5.0, 5.0

    def __init__(self, app, lid: str, parent=None):
        super().__init__(parent)
        self.app, self.lid, self._busy = app, lid, False
        self.rows = QSpinBox(minimum=1, maximum=8, value=1)
        self.cols = QSpinBox(minimum=1, maximum=8, value=1)
        self.w = QDoubleSpinBox(minimum=40, maximum=600, decimals=1, singleStep=5)
        self.h = QDoubleSpinBox(minimum=40, maximum=800, decimals=1, singleStep=5)
        self.title = QLineEdit()
        self.cells = QWidget()
        self.cell_form = QFormLayout(self.cells)
        self.combos: list[QComboBox] = []
        self._cells_sig = None
        self.error = QLabel()
        self.error.setStyleSheet("background: #fde7e9; color: #7a0010; padding: 6px")
        self.error.setWordWrap(True)
        self.error.hide()
        self.scroll = QScrollArea()
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet(_surround_css())
        export = QPushButton("Export...")
        export.clicked.connect(lambda: export_item(self, app, "layout", lid))
        form = QFormLayout()
        form.addRow("Name", self.title)
        form.addRow("Page width (mm)", self.w)
        form.addRow("Page height (mm)", self.h)
        form.addRow("Rows", self.rows)
        form.addRow("Columns", self.cols)
        form.addRow(self.cells)
        form.addRow(export)
        side = QWidget()
        side.setLayout(form)
        side.setMaximumWidth(340)
        split = QSplitter()
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.error)
        rl.addWidget(self.scroll, 1)
        split.addWidget(side)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self.title.editingFinished.connect(lambda: self._apply(lambda s: setattr(s, "title", self.title.text()), "Layout name"))
        for sp in (self.w, self.h):
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda _: self._apply(self._set_panels, "Layout page size"))
        for sp in (self.rows, self.cols):
            sp.valueChanged.connect(lambda _: self._grid_changed())
        for sig in (app.graph_changed, app.layout_changed, app.analysis_changed, app.table_changed, app.project_changed):
            sig.connect(lambda *_: self.refresh())
        self.refresh()

    @property
    def spec(self):
        return self.app.project.layouts[self.lid]

    def _panel_for_cell(self, i: int, graph_id: str):
        r, c = divmod(i, self.cols.value())
        W, H, m, g = self.w.value(), self.h.value(), self.MARGIN, self.GAP
        cw = (W - 2 * m - (self.cols.value() - 1) * g) / self.cols.value()
        ch = (H - 2 * m - (self.rows.value() - 1) * g) / self.rows.value()
        return plot.PanelSpec(graph_id, m + c * (cw + g), m + r * (ch + g), cw, ch)

    def _set_panels(self, s):
        s.width_mm, s.height_mm = self.w.value(), self.h.value()
        letters = iter(string.ascii_uppercase)
        s.panels = []
        for i, cb in enumerate(self.combos):
            if cb.currentData():
                p = self._panel_for_cell(i, cb.currentData())
                p.label = next(letters)
                s.panels.append(p)

    def _grid_changed(self):
        if self._busy:
            return
        old_cols = self._cells_sig[1] if self._cells_sig else 1
        keep = {(i // old_cols, i % old_cols): cb.currentData() for i, cb in enumerate(self.combos)}
        nc = self.cols.value()
        self._build_cells({r * nc + c: g for (r, c), g in keep.items() if r < self.rows.value() and c < nc})
        self._apply(self._set_panels, "Layout grid")

    def _apply(self, fn, text):
        if not self._busy:
            self.app.edit_layout(self.lid, fn, text)

    def _build_cells(self, assigned: dict[int, str]):
        """One graph chooser per grid cell; widgets are reused unless the grid or the graph list changed
        (deleting a combo inside its own signal handler is unsafe)."""
        graphs = [(gid, self.app.graph_title(gid)) for gid in self.app.project.graphs]
        sig = (self.rows.value(), self.cols.value(), graphs)
        if sig != self._cells_sig:
            self._cells_sig = sig
            while self.cell_form.rowCount():
                self.cell_form.removeRow(0)
            self.combos = []
            for i in range(self.rows.value() * self.cols.value()):
                cb = QComboBox()
                cb.addItem("(empty)", None)
                for gid, title in graphs:
                    cb.addItem(title, gid)
                cb.activated.connect(lambda _: self._apply(self._set_panels, "Assign graph"))
                r, c = divmod(i, self.cols.value())
                self.cell_form.addRow(f"Row {r + 1}, column {c + 1}", cb)
                self.combos.append(cb)
        for i, cb in enumerate(self.combos):
            cb.setCurrentIndex(max(cb.findData(assigned.get(i)), 0))

    def refresh(self):
        if self.lid not in self.app.project.layouts:
            return
        s = self.spec
        self._busy = True
        self.title.setText(s.title)
        self.w.setValue(s.width_mm)
        self.h.setValue(s.height_mm)
        assigned = {}
        if s.panels:  # recover the grid from the panel size: w = (W - 2m - (cols - 1) g) / cols
            p0, m, g = s.panels[0], self.MARGIN, self.GAP
            nc = max(1, round((s.width_mm - 2 * m + g) / (p0.w_mm + g)))
            nr = max(1, round((s.height_mm - 2 * m + g) / (p0.h_mm + g)))
            self.rows.setValue(nr)
            self.cols.setValue(nc)
            for p in s.panels:
                assigned[round((p.y_mm - m) / (p.h_mm + g)) * nc + round((p.x_mm - m) / (p.w_mm + g))] = p.graph_id
        self._build_cells(assigned)
        self._busy = False
        try:
            fig = self.app.layout_figure(self.lid)
        except Exception as exc:
            self.error.setText(f"Cannot draw this layout: {exc}")
            self.error.show()
            return
        self.error.hide()
        self.scroll.setWidget(make_canvas(fig, 100))



