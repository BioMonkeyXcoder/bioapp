"""Colour tokens, line icons and the application style: "glass" (default, rounded liquid-glass look) or "flat" (the
earlier macaron look).

``apply(app, style)`` is called at start-up and again when the user switches the style. Every colour used by the UI
comes from here, so a text colour is only ever placed on a background whose pair is checked in
``tests/test_ui_theme.py`` (WCAG contrast; the glass fills are checked as composites over the backdrop).

Glass = a plain paper-white backdrop (white in dialogs) painted once per window size (``backdrop``), translucent rounded cards
(``GlassCard``, chrome cached per size, shadow painted into it: no QGraphicsEffect, no live blur), a pill toolbar,
rounded inputs from ``GlassStyle`` (a Fusion proxy, so native combo / spin arrows stay) and the stylesheet below.
"""
from __future__ import annotations

from functools import lru_cache
from string import Template
from typing import NamedTuple

from PySide6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QImage, QPolygonF)
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog, QFrame,
                               QGroupBox, QLineEdit, QListView, QMenu, QProxyStyle, QRadioButton, QStyle, QStyledItemDelegate, QToolBar,
                               QTreeView)


class Tone(NamedTuple):
    base: str   # pastel fill (rows, highlights, pill backgrounds)
    deep: str   # selection / hover / focus
    ink: str    # lines and glyphs on the page background (>= 3:1)


ROSE = Tone("#EBC4C0", "#DDA6A0", "#B8665F")
PEACH = Tone("#F2D3B8", "#E6B88F", "#AF7240")
BUTTER = Tone("#F1E4B3", "#E3CD80", "#9A8228")
MINT = Tone("#C4E2D2", "#9BCBB1", "#4F8F72")
SKY = Tone("#C6DBEE", "#9CBFE0", "#4F82B1")
LAVENDER = Tone("#D6CDEB", "#B9ABDD", "#7D6CB0")

BG, PANEL, SURFACE = "#FAF8F5", "#F3EFEA", "#FFFFFF"
BORDER, BORDER_STRONG = "#E6E0D8", "#D3CBC0"
TEXT, TEXT2, DISABLED = "#3F3A36", "#615950", "#B3ABA2"
DANGER = "#9B3F39"      # text of an error line on BG
NOTE = "#1F4A73"       # excluded values in the data table (text on SURFACE and on a selected cell)

# navigator section -> tone
SECTION = {"tables": SKY, "analyses": LAVENDER, "graphs": MINT, "layouts": PEACH}


def mix(a: str, b: str, t: float) -> str:
    """Colour ``t`` of the way from ``a`` to ``b`` (hex strings)."""
    ca, cb = QColor(a), QColor(b)
    return QColor(*(round(x + (y - x) * t) for x, y in zip(ca.getRgb()[:3], cb.getRgb()[:3]))).name().upper()


def pale(tone: Tone, t: float = 0.5) -> str:
    """The pastel fill diluted towards white: header and stripe fills."""
    return mix(tone.base, SURFACE, t)


def luminance(c: str) -> float:
    q = QColor(c)
    lin = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(q.redF()) + 0.7152 * lin(q.greenF()) + 0.0722 * lin(q.blueF())


def contrast(a: str, b: str) -> float:
    """WCAG 2.x contrast ratio of two colours."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def rgba(c: str, a: float) -> str:
    """QSS colour of hex ``c`` at opacity ``a`` (percent form: valid in every Qt style sheet)."""
    q = QColor(c)
    return f"rgba({q.red()}, {q.green()}, {q.blue()}, {round(a * 100)}%)"


def over(fg: str, alpha: float, bg: str) -> str:
    """The colour ``fg`` at opacity ``alpha`` composited over opaque ``bg`` (hex): what the eye sees through glass."""
    return mix(bg, fg, alpha)


# ---------------------------------------------------------------- style state

STYLES = ("glass", "flat")
_state = {"style": "flat"}  # "flat" until apply() runs: an unthemed QApplication (tests) paints no glass


class _Notifier(QObject):
    changed = Signal(str)   # emitted by apply(): cards, the toolbar and the main window re-margin and repaint


notifier = _Notifier()


def style() -> str:
    """The style applied last: "glass" or "flat"."""
    return _state["style"]


def saved_style() -> str:
    """The user's choice from the settings (dialogs._settings): "glass" unless "flat" was saved."""
    from .dialogs import _settings
    v = _settings().value("ui/style", "glass", type=str)
    return v if v in STYLES else "glass"


def save_style(s: str) -> None:
    from .dialogs import _settings
    _settings().setValue("ui/style", s)


# ---------------------------------------------------------------- glass tokens

SHADOW = 5                  # px of soft shadow painted around a card (inside the card widget's margin)
CARD_R = 16                 # card corner radius
CARD_TOP, CARD_BOT = 0.74, 0.60   # white fill opacity at the top / bottom of a card
GLASS_HOVER, GLASS_FILL = 0.88, 0.72   # buttons and fields: hover / rest opacity of the white glass
SEL_ALPHA = 0.85            # selection capsule: SKY.deep at this opacity ...
SEL_EDGE = 1.5              # ... with a border of this width (px) in the tone's ink colour
STRIPE_ALPHA = 0.30         # alternate rows / sheet stripes: white at this opacity
TINT_ALPHA = 0.30           # inner sections and group boxes: INNER at this opacity
INNER = "#E4DFD6"           # neutral warm grey of inner sections (no colour cast)
BACKDROP_BASE = "#F5F3EE"   # paper white behind the main window's cards
DIALOG_BASE = "#FFFFFF"     # dialogs: plain white


def card_over(bg: str, alpha: float = CARD_BOT) -> str:
    """A glass card (white at ``alpha``, the thinnest part by default) seen over backdrop colour ``bg``."""
    return over("#FFFFFF", alpha, bg)


def worst_backdrops() -> list[str]:
    """Backdrop colours text on a card can sit on."""
    return [BACKDROP_BASE, DIALOG_BASE]


def backdrop_image(w: int, h: int, light: bool = False) -> QImage:
    """The plain backdrop: paper white (window) or white (``light``: dialogs)."""
    img = QImage(max(1, w), max(1, h), QImage.Format_RGB32)
    img.fill(QColor(DIALOG_BASE if light else BACKDROP_BASE))
    return img


@lru_cache(maxsize=6)
def _backdrop(pw: int, ph: int, dpr: float, light: bool) -> QPixmap:
    pm = QPixmap.fromImage(backdrop_image(pw, ph, light))
    pm.setDevicePixelRatio(dpr)
    return pm


def backdrop(w: int, h: int, dpr: float = 1.0, light: bool = False) -> QPixmap:
    """The cached backdrop pixmap of a w x h (logical px) window; built again only for a new size
    (``backdrop_builds()`` counts the builds)."""
    return _backdrop(max(1, round(w * dpr)), max(1, round(h * dpr)), dpr, light)


def backdrop_builds() -> int:
    return _backdrop.cache_info().misses


def _chrome(w: int, h: int, dpr: float) -> QPixmap:
    """Card chrome for a w x h widget: soft shadow in the outer SHADOW px, translucent white fill (brighter at the top),
    1 px faint dark outer border and a 1 px white highlight that is strongest along the top edge."""
    pm = QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    box = QRectF(SHADOW, SHADOW, w - 2 * SHADOW, h - 2 * SHADOW)
    for k in range(SHADOW, 0, -1):  # soft falloff outside the card, nudged down by 1.5 px
        path = QPainterPath()
        path.addRoundedRect(box.adjusted(-k, -k + 1.5, k, k + 1.5), CARD_R + k, CARD_R + k)
        p.setPen(QPen(QColor(70, 50, 40, round(26 * (1 - k / (SHADOW + 1)) ** 2)), 1))
        p.drawPath(path)
    shape = QPainterPath()
    shape.addRoundedRect(box, CARD_R, CARD_R)
    p.setCompositionMode(QPainter.CompositionMode_Clear)  # the shadow must not tint the glass
    p.fillPath(shape, Qt.black)
    p.setCompositionMode(QPainter.CompositionMode_SourceOver)
    fill = QLinearGradient(box.topLeft(), box.bottomLeft())
    fill.setColorAt(0, QColor(255, 255, 255, round(CARD_TOP * 255)))
    fill.setColorAt(1, QColor(255, 255, 255, round(CARD_BOT * 255)))
    p.fillPath(shape, fill)
    p.setPen(QPen(QColor(70, 50, 40, 38), 1))
    p.setBrush(Qt.NoBrush)
    edge = QPainterPath()
    edge.addRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), CARD_R - 0.5, CARD_R - 0.5)
    p.drawPath(edge)
    hl = QLinearGradient(box.topLeft(), box.bottomLeft())
    hl.setColorAt(0, QColor(255, 255, 255, 235))
    hl.setColorAt(0.25, QColor(255, 255, 255, 120))
    hl.setColorAt(0.7, QColor(255, 255, 255, 30))
    hl.setColorAt(1, QColor(255, 255, 255, 90))
    p.setPen(QPen(hl, 1))
    inner = QPainterPath()
    inner.addRoundedRect(box.adjusted(1.5, 1.5, -1.5, -1.5), CARD_R - 1.5, CARD_R - 1.5)
    p.drawPath(inner)
    p.end()
    return pm


class GlassCard(QFrame):
    """A translucent rounded panel. Paints its cached chrome in the glass style and nothing in the flat style; its
    layout margin is the shadow band plus ``pad`` (glass) or ``flat_pad`` (flat). Content widgets sit on it as they are:
    scrolling views inside are transparent (QSS), so no widget needs a graphics effect."""

    chrome_renders = 0   # class counter: chrome is rebuilt only when a card changes size

    def __init__(self, pad=14, flat_pad=0, parent=None):
        super().__init__(parent)
        self.setProperty("glass", "card")
        self._pad, self._flat_pad, self._key, self._pm = pad, flat_pad, None, None
        notifier.changed.connect(self.restyle)

    def setLayout(self, lay):
        super().setLayout(lay)
        self.restyle()

    def restyle(self, *_):
        """(Re)apply the margins of the current style to the layout; call after creating the layout."""
        lay = self.layout()
        if lay is not None:
            if style() == "glass":
                p = self._pad if isinstance(self._pad, tuple) else (self._pad,) * 4
                lay.setContentsMargins(*(SHADOW + v for v in p))
            else:
                f = self._flat_pad
                lay.setContentsMargins(*(f if isinstance(f, tuple) else (f,) * 4))
        self.update()

    def set_inner_width(self, w: int) -> None:
        """Fix the width of the content area; the margins of the current style are added."""
        m = self.layout().contentsMargins()
        self.setFixedWidth(w + m.left() + m.right())

    def paintEvent(self, e):
        if style() != "glass":
            return super().paintEvent(e)
        key = (self.width(), self.height(), self.devicePixelRatioF())
        if key != self._key:
            self._pm, self._key = _chrome(*key), key
            GlassCard.chrome_renders += 1
        p = QPainter(self)
        p.drawPixmap(0, 0, self._pm)
        p.end()


class PillToolBar(QToolBar):
    """Toolbar drawn as one more glass card (same chrome, radius and shadow as the panels, same 12 px window inset);
    the bar itself is transparent."""

    _key = _pm = None

    def paintEvent(self, e):
        if style() == "glass":
            x = 12 - SHADOW  # the panels' inset (MainWindow._restyle)
            key = (self.width() - 2 * x, self.height(), self.devicePixelRatioF())
            if key != self._key:
                self._pm, self._key = _chrome(*key), key
            p = QPainter(self)
            p.drawPixmap(x, 0, self._pm)
            p.end()
        super().paintEvent(e)


# kind -> (background, left bar); text is always TEXT
_BANNERS = {"failed": (pale(ROSE, 0.45), ROSE.deep), "warning": (pale(BUTTER, 0.45), BUTTER.deep),
            "stale": (pale(SKY, 0.45), SKY.deep), "info": (PANEL, BORDER_STRONG)}
# glass: kind -> tone of the tinted capsule (info = white glass)
_BANNER_TONE = {"failed": ROSE, "warning": BUTTER, "stale": SKY, "info": None}
BANNER_ALPHA = 0.62


def banner_css(kind: str) -> str:
    """Style of a status line: flat = muted fill with a thin coloured bar on the left; glass = tinted rounded capsule."""
    if style() == "glass":
        tone = _BANNER_TONE[kind]
        fill = rgba(tone.base if tone else "#FFFFFF", BANNER_ALPHA)
        return (f"background: {fill}; color: {TEXT}; border: 1px solid {rgba('#FFFFFF', 0.7)}; border-radius: 13px; "
                f"padding: 7px 14px")
    bg, bar = _BANNERS[kind]
    return (f"background: {bg}; color: {TEXT}; border-left: 3px solid {bar}; border-radius: 0; "
            f"border-top-right-radius: 6px; border-bottom-right-radius: 6px; padding: 6px 8px")


# ---------------------------------------------------------------- line icons (painted, no image files)

def _tray(p):
    p.drawPolyline([QPointF(4, 12), QPointF(4, 16), QPointF(16, 16), QPointF(16, 12)])


def _g_table(p):
    p.drawRoundedRect(QRectF(3, 5, 12, 11), 2, 2)
    p.drawLine(QPointF(3, 9), QPointF(15, 9))
    p.drawLine(QPointF(9, 9), QPointF(9, 16))
    p.drawLine(QPointF(16.5, 2.5), QPointF(16.5, 7.5))
    p.drawLine(QPointF(14, 5), QPointF(19, 5))


def _g_import(p):
    _tray(p)
    p.drawLine(QPointF(10, 3), QPointF(10, 12))
    p.drawPolyline([QPointF(6.5, 8.5), QPointF(10, 12), QPointF(13.5, 8.5)])


def _g_export(p):
    _tray(p)
    p.drawLine(QPointF(10, 12), QPointF(10, 3))
    p.drawPolyline([QPointF(6.5, 6.5), QPointF(10, 3), QPointF(13.5, 6.5)])


def _g_analyze(p):  # sigma
    p.drawPolyline([QPointF(14.5, 4), QPointF(6, 4), QPointF(11, 10), QPointF(6, 16), QPointF(14.5, 16)])


def _g_graph(p):
    p.drawPolyline([QPointF(3, 3), QPointF(3, 17), QPointF(17, 17)])
    p.drawPolyline([QPointF(6.5, 13), QPointF(9.5, 9), QPointF(12.5, 11), QPointF(16, 5.5)])


def _g_undo(p):
    p.drawPolyline([QPointF(7.5, 4.5), QPointF(4, 8), QPointF(7.5, 11.5)])
    path = QPainterPath(QPointF(4, 8))
    path.lineTo(11, 8)
    path.cubicTo(17, 8, 17, 15.5, 11, 15.5)
    path.lineTo(7, 15.5)
    p.drawPath(path)


def _g_redo(p):
    p.translate(20, 0)
    p.scale(-1, 1)
    _g_undo(p)


_GLYPHS = {"table": _g_table, "import": _g_import, "export": _g_export, "analyze": _g_analyze, "graph": _g_graph,
           "undo": _g_undo, "redo": _g_redo}


def icon(name: str, size: int = 20) -> QIcon:
    """Line icon in the text colour (normal) or the disabled colour; drawn at 2x for sharp edges."""
    ic = QIcon()
    for mode, colour in ((QIcon.Normal, TEXT), (QIcon.Disabled, DISABLED)):
        pm = QPixmap(size * 2, size * 2)
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(size / 20, size / 20)
        p.setPen(QPen(QColor(colour), 1.9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))  # matches the semibold text
        _GLYPHS[name](p)
        p.end()
        ic.addPixmap(pm, mode)
    return ic


# ---------------------------------------------------------------- application style

_QSS_FLAT = Template("""
QMainWindow, QDialog { background: $bg; }
QToolTip { background: $panel; color: $text; border: 1px solid $border; padding: 4px 6px; }
QLabel { background: transparent; }

QMenuBar { background: $bg; border-bottom: 1px solid $border; padding: 2px 6px; }
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 6px; }
QMenuBar::item:selected { background: $sky; }
QMenu { background: $surface; border: 1px solid $border; padding: 4px; }
QMenu::item { padding: 5px 24px 5px 14px; border-radius: 4px; }
QMenu::item:selected { background: $sky; color: $text; }
QMenu::item:disabled { color: $disabled; background: transparent; }
QMenu::separator { height: 1px; background: $border; margin: 4px 8px; }

QToolBar { background: $bg; border: none; border-bottom: 1px solid $border; spacing: 2px; padding: 4px 8px; }
QToolBar::separator { background: $border; width: 1px; margin: 5px 8px; }
QToolBar QToolButton { background: transparent; border: none; border-radius: 6px; padding: 4px 10px; }
QToolBar QToolButton:hover { background: $sky; }
QToolBar QToolButton:pressed { background: $sky_deep; }
QToolBar QToolButton:disabled { color: $disabled; background: transparent; }
QToolButton { background: transparent; border: none; border-radius: 4px; padding: 2px; }
QToolButton:hover { background: $sky; }

QPushButton { background: $surface; border: 1px solid $border_strong; border-radius: 6px; padding: 4px 14px;
              min-height: 18px; }
QPushButton:hover { background: $sky; border-color: $sky_deep; }
QPushButton:pressed { background: $sky_deep; }
QPushButton:default { background: $sky; border-color: $sky_deep; }
QPushButton:focus { border-color: $sky_deep; }
QPushButton:disabled { background: $panel; color: $disabled; border-color: $border; }
QStatusBar QPushButton { background: transparent; border: none; padding: 2px 10px; min-height: 16px; }
QStatusBar QPushButton:hover { background: $sky; }
QStatusBar QPushButton:disabled { background: transparent; }

QComboBox QAbstractItemView { background: $surface; border: 1px solid $border_strong; border-radius: 0;
    selection-background-color: $sky; selection-color: $text; outline: 0; padding: 2px; }
QCheckBox, QRadioButton { spacing: 6px; background: transparent; }

QAbstractItemView { background: $surface; alternate-background-color: $alt; border: 1px solid $border;
    border-radius: 6px; selection-background-color: $sky; selection-color: $text; gridline-color: $border; outline: 0; }
QTreeView { show-decoration-selected: 0; }
QListView { show-decoration-selected: 0; }
QListView::item:hover { background: $panel; }
QTreeView::item:selected, QListView::item:selected { background: $sky; color: $text; }
QTableView { border: none; border-radius: 0; }
QTableView::item:selected { background: $sky; color: $text; }
QHeaderView { background: transparent; }
QHeaderView::section { background: $head; color: $text; border: none; border-right: 1px solid $border;
    border-bottom: 1px solid $border; padding: 4px 8px; font-weight: 700; }
QTableCornerButton::section { background: $head; border: none; border-right: 1px solid $border; border-bottom: 1px solid $border; }

QTabWidget::pane { border: none; border-top: 1px solid $border; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: $text2; padding: 6px 14px; border: none;
    border-bottom: 2px solid transparent; margin-right: 2px; }
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $lav_deep; }

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle:vertical { background: $border_strong; border-radius: 4px; min-height: 28px; margin: 2px; }
QScrollBar::handle:horizontal { background: $border_strong; border-radius: 4px; min-width: 28px; margin: 2px; }
QScrollBar::handle:hover { background: $disabled; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QSplitter::handle:horizontal { width: 7px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 $bg, stop:0.42 $bg, stop:0.42 $border, stop:0.58 $border, stop:0.58 $bg, stop:1 $bg); }
QSplitter::handle:vertical { height: 7px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    stop:0 $bg, stop:0.42 $bg, stop:0.42 $border, stop:0.58 $border, stop:0.58 $bg, stop:1 $bg); }

QGroupBox { background: transparent; border: 1px solid $border; border-radius: 6px; margin-top: 9px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px; }
QStatusBar { background: $bg; color: $text2; border-top: 1px solid $border; }
QStatusBar::item { border: none; }
QTreeWidget#nav { background: $panel; border: none; border-radius: 0; padding: 6px 4px;
    selection-background-color: transparent; }
QTreeWidget#nav::item:selected, QTreeWidget#nav::branch:selected { background: transparent; }
""")




# ---------------------------------------------------------------- glass: rounded inputs on top of Fusion

def _in_view(w) -> bool:
    """True for the editor of an item-view cell (it keeps a plain rectangular frame)."""
    p = w.parentWidget() if w is not None else None
    return p is not None and isinstance(p.parentWidget(), QAbstractScrollArea)


class GlassStyle(QProxyStyle):
    """Fusion with capsule-shaped line edits, combo boxes and spin boxes (soft focus ring) and rounded check boxes /
    radio buttons. The arrows are Fusion's own primitives, so they keep working (a QSS border on these widgets would
    drop them). It is set on those widgets only (``_AppFilter``), never application-wide: a Python style called for every
    cell of a 100k-row table would slow its scrolling."""

    def __init__(self):
        super().__init__("Fusion")

    # -- helpers
    def _capsule(self, opt, p, w) -> QRectF:
        s = opt.state
        en, focus = bool(s & QStyle.State_Enabled), bool(s & (QStyle.State_HasFocus | QStyle.State_Selected))
        hover = bool(s & QStyle.State_MouseOver)
        r = QRectF(opt.rect).adjusted(0.5, 0.5, -0.5, -0.5)
        rad = min(r.height() / 2, 15)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        a = (0.95 if focus else GLASS_HOVER if hover else GLASS_FILL) if en else 0.35
        p.setBrush(QColor(255, 255, 255, round(a * 255)))
        p.setPen(QPen(QColor(80, 62, 52, 40 if en else 22), 1))
        p.drawRoundedRect(r, rad, rad)
        if focus and en:  # soft glow: a translucent ring inside the border, then a crisp border
            p.setBrush(Qt.NoBrush)
            ring = QColor(SKY.deep)
            ring.setAlpha(150)
            p.setPen(QPen(ring, 3))
            p.drawRoundedRect(r.adjusted(1.5, 1.5, -1.5, -1.5), rad - 1.5, rad - 1.5)
            p.setPen(QPen(QColor(SKY.ink), 1))
            p.drawRoundedRect(r, rad, rad)
        p.restore()
        return r

    def _indicator(self, opt, p, radio: bool) -> None:
        s = opt.state
        en, on, mid, hover = (bool(s & QStyle.State_Enabled), bool(s & QStyle.State_On), bool(s & QStyle.State_NoChange),
                              bool(s & QStyle.State_MouseOver))
        r = QRectF(opt.rect).adjusted(1.5, 1.5, -1.5, -1.5)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        if on or mid:
            p.setBrush(QColor(LAVENDER.ink if en else DISABLED))
            p.setPen(QPen(QColor(LAVENDER.ink if en else DISABLED).darker(115), 1))
        else:
            p.setBrush(QColor(255, 255, 255, 235 if en else 110))
            p.setPen(QPen(QColor(SKY.ink) if hover and en else QColor(80, 62, 52, 95 if en else 40), 1))
        if radio:
            p.drawEllipse(r)
        else:
            p.drawRoundedRect(r, 5, 5)
        if on or mid:
            p.setPen(QPen(QColor("#FFFFFF"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            c, u = r.center(), r.width() / 16
            if radio:
                p.setBrush(QColor("#FFFFFF"))
                p.setPen(Qt.NoPen)
                p.drawEllipse(c, 3 * u, 3 * u)
            elif mid:
                p.drawLine(QPointF(c.x() - 3.5 * u, c.y()), QPointF(c.x() + 3.5 * u, c.y()))
            else:
                p.drawPolyline(QPolygonF([QPointF(c.x() - 4 * u, c.y() + 0.2 * u), QPointF(c.x() - 1.2 * u, c.y() + 3 * u),
                                          QPointF(c.x() + 4 * u, c.y() - 3 * u)]))
        p.restore()

    # -- QStyle overrides
    def drawPrimitive(self, pe, opt, p, w=None):
        if pe == QStyle.PE_PanelLineEdit:
            if getattr(opt, "lineWidth", 1) > 0 and not _in_view(w):
                self._capsule(opt, p, w)
                return
            if w is not None and isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox)):
                return  # the frameless editor of a spin / combo box: the capsule behind it is the field
        elif pe in (QStyle.PE_IndicatorCheckBox, QStyle.PE_IndicatorItemViewItemCheck):
            self._indicator(opt, p, False)
            return
        elif pe == QStyle.PE_IndicatorRadioButton:
            self._indicator(opt, p, True)
            return
        super().drawPrimitive(pe, opt, p, w)

    def drawComplexControl(self, cc, opt, p, w=None):
        if cc in (QStyle.CC_ComboBox, QStyle.CC_SpinBox) and not _in_view(w):
            self._capsule(opt, p, w)
            from PySide6.QtWidgets import QStyleOptionComboBox, QStyleOptionSpinBox
            if cc == QStyle.CC_ComboBox and isinstance(opt, QStyleOptionComboBox):
                a = QStyleOptionComboBox(opt)
                a.rect = self.subControlRect(cc, opt, QStyle.SC_ComboBoxArrow, w)
                super().drawPrimitive(QStyle.PE_IndicatorArrowDown, a, p, w)
                return
            if cc == QStyle.CC_SpinBox and isinstance(opt, QStyleOptionSpinBox):
                for sc, pe in ((QStyle.SC_SpinBoxUp, QStyle.PE_IndicatorSpinUp), (QStyle.SC_SpinBoxDown, QStyle.PE_IndicatorSpinDown)):
                    a = QStyleOptionSpinBox(opt)
                    a.rect = self.subControlRect(cc, opt, sc, w)
                    a.state &= ~QStyle.State_MouseOver
                    super().drawPrimitive(pe, a, p, w)
                return
        super().drawComplexControl(cc, opt, p, w)

    def subControlRect(self, cc, opt, sc, w=None):
        r = super().subControlRect(cc, opt, sc, w)
        if cc == QStyle.CC_SpinBox:
            if sc in (QStyle.SC_SpinBoxUp, QStyle.SC_SpinBoxDown):
                return r.translated(-6, 0)
            if sc == QStyle.SC_SpinBoxEditField:
                return r.adjusted(2, 0, -6, 0)
        elif cc == QStyle.CC_ComboBox:
            if sc == QStyle.SC_ComboBoxArrow:
                return r.translated(-4, 0)
            if sc == QStyle.SC_ComboBoxEditField:
                return r.adjusted(8, 0, -2, 0)
        return r

    def subElementRect(self, se, opt, w=None):
        r = super().subElementRect(se, opt, w)
        if se == QStyle.SE_LineEditContents and not _in_view(w):
            return r.adjusted(7, 0, -7, 0)
        return r

    def sizeFromContents(self, ct, opt, size, w=None):
        s = super().sizeFromContents(ct, opt, size, w)
        if ct in (QStyle.CT_LineEdit, QStyle.CT_ComboBox, QStyle.CT_SpinBox) and not _in_view(w):
            return s + QSize(10 if ct != QStyle.CT_LineEdit else 14, 6)
        return s

    def pixelMetric(self, pm, opt=None, w=None):
        if pm in (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight, QStyle.PM_ExclusiveIndicatorWidth,
                  QStyle.PM_ExclusiveIndicatorHeight):
            return 18
        return super().pixelMetric(pm, opt, w)


class CapsuleDelegate(QStyledItemDelegate):
    """Tree / list rows with a translucent capsule as selection (glass); the standard selection in the flat style."""

    def initStyleOption(self, opt, idx):
        super().initStyleOption(opt, idx)
        if _state["style"] == "glass":
            opt.state &= ~QStyle.State_Selected  # text stays in the normal colour on the tinted capsule

    def paint(self, p, opt, idx):
        if _state["style"] == "glass" and opt.state & QStyle.State_Selected:
            p.save()
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(SKY.ink), SEL_EDGE))
            c = QColor(SKY.deep)
            c.setAlphaF(SEL_ALPHA)
            p.setBrush(c)
            r = QRectF(opt.rect).adjusted(2.5, 1.5, -2.5, -1.5)
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            p.restore()
        super().paint(p, opt, idx)


class _DialogPainter(QObject):
    """Paints the light backdrop under the children of one dialog (a Paint filter: every QDialog, also those of other
    modules and Qt's own message boxes, gets it without a base class)."""

    def eventFilter(self, o, e):
        if e.type() == QEvent.Paint and _state["style"] == "glass":
            p = QPainter(o)
            p.drawPixmap(0, 0, backdrop(o.width(), o.height(), o.devicePixelRatioF(), light=True))
            p.end()
        return False


# widgets that get the capsule / rounded-indicator style (small views only: big tables keep the C++ Fusion style)
_GLASS_WIDGETS = (QLineEdit, QComboBox, QAbstractSpinBox, QCheckBox, QRadioButton, QGroupBox, QListView, QTreeView)


def _glass_style_widgets(app, on: bool) -> None:
    """Give (or take back) the glass style of every existing widget of the classes above."""
    for w in app.allWidgets():
        if isinstance(w, _GLASS_WIDGETS) and bool(w.property("glassStyle")) != on:
            w.setProperty("glassStyle", on)
            w.setStyle(_keep["style"] if on else None)


class _AppFilter(QObject):
    """Application-wide: sets the glass style on new inputs, attaches the backdrop painter to a dialog when it first shows,
    and makes every menu a frameless translucent window so its rounded corners are real."""

    def __init__(self, app):
        super().__init__(app)
        self.painter = _DialogPainter(self)

    def eventFilter(self, o, e):
        t = e.type()
        if t == QEvent.Polish and isinstance(o, _GLASS_WIDGETS):
            if _state["style"] == "glass" and not o.property("glassStyle"):
                o.setProperty("glassStyle", True)
                o.setStyle(_keep["style"])
        elif t == QEvent.Show and isinstance(o, QDialog) and not o.property("glassPaint"):
            o.setProperty("glassPaint", True)
            o.installEventFilter(self.painter)
        elif t == QEvent.Polish and isinstance(o, QMenu) and _state["style"] == "glass" and not o.property("glassMenu"):
            o.setProperty("glassMenu", True)
            o.setWindowFlags(o.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
            o.setAttribute(Qt.WA_TranslucentBackground, True)
        return False


# ---------------------------------------------------------------- application style

_QSS_GLASS = Template("""
QMainWindow { background: $bg; }
QToolTip { background: #FFFFFF; color: $text; border: 1px solid $line; border-radius: 8px; padding: 5px 9px; }
QLabel { background: transparent; }

QMenuBar { background: transparent; border: none; padding: 6px 12px 0 12px; spacing: 2px; }
QMenuBar::item { background: transparent; padding: 5px 13px; border-radius: 14px; }
QMenuBar::item:selected { background: $w85; }
QMenu { background: $w96; border: 1px solid $line; border-radius: 14px; padding: 6px; }
QMenu::item { padding: 6px 26px 6px 16px; border-radius: 10px; margin: 1px 2px; }
QMenu::item:selected { background: $sel; color: $text; }
QMenu::item:disabled { color: $disabled; background: transparent; }
QMenu::separator { height: 1px; background: $line2; margin: 5px 12px; }

QToolBar { background: transparent; border: none; spacing: 2px; padding: 11px 22px 11px 22px; }
QToolBar::separator { background: $line; width: 1px; margin: 6px 8px; }
QToolBar QToolButton { background: transparent; border: 1px solid transparent; border-radius: 10px; padding: 4px 12px; }
QToolBar QToolButton:hover { background: $w85; border-color: $w80; }
QToolBar QToolButton:pressed { background: $sel; }
QToolBar QToolButton:disabled { color: $disabled; background: transparent; }
QToolButton { background: transparent; border: none; border-radius: 10px; padding: 3px 7px; }
QToolButton:hover { background: $w75; }
QToolButton:checked { background: $sel; }

QPushButton { background: $w72; border: 1px solid $line; border-radius: 16px; padding: 5px 18px; min-height: 20px; }
QPushButton:hover { background: $w92; border-color: $sky_deep; }
QPushButton:pressed { background: $sel; }
QPushButton:default { background: $sel; border-color: $sky_ink_a; }
QPushButton:focus { border-color: $sky_ink_a; }
QPushButton:disabled { background: $w35; color: $disabled; border-color: $line2; }
QStatusBar QPushButton { background: transparent; border: none; padding: 2px 12px; min-height: 16px; border-radius: 12px; }
QStatusBar QPushButton:hover { background: $w75; }
QStatusBar QPushButton:disabled { background: transparent; }

QComboBox QAbstractItemView { background: #FFFFFF; border: 1px solid $line; border-radius: 10px;
    selection-background-color: $sel; selection-color: $text; outline: 0; padding: 4px; }
QCheckBox, QRadioButton { spacing: 8px; background: transparent; }

QAbstractItemView { background: $w58; alternate-background-color: $w30; border: 1px solid $w75; border-radius: 14px;
    selection-background-color: $sel; selection-color: $text; gridline-color: $line3; outline: 0; padding: 3px; }
QFrame[glass="card"] QAbstractItemView { background: transparent; border: none; border-radius: 0; padding: 0; }
QTreeView { show-decoration-selected: 0; }
QListView { show-decoration-selected: 0; }
QListView::item, QTreeView::item { border-radius: 9px; padding: 2px 3px; }
QListView::item:hover, QTreeView::item:hover { background: $w55; }
QListView::item:selected { background: $sel; color: $text; }
QTreeView::item:selected { background: transparent; color: $text; }
QTreeView::branch:selected, QTreeView::branch:hover { background: transparent; }
QTableView { border: none; border-radius: 0; }
QTableView::item:selected { background: $sel; color: $text; }
QHeaderView { background: transparent; }
QHeaderView::section { background: transparent; color: $text; border: none; border-bottom: 1px solid $line;
    padding: 6px 8px; font-weight: 700; }
QTableCornerButton::section { background: transparent; border: none; }

QTabWidget::pane { border: none; background: transparent; top: 2px; }
QTabWidget::tab-bar { left: 0px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: $text2; padding: 5px 16px; margin: 2px 2px; border-radius: 14px;
    border: 1px solid transparent; }
QTabBar::tab:hover { color: $text; background: $w55; }
QTabBar::tab:selected { color: $text; background: $w92; border: 1px solid $line2; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QAbstractScrollArea::corner { background: transparent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 4px 2px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px 4px; }
QScrollBar::handle:vertical { background: rgba(80, 62, 52, 30%); border-radius: 4px; min-height: 30px; }
QScrollBar::handle:horizontal { background: rgba(80, 62, 52, 30%); border-radius: 4px; min-width: 30px; }
QScrollBar::handle:hover { background: rgba(80, 62, 52, 48%); }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QSplitter::handle { background: transparent; }

QSlider::groove:horizontal { height: 4px; background: $line; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $sky_deep; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; border-radius: 9px; background: #FFFFFF;
    border: 1px solid $line; }

QGroupBox { background: $tint; border: 1px solid $w85; border-radius: 14px; margin-top: 20px; padding: 10px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 6px; }
QFrame[glass="inner"] { background: $tint; border: 1px solid $w85; border-radius: 14px; }
QFrame[glass="inner"] QLabel, QFrame[glass="inner"] QToolButton { background: transparent; }
QStatusBar { background: transparent; color: $text2; padding: 0 12px 4px 12px; }
QStatusBar QLabel { padding-left: 14px; }
QStatusBar::item { border: none; }
QTreeWidget#nav { background: transparent; border: none; border-radius: 0; padding: 2px; show-decoration-selected: 0; }
QTreeWidget#nav::item:selected, QTreeWidget#nav::branch:selected { background: transparent; }
QTreeWidget#nav::item:hover { background: transparent; }
""")

def stylesheet(which: str | None = None) -> str:
    which = which or style()
    if which == "flat":
        return _QSS_FLAT.substitute(
            bg=BG, panel=PANEL, surface=SURFACE, border=BORDER, border_strong=BORDER_STRONG, text=TEXT, text2=TEXT2,
            disabled=DISABLED, sky=SKY.base, sky_deep=SKY.deep, lav_deep=LAVENDER.deep, alt=mix(BG, PANEL, 0.6),
            head=pale(SKY, 0.45))
    w = {f"w{n}": rgba("#FFFFFF", n / 100) for n in (30, 35, 50, 55, 58, 72, 75, 80, 85, 92, 96)}
    return _QSS_GLASS.substitute(
        bg=BG, text=TEXT, text2=TEXT2, disabled=DISABLED, line=rgba("#503E34", 0.16), line2=rgba("#503E34", 0.09), line3=rgba("#503E34", 0.06),
        sel=rgba(SKY.deep, SEL_ALPHA), tint=rgba(INNER, TINT_ALPHA), sky_deep=SKY.deep, sky_ink_a=rgba(SKY.ink, 0.8), **w)


def info_css() -> str:
    """The equation / parameter info box of the model picker."""
    if style() == "glass":
        return f"QLabel {{ background: {rgba('#FFFFFF', 0.5)}; border-radius: 12px; padding: 10px; }}"
    return f"QLabel {{ background: {PANEL}; border-radius: 6px; padding: 8px; }}"


def surround_css() -> str:
    """Background around a figure canvas (the scroll area of a graph or layout view)."""
    if style() == "glass":
        return f"QScrollArea {{ background: {rgba(INNER, 0.55)}; border: none; border-radius: 12px; }}"
    return f"QScrollArea {{ background: {PANEL}; }}"


def palette() -> QPalette:
    pal = QPalette()
    for role, c in ((QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE),
                    (QPalette.AlternateBase, mix(BG, PANEL, 0.6)), (QPalette.Text, TEXT), (QPalette.Button, PANEL),
                    (QPalette.ButtonText, TEXT), (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT),
                    (QPalette.Highlight, SKY.base), (QPalette.HighlightedText, TEXT), (QPalette.PlaceholderText, TEXT2),
                    (QPalette.Link, NOTE), (QPalette.Mid, BORDER_STRONG), (QPalette.Midlight, BORDER),
                    (QPalette.Dark, BORDER_STRONG), (QPalette.Light, SURFACE)):
        pal.setColor(role, QColor(c))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(DISABLED))
    return pal


FONT_PT = 12    # interface font size (semibold)
_keep: dict = {}   # the style object and the event filter live as long as the application


def apply(app: QApplication, style: str = "glass") -> None:
    """Style + palette + font + stylesheet; "glass" (default) or "flat". Call right after the QApplication is created,
    and again to switch the look at run time (cards, toolbar and windows follow through ``notifier``)."""
    if style not in STYLES:
        raise ValueError(f"unknown interface style {style!r}; choose one of {STYLES}")
    _state["style"] = style
    app.setStyle("Fusion")
    if style == "glass":
        _keep.setdefault("style", GlassStyle())
        if "filter" not in _keep:
            _keep["filter"] = _AppFilter(app)
            app.installEventFilter(_keep["filter"])
    elif "style" in _keep:
        _glass_style_widgets(app, False)
    app.setPalette(palette())
    f = QFont("Arial", FONT_PT)
    f.setFamilies(["Arial", "Microsoft YaHei UI"])  # Chinese text falls back to YaHei
    f.setWeight(QFont.DemiBold)
    app.setFont(f)
    app.setAttribute(Qt.AA_DontShowIconsInMenus, True)  # the line icons belong to the toolbar only
    from pathlib import Path
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("bioapp.ico"))))  # title bar + taskbar
    app.setStyleSheet(stylesheet(style))
    if style == "glass":
        _glass_style_widgets(app, True)
    notifier.changed.emit(style)


def unapply(app: QApplication) -> None:
    """Undo ``apply`` (tests): no stylesheet, no glass painting, Fusion."""
    _state["style"] = "flat"
    if "style" in _keep:
        _glass_style_widgets(app, False)
    if "filter" in _keep:
        app.removeEventFilter(_keep.pop("filter"))
    app.setStyleSheet("")
    app.setStyle("Fusion")
    notifier.changed.emit("flat")
