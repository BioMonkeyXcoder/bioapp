"""Graph specs and rendering (Matplotlib, never pyplot) plus figure export.

A figure is always rebuilt from a ``PlotSpec`` + the table + stored analysis results, so changing
colours, fonts or sizes never needs a re-analysis.

Conventions
- Sizes in mm, fonts and line widths in pt. Layout panels use a top-left origin (y grows downwards).
- Excluded cells: drawn as hollow markers in "replicates"/scatter/point modes; summary-based modes
  (mean +- error, bars) omit them, exactly like the statistics do.
- Every series/fit/bracket/axis/title/legend artist carries a ``gid`` (see DESIGN.md "Plot contract").
  All parts of one object (marker line, error-bar caps, ...) share the same gid.
- Brackets stack above the data in greedy levels (narrow spans lowest); XY graphs and log-Y
  category graphs do not draw brackets. Grouped comparisons with ``row`` None are skipped.
- Figures are ``_Figure`` objects: drawing them (Qt canvas, savefig, ...) applies the mathtext font
  settings automatically, so no caller needs its own rc_context except ``export`` for file font types.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import Bbox, ScaledTranslation
from scipy.signal import find_peaks
from scipy.stats import t as _t

MM = 25.4
# Colour-blind-safe (Okabe-Ito order, black first as in Prism).
PALETTE = ["#000000", "#D55E00", "#0072B2", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#7F7F7F"]
MARKERS = ["o", "s", "^", "v", "D", "<", ">", "p"]
FALLBACK_FONTS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
STYLES = {
    "xy": ("mean_error", "replicates"),
    "column": ("scatter", "bar", "box", "violin"),
    "grouped": ("bars", "scatter"),
    "histogram": ("bars", "overlay", "points"),
    "contingency": ("grouped", "stacked"),
    "nested": ("subject_markers", "subject_colors"),
    "trace": ("line", "dots"),
}
BLI_SHOW = ("both", "raw", "fit")
BLI_PALETTES = ("Blues", "Greens", "Oranges", "Purples", "Reds", "Greys", "viridis_r", "magma_r", "Vivid blues")  # high = dark
# colours sampled from a Prism sensorgram, lowest concentration first; spread over the whole range of the sensors
BLI_CUSTOM = {"Vivid blues": ["#DEF3FF", "#9FCFFF", "#5EAEFF", "#0281FF", "#004F9D"]}
TRACE_NORMALIZE = ("none", "max", "window")
TRACE_BASELINE = ("none", "min", "window")
DECIMATE_FACTOR = 4  # preview: a trace with more samples than this many per pixel column is drawn as its min/max envelope
ERRORS = ("sd", "sem", "ci95", "none")
BANDS = ("none", "ci", "pi", "both")
BAND_STYLES = ("fill", "dotted")
LEGEND_LOCS = ("outside", "best", "upper left", "upper right", "lower left", "lower right", "manual")
HEAT_GAP_MM, HEAT_DEND_MM, HEAT_CB_MM = 1.5, 12.0, 3.5
_LS = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
LINE_STYLES = (*_LS, "custom")
ERR_DIRS = ("both", "above", "below")
HATCHES = ("", "/", "//", "\\", "\\\\", "|", "-", "+", "x", "o", ".", "*")
BACKGROUNDS = ("white", "transparent")


# --------------------------------------------------------------------------- specs
class _Dict:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class SeriesStyle(_Dict):
    """Style of one data set. Every field after ``visible`` is optional: its default reproduces the look of the
    graphs made before it existed ("" / 0 / -1 / None = automatic), so old .biofig files render unchanged.
    Colours are anything Matplotlib accepts (use "#RRGGBB"); alphas run 0-1; widths are in pt."""
    dataset_id: str
    label: str = ""
    color: str = "#000000"
    marker: str = "o"
    marker_size: float = 5.0
    fill: bool = True
    line_width: float = 1.0
    visible: bool = True
    # markers
    marker_alpha: float = 1.0
    marker_fill: str = ""  # "" follows ``fill`` | "filled" | "hollow" | a fill colour
    marker_edge_color: str = ""  # "" = series colour
    marker_edge_width: float = 0.0  # 0 = 0.8 * line_width
    # line: connecting line, fitted curve, mean line, survival / ROC line
    curve_color: str = ""  # "" = series colour
    curve_alpha: float = 1.0
    curve_width: float = 0.0  # 0 = line_width (x1.2 for fitted curves and mean lines)
    curve_style: str = "solid"  # solid | dashed | dotted | dashdot | custom
    curve_dash: str = ""  # custom pattern "on,off[,on,off...]" in units of the line width
    # error bars
    err_color: str = ""  # "" = series colour
    err_alpha: float = 1.0
    err_width: float = 0.0  # 0 = line_width
    err_cap: float = -1.0  # cap length in pt; -1 = 2.5 + line_width
    err_dir: str = "both"  # both | above | below
    # bars, boxes, violins, histogram / contingency bars
    fill_color: str = ""  # "" = the series colour lightened
    fill_alpha: float | None = None  # None = opaque (35 % for overlaid histograms)
    edge_color: str = ""  # "" = series colour
    edge_width: float = 0.0  # 0 = line_width
    hatch: str = ""
    # confidence / prediction band
    band_color: str = ""  # "" = series colour
    band_alpha: float | None = None  # None = 18 % (confidence) / 10 % (prediction); a value applies to the confidence band


@dataclass
class AxisSpec(_Dict):
    """``log``: true log axis for raw values. ``log10_values``: the values are already log10 and the
    axis stays linear; "values" labels ticks with the log values, "power" labels them 10^n."""
    title: str = ""
    min: float | None = None
    max: float | None = None
    major_tick: float | None = None
    log: bool = False
    log10_values: str = "off"  # off | values | power
    font_size: float | None = None  # None = PlotSpec.font_size_pt
    minor_ticks: bool = False
    # canvas editing: axis title moved by (title_dx, title_dy) pt from its automatic place, bold / italic title
    title_dx: float = 0.0
    title_dy: float = 0.0
    title_bold: bool = False
    title_italic: bool = False
    # tick labels: rotated by ``tick_rotation`` degrees (read upwards, ending at the tick); ``tick_decimals`` fixes the
    # number of decimals (None = automatic)
    tick_rotation: float = 0.0
    tick_decimals: int | None = None
    title_size_pt: float | None = None  # None = the tick label size


@dataclass
class BracketSpec(_Dict):
    comparison_id: str
    label: str | None = None  # None = the comparison's summary (ns / * / ** ...)
    show_ns: bool = False
    dy_pt: float = 0.0  # dragged up (+) or down (-) from its stacked place, in pt


@dataclass
class BliSpec(_Dict):
    """BLI sensorgram preset of a "trace" graph (empty ``sensors`` = an ordinary trace graph).

    ``sensors``: one record per sensor {"data": data set id of the raw curve, "fit": data set id of the fitted curve
    ("" = none), "conc": molar concentration or None, "well", "start", "stop", "metrics": {file key: number}}.
    ``boundary``: association / dissociation boundary (s, in the data's own time; None = none), ``t0``: association start
    (s) that ``rebase`` moves to 0 (None = the first sample). ``show``: both | raw | fit. Raw curves take the colours of
    their series (``apply_bli_palette`` gives them a gradient by concentration, highest darkest); fitted curves are
    drawn on top with ``fit_color`` ("" = the colour of their concentration; with ``show`` "fit" this makes the fits
    follow the gradient), ``fit_alpha``, ``fit_width`` (pt), ``fit_style`` / ``fit_dash`` and are left out of the
    legend. The legend lists concentrations from high to low. ``phase_line``: dashed vertical line at the boundary,
    ``phase_labels``: "Association" / "Dissociation" above the two phases. ``metrics``: names of file metrics
    (e.g. "KOff", "KObs") printed in the legend: a value shared by all sensors in the legend title (with its
    "Error ..." entry as +-), a per-sensor value after each concentration. ``phase_width`` / ``phase_color`` style the
    phase line; ``kinetics`` + ``kinetics_show`` print KD / kon / koff lines under the legend (``bli_kinetics``).
    Nothing is ever computed from the data."""
    sensors: list = field(default_factory=list)
    boundary: float | None = None
    t0: float | None = None
    show: str = "both"
    rebase: bool = False
    phase_line: bool = True
    phase_labels: bool = False
    fit_color: str = "#000000"
    fit_alpha: float = 1.0
    fit_width: float = 0.5
    fit_style: str = "solid"
    fit_dash: str = ""
    palette: str = "Blues"
    metrics: list = field(default_factory=list)
    # one style for every raw curve (None / "" = each curve keeps its own series setting); colours stay per curve
    raw_width: float | None = None
    raw_alpha: float | None = None
    raw_style: str = ""
    raw_dash: str = ""
    # phase line: width in pt (None = 0.8 x the axis width) and colour
    phase_width: float | None = None
    phase_color: str = "#6E6E6E"
    # kinetics block below the legend: [{"name": "KD" | "kon" | "koff" | free text, "value": number | None, "unit": text}],
    # [] = the file's own KD / kon / koff metrics (see ``bli_kinetics``). Typed-in or file values only, never computed.
    kinetics: list = field(default_factory=list)
    kinetics_show: bool = False


@dataclass
class TraceSpec(_Dict):
    """Options of a "trace" graph: lines drawn straight from the raw samples of one or more XY tables.

    ``sources``: data set id -> table id for series that live in another table than ``PlotSpec.table_id``.
    ``right``: data set ids drawn against the right Y axis (``y2_axis``). Baseline subtraction, normalisation and the
    stacking offset apply to the left-axis traces, in that order. ``baseline``: min (subtract the trace minimum) |
    window (subtract the mean inside ``baseline_window`` = [x0, x1]). ``normalize``: max (divide by the trace maximum) |
    window (divide by the maximum inside ``norm_window``); the left axis is then titled ``norm_title``.
    ``stack_offset``: trace k (left axis, in series order) is lifted by k * stack_offset (Y tick labels are hidden).
    ``events``: [{"x", "label", "kind", "run"}] drawn as small ticks with labels below the X axis (thinned when
    crowded); ``event_run`` "" shows all, otherwise only that run's; ``event_kinds`` [] shows all kinds, otherwise only
    those ("Fractions", "Injection", ...). ``peaks``: label the apex volume of every peak with
    prominence >= ``peak_prominence`` * the trace's Y span (``peak_series``: data set ids, empty = the first left trace)."""
    sources: dict = field(default_factory=dict)
    right: list = field(default_factory=list)
    y2_axis: AxisSpec = field(default_factory=AxisSpec)
    normalize: str = "none"
    norm_window: list = field(default_factory=list)
    norm_title: str = "Normalised signal"
    baseline: str = "none"
    baseline_window: list = field(default_factory=list)
    stack_offset: float = 0.0
    show_events: bool = True
    event_run: str = ""
    event_kinds: list = field(default_factory=list)
    events: list = field(default_factory=list)
    peaks: bool = False
    peak_prominence: float = 0.05
    peak_decimals: int = 2
    peak_series: list = field(default_factory=list)
    bli: BliSpec = field(default_factory=BliSpec)

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        d["y2_axis"] = AxisSpec.from_dict(d["y2_axis"]) if "y2_axis" in d else AxisSpec()
        d["bli"] = BliSpec.from_dict(d["bli"]) if "bli" in d else BliSpec()
        return cls(**d)


@dataclass
class PlotSpec(_Dict):
    id: str
    table_id: str
    kind: str = "xy"  # xy | column | grouped | contingency | nested | trace | NEW_KINDS (result-driven)
    style: str = "mean_error"
    error: str = "sd"
    fit_analysis_id: str | None = None
    stats_analysis_id: str | None = None
    series: list[SeriesStyle] = field(default_factory=list)
    x_axis: AxisSpec = field(default_factory=AxisSpec)
    y_axis: AxisSpec = field(default_factory=AxisSpec)
    title: str = ""
    brackets: list[BracketSpec] = field(default_factory=list)
    legend: bool = False
    legend_loc: str = "outside"  # outside (right of the axes) | best (inside, away from the data) | upper left | ...
    name: str = ""  # navigator name; "" = the title or "Graph of <table>"
    width_mm: float = 80.0
    height_mm: float = 60.0
    font_family: str = "Arial"
    font_size_pt: float = 8.0
    line_width_pt: float = 1.0
    jitter_seed: int = 0
    whisker: str = "minmax"  # box plots: minmax | tukey
    show_points: bool = True  # box / violin: overlay the individual values
    # --- P6-P8 options
    analysis_id: str | None = None  # result read by the result-driven kinds (and xy bands / interpolated points)
    x_dataset: str | None = None  # xy on a MultipleVariables table: the data set used as X
    value: str = ""  # histogram counts|relative|cumulative; contingency counts|percent; heatmap raw|zscore;
    #                  scree eigenvalues|explained ("" = first option)
    band: str = "none"  # xy: none | ci | pi | both (needs curve ci_low/ci_high/pi_low/pi_high)
    band_style: str = "fill"  # fill | dotted
    show_interpolated: bool = True  # xy: interpolated unknowns with drop lines
    percent: bool = False  # survival: Y as percent instead of fraction
    show_ci: bool = False  # survival: shade the confidence interval
    show_censor: bool = True  # survival: censor tick marks
    show_at_risk: bool = False  # survival: number-at-risk table below the axes
    annotation: str = ""  # free text in the upper right corner (e.g. "Log-rank P = 0.012")
    pc_x: int = 1  # PCA plots: principal components on X / Y (1-based)
    pc_y: int = 2
    color_by: str | None = None  # PCA scores / biplot: categorical data set of the source table
    show_values: bool = True  # correlation matrix: r printed in the cells
    show_dendrograms: bool = True  # heatmap: draw dendrograms when the result has them
    colormap: str = "viridis"
    # --- graph-level style
    axis_color: str = "#000000"  # axis lines and tick marks
    axis_width: float = 0.0  # 0 = line_width_pt
    background: str = "white"  # "white" | "transparent" | a colour
    bracket_color: str = "#000000"
    bracket_width: float = 0.0  # 0 = 0.8 * line_width_pt
    star_size_pt: float = 0.0  # bracket label (stars / P value) size; 0 = font_size_pt
    legend_frame: bool = False
    # --- look (None / False = the automatic values of the graphs made before)
    tick_bold: bool = False  # bold tick labels
    tick_length_pt: float | None = None  # None = 3.5
    tick_width_pt: float | None = None  # None = the axis width
    legend_handle_len: float | None = None  # legend line / patch length in legend font sizes; None = 1.6
    legend_size_pt: float | None = None  # None = font_size_pt
    legend_spacing: float | None = None  # space between legend entries in legend font sizes; None = 0.5
    # --- canvas editing (all optional; the defaults draw the graph as before)
    legend_x: float | None = None  # legend_loc "manual": upper-left corner of the legend in axes fractions
    legend_y: float | None = None
    title_dx: float = 0.0  # graph title moved by (dx, dy) pt from its automatic place
    title_dy: float = 0.0
    title_size_pt: float = 0.0  # 0 = 1.1 * font_size_pt
    title_bold: bool = False
    title_italic: bool = False
    texts: list[dict] = field(default_factory=list)  # free text, see ``new_text``
    trace: TraceSpec = field(default_factory=TraceSpec)  # kind "trace" only

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        d["trace"] = TraceSpec.from_dict(d["trace"]) if "trace" in d else TraceSpec()
        d["series"] = [SeriesStyle.from_dict(s) for s in d.get("series", [])]
        d["x_axis"] = AxisSpec.from_dict(d["x_axis"]) if "x_axis" in d else AxisSpec()
        d["y_axis"] = AxisSpec.from_dict(d["y_axis"]) if "y_axis" in d else AxisSpec()
        d["brackets"] = [BracketSpec.from_dict(b) for b in d.get("brackets", [])]
        return cls(**d)


@dataclass
class PanelSpec(_Dict):
    graph_id: str
    x_mm: float
    y_mm: float  # from the top of the page
    w_mm: float
    h_mm: float
    label: str = ""


@dataclass
class LayoutSpec(_Dict):
    id: str
    title: str = ""
    width_mm: float = 180.0
    height_mm: float = 150.0
    panels: list[PanelSpec] = field(default_factory=list)
    label_size_pt: float = 12.0

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        d["panels"] = [PanelSpec.from_dict(p) for p in d.get("panels", [])]
        return cls(**d)


NEW_KINDS = ("survival", "histogram", "contingency", "nested", "qq", "residuals", "heatmap", "pca_scores",
             "pca_loadings", "biplot", "scree", "roc", "correlation_matrix")
_TABLE_KIND = {"XY": "xy", "Column": "column", "Grouped": "grouped", "Contingency": "contingency",
               "Survival": "survival", "Nested": "nested"}
_KIND_STYLE = {"xy": "mean_error", "column": "scatter", "grouped": "bars", "contingency": "grouped",
               "nested": "subject_markers", "histogram": "bars"}
_KIND_TITLES = {"survival": ("Time", "Fraction surviving"), "histogram": ("", "Frequency"),
                "contingency": ("", "Count"), "qq": ("Theoretical quantiles", "Sample quantiles"),
                "residuals": ("X", "Residual"), "roc": ("1 − Specificity", "Sensitivity"),
                "scree": ("Principal component", "Eigenvalue")}
_NO_LEGEND = ("column", "nested", "heatmap", "correlation_matrix", "scree", "pca_scores", "pca_loadings", "biplot",
              "roc")


def new_text(x: float, y: float, text: str = "Text", size_pt: float = 8.0, color: str = "#000000") -> dict:
    """A free text of ``PlotSpec.texts``: (x, y) in axes fractions (the text is left-aligned, centred on y)."""
    return {"text": text, "x": x, "y": y, "size_pt": size_pt, "color": color, "rotation": 0.0, "bold": False,
            "italic": False}


def default_series(table) -> list[SeriesStyle]:
    return [SeriesStyle(ds.id, ds.title, PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])
            for i, ds in enumerate(table.datasets)]


# ---- palettes and style templates
# Macaron: muted hues, each at least 3:1 against white (readable as points / lines) and >= 20 CIE76 apart.
# Prism Colors: an approximation of Prism's "Colors" scheme (the exact values are unverified).
PALETTES = {
    "Colour-blind safe": PALETTE,
    "Macaron": ["#C4577F", "#CE7A3A", "#A98A22", "#3F9E7E", "#4A8CC4", "#8A6CC0", "#BA5DA5", "#7F8C73"],
    "Prism Colors": ["#000000", "#E31A1C", "#1F4FB4", "#2CA02C", "#FF7F0E", "#7B3FA0", "#17A2B8", "#8C564B"],
    "Grayscale": ["#000000", "#404040", "#707070", "#999999", "#B8B8B8", "#202020", "#555555", "#858585"],
    "Viridis": [mpl.colors.to_hex(mpl.colormaps["viridis"](v)) for v in np.linspace(0, 0.8, 8)],
}
_COLOR_OVERRIDES = ("marker_edge_color", "curve_color", "err_color", "fill_color", "edge_color", "band_color")
_IDENTITY = ("dataset_id", "label", "visible")  # data bindings: never part of a style
GRAPH_STYLE_FIELDS = ("error", "band", "band_style", "font_family", "font_size_pt", "line_width_pt", "axis_color",
                      "axis_width", "background", "bracket_color", "bracket_width", "star_size_pt", "legend_frame",
                      "legend_loc")


def recolor(s: SeriesStyle, color: str) -> None:
    """Give a series one colour: its per-part colour overrides (line, error bars, fill, band) follow it again."""
    s.color = color
    for f in _COLOR_OVERRIDES:
        setattr(s, f, "")
    if s.marker_fill not in ("", "filled", "hollow"):
        s.marker_fill = ""


def apply_palette(spec: PlotSpec, name: str) -> None:
    """Recolour every series with palette ``name`` (cycling) and drop their per-part colour overrides, so lines,
    error bars and bands follow the new colour. Macaron also gives pastel fills with slightly deeper edges."""
    cols = PALETTES[name]
    for i, s in enumerate(spec.series):
        recolor(s, cols[i % len(cols)])
        if name == "Macaron":
            s.fill_color = mpl.colors.to_hex(_lighten(s.color, 0.3))
            s.edge_color = mpl.colors.to_hex([0.85 * c for c in mpl.colors.to_rgb(s.color)])


def apply_bli_palette(spec: PlotSpec, name: str | None = None) -> None:
    """Colour the raw curves of a BLI graph by concentration with the sequential palette ``name`` (default: the one
    stored in ``spec.trace.bli.palette``): the highest concentration gets the darkest colour. Drops colour overrides."""
    bli = spec.trace.bli
    name = name or bli.palette
    if name not in BLI_PALETTES:
        raise ValueError(f"unknown BLI palette {name!r}; use one of {BLI_PALETTES}")
    bli.palette = name
    by = {s.dataset_id: s for s in spec.series}
    low_to_high = sorted((r for r in bli.sensors if r["data"] in by), key=lambda r: (r["conc"] is not None, r["conc"] or 0.0))
    if name in BLI_CUSTOM:  # straight blend between the anchor colours, exact at the anchors
        anchors = np.array([mpl.colors.to_rgb(c) for c in BLI_CUSTOM[name]])
        cmap = lambda pos: tuple(np.interp(pos, np.linspace(0, 1, len(anchors)), anchors[:, i]) for i in range(3))  # noqa: E731
    else:
        cmap = mpl.colormaps[name]
    for k, r in enumerate(low_to_high):
        n = len(low_to_high)
        pos = (k / (n - 1) if n > 1 else 1.0) if name in BLI_CUSTOM else 0.45 + 0.55 * k / (n - 1) if n > 1 else 0.8
        recolor(by[r["data"]], mpl.colors.to_hex(cmap(pos)))


LOOKS = ("default", "bold")
BOLD_PLOT_MM = 57.0  # side of the plot area of the bold look (square)


def nice_tick_step(span: float, target: int = 8) -> float:
    """A round tick interval (1, 1.5, 2, 2.5 or 5 x 10^n) that gives about ``target`` intervals over ``span``."""
    raw = span / target
    e = math.floor(math.log10(raw))
    return min((m * 10.0 ** k for k in range(e - 1, e + 2) for m in (1, 1.5, 2, 2.5, 5)), key=lambda c: abs(math.log(c / raw)))


def _x_span(spec: PlotSpec) -> float | None:
    """The X range a trace graph will show, from its axis limits or the sensors' start / stop times; None if unknown."""
    xa, recs = spec.x_axis, [r for r in spec.trace.bli.sensors if r.get("start") is not None and r.get("stop") is not None]
    if xa.min is not None and xa.max is not None:
        return xa.max - xa.min
    return max(r["stop"] for r in recs) - min(r["start"] for r in recs) if recs and xa.min is None and xa.max is None else None


def apply_look(spec: PlotSpec, name: str) -> None:
    """Set the look fields of a graph. "bold" = the heavy publication look (bold tick labels and axis titles, 12 pt,
    2.5 pt axis lines with long outward ticks, frameless legend with long handles, round X ticks at 45 degrees when
    crowded, one-decimal Y ticks, a square plot area with the legend outside); BLI graphs also get thick opaque
    "Vivid blues" raw curves, thin grey fits on top, a thick dashed grey phase line and the kinetics block. "default"
    puts the same fields back to the values of a new graph. Data, titles and axis limits stay as they are."""
    if name not in LOOKS:
        raise ValueError(f"unknown look {name!r}; use one of {LOOKS}")
    bold = name == "bold"
    sized = spec.kind in ("trace", "xy", "column", "grouped")
    kept_size = sized and (spec.width_mm, spec.height_mm) == _bold_size(spec)  # still the size the bold look gave it
    spec.font_size_pt = 12.0 if bold else 8.0
    spec.tick_bold, spec.title_bold, spec.legend_frame = bold, bold, False
    spec.axis_width = 1.7 if bold else 0.0
    spec.tick_length_pt, spec.tick_width_pt = (5.5, 1.7) if bold else (None, None)
    spec.legend_handle_len, spec.legend_size_pt, spec.legend_spacing = (1.4, 13.2, 0.8) if bold else (None, None, None)
    if bold and spec.legend_loc == "outside":  # a little clear of the axis and starting above the plot, as Prism does
        spec.legend_loc, spec.legend_x, spec.legend_y = "manual", 1.07, 1.09
    elif not bold and spec.legend_loc == "manual" and (spec.legend_x or 0.0) >= 1.0:
        spec.legend_loc, spec.legend_x, spec.legend_y = "outside", None, None
    spec.title_size_pt = 15.6 if bold else 0.0
    for ax in (spec.x_axis, spec.y_axis):
        ax.title_bold, ax.tick_rotation, ax.tick_decimals = bold, 0.0, None
        ax.title_size_pt = 13.2 if bold else None
    spec.x_axis.major_tick = None
    spec.y_axis.tick_decimals = 1 if bold else None
    span = _x_span(spec)
    if bold and span:
        spec.x_axis.major_tick = nice_tick_step(span)
        spec.x_axis.tick_rotation = 45.0 if span / spec.x_axis.major_tick >= 5.5 else 0.0
    b = spec.trace.bli
    if b.sensors:
        b.raw_width, b.raw_alpha = (3.75, 1.0) if bold else (None, None)
        b.fit_color, b.fit_alpha, b.fit_width, b.fit_style, b.fit_dash = (
            ("#8C8C8C", 1.0, 1.4, "solid", "") if bold else ("#000000", 1.0, 0.5, "solid", ""))
        b.phase_width, b.phase_color = (1.7, "#8C8C8C") if bold else (None, "#6E6E6E")
        b.kinetics_show = bold
        apply_bli_palette(spec, "Vivid blues" if bold else "Blues")
    if sized and (bold or kept_size):
        spec.width_mm, spec.height_mm = _bold_size(spec) if bold else (90.0 if b.sensors else 120.0, 60.0) if spec.kind == "trace" else (80.0, 60.0)


def _bold_size(spec: PlotSpec) -> tuple[float, float]:
    """Page size (mm) that gives the bold look a square plot area of ``BOLD_PLOT_MM`` plus its margins and legend."""
    bli = replace(spec.trace.bli, kinetics_show=True)
    s = replace(spec, font_size_pt=12.0, tick_bold=True, axis_width=1.7, tick_length_pt=5.5, tick_width_pt=1.7,
                legend_handle_len=1.4, legend_spacing=0.8, trace=replace(spec.trace, bli=bli),
                x_axis=replace(spec.x_axis, tick_rotation=45.0 if spec.x_axis.tick_rotation or _x_span(spec) else 0.0))
    left, bottom, right, top = _margins(s, None, {})
    return round(left + BOLD_PLOT_MM + right, 1), round(bottom + BOLD_PLOT_MM + top, 1)


def style_template(spec: PlotSpec, dataset_id: str | None = None) -> dict:
    """JSON-compatible style of a graph: everything except data bindings (ids, labels, visibility, tables, results).
    With ``dataset_id`` the template holds that one series (it is then applied to every series)."""
    own = [s for s in spec.series if dataset_id in (None, s.dataset_id)]
    return {"bioapp_style": 1, "graph": {f: getattr(spec, f) for f in GRAPH_STYLE_FIELDS},
            "series": [{k: v for k, v in asdict(s).items() if k not in _IDENTITY} for s in own]}


def apply_style(spec: PlotSpec, tmpl: dict, graph: bool = True, keep_color: bool = False) -> None:
    """Paste a style template: series ``i`` takes template series ``i`` (cycling). ``keep_color`` leaves each
    series' own colour (and marker shape) alone. Unknown keys are ignored."""
    if not isinstance(tmpl, dict) or tmpl.get("bioapp_style") != 1:
        raise ValueError("not a bioapp style template")
    known = {k for k in SeriesStyle.__dataclass_fields__ if k not in _IDENTITY}
    if graph:
        for f, v in (tmpl.get("graph") or {}).items():
            if f in GRAPH_STYLE_FIELDS:
                setattr(spec, f, v)
    src = [{k: v for k, v in d.items() if k in known} for d in tmpl.get("series") or []]
    for i, s in enumerate(spec.series if src else []):
        d = dict(src[i % len(src)])
        if keep_color:
            d.pop("color", None)
            d.pop("marker", None)
        for k, v in d.items():
            setattr(s, k, v)


def save_style(tmpl: dict, path) -> None:
    Path(path).write_text(json.dumps(tmpl, indent=1), encoding="utf-8")


def load_style(path) -> dict:
    tmpl = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tmpl, dict) or tmpl.get("bioapp_style") != 1:
        raise ValueError(f"{path} is not a bioapp style file")
    return tmpl


def default_spec(table, graph_id: str, analysis_id: str | None = None, kind: str | None = None) -> PlotSpec:
    """Prism-like defaults for the table kind.

    Survival -> "survival" (set ``analysis_id`` to the Kaplan-Meier result), Contingency -> "contingency",
    Nested -> "nested". MultipleVariables -> "pca_scores" when ``analysis_id`` (a PCA result) is given, otherwise a
    scatter ("xy") of the first two numeric variables. ``kind`` overrides the choice (e.g. "heatmap", "qq").
    """
    if kind == "trace":
        spec = trace_graph(graph_id, [(table, ds.id, ds.title, False) for ds in table.datasets])
        spec.y_axis.title = table.datasets[0].title if len(table.datasets) == 1 else ""
        return spec
    series = default_series(table)
    numeric = [ds for ds in table.datasets if ds.categories is None]
    if table.kind == "MultipleVariables" and kind in (None, "xy"):
        if analysis_id is None:
            if len(numeric) < 2:
                raise ValueError("a scatter needs at least two numeric variables")
            xd, yd = numeric[:2]
            return PlotSpec(id=graph_id, table_id=table.id, kind="xy", style="replicates",
                            series=[SeriesStyle(yd.id, yd.title)], x_dataset=xd.id,
                            x_axis=AxisSpec(title=xd.title), y_axis=AxisSpec(title=yd.title))
        kind = kind or "pca_scores"
    kind = kind or _TABLE_KIND[table.kind]
    xt, yt = _KIND_TITLES.get(kind, (table.x_title if kind == "xy" else "", ""))
    if kind == "survival" and table.x_title != "X":
        xt = table.x_title
    cats = [ds for ds in table.datasets if ds.categories is not None]
    color_by = cats[0].id if kind in ("pca_scores", "biplot") and cats else None
    # log-dose X (the title says so): integer ticks written as plain numbers
    log_x = kind == "xy" and table.kind == "XY" and re.search(r"(?<![a-z])log", table.x_title.lower()) is not None
    return PlotSpec(id=graph_id, table_id=table.id, kind=kind, style=_KIND_STYLE.get(kind, "default"), series=series,
                    x_axis=AxisSpec(title=xt, log10_values="values" if log_x else "off"), y_axis=AxisSpec(title=yt),
                    analysis_id=analysis_id,
                    color_by=color_by, colormap="RdBu_r" if kind == "correlation_matrix" else "viridis",
                    legend=bool(color_by) or (kind not in _NO_LEGEND and len(series) > 1))


def trace_graph(graph_id: str, entries, events=(), title: str = "") -> PlotSpec:
    """A "trace" graph. ``entries``: (table, data set id, legend label, on the right Y axis) per trace; the first table
    is the graph's ``table_id`` and every table must be an XY table. ``events``: dicts as in ``TraceSpec.events``."""
    if not entries:
        raise ValueError("a trace graph needs at least one data set")
    first = entries[0][0]
    for t, ds_id, _, _ in entries:
        if t.kind != "XY":
            raise ValueError(f"a trace graph needs XY tables, {t.title!r} is {t.kind}")
        t.dataset(ds_id)
    series = [SeriesStyle(ds_id, str(label), PALETTE[i % len(PALETTE)], line_width=0.6)
              for i, (_, ds_id, label, _) in enumerate(entries)]
    trace = TraceSpec(sources={ds_id: t.id for t, ds_id, _, _ in entries if t.id != first.id},
                      right=[ds_id for _, ds_id, _, r in entries if r], events=[dict(e) for e in events])
    return PlotSpec(id=graph_id, table_id=first.id, kind="trace", style="line", series=series, title=title,
                    x_axis=AxisSpec(title=first.x_title), y_axis=AxisSpec(), legend=len(series) > 1,
                    width_mm=120.0, height_mm=60.0, trace=trace)


def prune_trace_graphs(graphs: dict, doomed: set) -> None:
    """A trace graph whose tables are (partly) being deleted keeps the series of the others (the first remaining table
    becomes its ``table_id``). One that loses all of them keeps its ``table_id``, so the caller deletes it like any graph
    of a deleted table (``table_id in doomed``). Call before removing the tables."""
    for s in graphs.values():
        if s.kind != "trace":
            continue
        table_of = {x.dataset_id: s.trace.sources.get(x.dataset_id, s.table_id) for x in s.series}
        keep = [x for x in s.series if table_of[x.dataset_id] not in doomed]
        if not keep or len(keep) == len(s.series):
            continue
        ids, first = {x.dataset_id for x in keep}, table_of[keep[0].dataset_id]
        s.series, s.table_id = keep, first
        s.trace.sources = {d: t for d, t in table_of.items() if d in ids and t != first}
        s.trace.right = [d for d in s.trace.right if d in ids]
        s.trace.peak_series = [d for d in s.trace.peak_series if d in ids]
        for r in s.trace.bli.sensors:
            r.update({k: "" for k in ("data", "fit") if r[k] not in ids})
        s.trace.bli.sensors = [r for r in s.trace.bli.sensors if r["data"] or r["fit"]]


def _has_curve(res: dict) -> bool:
    return any(d.get("curve") for d in (res.get("datasets") or {}).values())


def _has_band(res: dict) -> bool:
    return any("ci_low" in (d.get("curve") or {}) for d in (res.get("datasets") or {}).values())


FIT_KINDS = ("nonlin_fit", "linear_regression", "deming_regression")  # analyses that draw curves on an XY graph
# analyses that give a Column / Grouped graph its natural default (with brackets when they have comparisons)
STAT_KINDS = ("descriptive", "t_test", "mann_whitney", "wilcoxon_paired", "one_way_anova", "kruskal_wallis",
              "friedman", "two_way_anova")
TRANSFORM_KINDS = ("normalize", "subtract_baseline")  # their graph is the graph of the output table


def default_brackets(res: dict | None) -> list[BracketSpec]:
    """Brackets shown by default: every comparison of the result that is significant (ns stays hidden)."""
    return [BracketSpec(c["id"]) for c in (res or {}).get("comparisons") or [] if c.get("summary") != "ns"]


def auto_link(spec: PlotSpec, analyses) -> None:
    """Feed a new graph from the most recent fit (XY) / comparison test (Column, Grouped, Nested) of its table.
    ``analyses``: (id, kind, result) of the table's analyses in creation order; a result still being computed
    (None) counts for the fit kinds, so the curves appear as soon as it arrives."""
    for aid, kind, res in analyses:
        res = res or {}
        if res.get("status") == "failed":
            continue
        if spec.kind == "xy" and (kind in FIT_KINDS or _has_curve(res)):
            spec.fit_analysis_id = aid
        if spec.kind in ("column", "grouped", "nested") and res.get("comparisons"):
            spec.stats_analysis_id, spec.brackets = aid, default_brackets(res)


def specs_for_result(table, analysis_id: str, kind: str, res: dict, make_id) -> list[PlotSpec]:
    """The natural default graph(s) of one analysis result ("Graph this result"); [] when it has none.
    ``make_id(plot_kind)`` returns the id of each new spec."""
    if not res or res.get("status") == "failed":
        return []
    out = []

    def add(k, **kw):
        s = default_spec(table, make_id(k), analysis_id, k)
        for a, v in kw.items():
            setattr(s, a, v)
        out.append(s)
    if res.get("curves"):
        add("survival", annotation="auto", show_at_risk=bool(res.get("at_risk")))
    if res.get("pca"):
        add("pca_scores")
        add("scree")
    m = res.get("matrix") or {}
    if "values" in m:
        add("heatmap")
    if "r" in m:
        add("correlation_matrix")
    if res.get("qq"):
        add("qq")
    if res.get("histograms"):
        add("histogram")
    if _roc_curves(res):
        add("roc", legend=len(_roc_curves(res)) > 1)
    if table.kind == "XY" and _has_curve(res):
        add("xy", fit_analysis_id=analysis_id, band="ci" if _has_band(res) else "none",
            style="replicates" if kind in ("linear_regression", "deming_regression") and table.y_format == "replicates"
            else "mean_error")
    if res.get("residuals"):
        add("residuals")
    if table.kind in ("Column", "Grouped", "Nested") and (kind in STAT_KINDS or res.get("comparisons")):
        add(_TABLE_KIND[table.kind], analysis_id=None, stats_analysis_id=analysis_id, brackets=default_brackets(res))
    if kind in TRANSFORM_KINDS and table.kind in ("XY", "Column", "Grouped"):
        add(_TABLE_KIND[table.kind])  # the app swaps in the output table
    return out


def graph_choices(table, results: dict[str, tuple[str, dict]]) -> list[tuple[str, str, str | None]]:
    """What "New graph" offers for ``table``: (label, plot kind, analysis id or None).
    ``results``: analysis id -> (title, result) of the analyses on this table."""
    base = {"XY": ("xy", "XY graph"), "Column": ("column", "Column graph"), "Grouped": ("grouped", "Grouped graph"),
            "Contingency": ("contingency", "Contingency graph"), "Nested": ("nested", "Nested graph")}
    out = []
    if table.kind in base:
        out.append((base[table.kind][1], base[table.kind][0], None))
    if table.kind == "XY":
        out.append(("Trace graph (lines from the raw samples)", "trace", None))
    if table.kind == "MultipleVariables" and sum(d.categories is None for d in table.datasets) >= 2:
        out.append(("Scatter of two variables", "xy", None))
    if table.kind in ("Grouped", "MultipleVariables"):
        out.append(("Heat map of the table", "heatmap", None))
    for k in RESULT_HAS:
        for aid, (title, r) in results.items():
            if result_ok(k, r):
                out.append((f"{k.replace('_', ' ').capitalize()} of {title}", k, aid))
    return out


# what a result must hold for each result-driven kind
RESULT_HAS = {"survival": lambda r: r.get("curves"), "pca_scores": lambda r: r.get("pca"),
              "pca_loadings": lambda r: r.get("pca"), "biplot": lambda r: r.get("pca"), "scree": lambda r: r.get("pca"),
              "heatmap": lambda r: "values" in (r.get("matrix") or {}),
              "correlation_matrix": lambda r: "r" in (r.get("matrix") or {}),
              "qq": lambda r: r.get("qq"), "histogram": lambda r: r.get("histograms"), "roc": lambda r: _roc_curves(r),
              "residuals": lambda r: r.get("residuals")}
# choices of PlotSpec.value per kind (the first is the default)
VALUES = {"histogram": ("counts", "relative", "cumulative"), "contingency": ("counts", "percent"),
          "heatmap": ("raw", "zscore"), "scree": ("eigenvalues", "explained")}


def result_ok(kind: str, res: dict | None) -> bool:
    return bool(res) and res.get("status") != "failed" and bool(RESULT_HAS[kind](res))


# --------------------------------------------------------------------------- fonts / figure
@lru_cache(maxsize=None)
def _installed(family: str) -> bool:
    return any(f.name == family for f in font_manager.fontManager.ttflist)


def _families(family: str) -> list[str]:
    return [f for f in [family, *FALLBACK_FONTS] if _installed(f)] or ["DejaVu Sans"]


def _math_rc(family: str) -> dict:
    if not _installed(family):
        return {"mathtext.fontset": "dejavusans", "mathtext.default": "regular"}
    f = family.replace(" ", r"\ ")
    return {"mathtext.fontset": "custom", "mathtext.default": "regular", "mathtext.rm": f,
            "mathtext.it": f + ":italic", "mathtext.bf": f + ":bold", "mathtext.fallback": "stix"}


class _Figure(Figure):
    """Figure whose draw applies the mathtext font settings of its font family."""
    family = "Arial"
    exact = False  # True while exporting: trace lines then keep every sample (see ``_TraceLine``)
    has_traces = False  # a trace graph is on this figure

    def draw(self, renderer):
        with mpl.rc_context(_math_rc(self.family)):
            super().draw(renderer)


def _fp(spec: PlotSpec, size: float | None = None, weight: str = "normal", style: str = "normal") -> FontProperties:
    return FontProperties(family=_families(spec.font_family), size=size or spec.font_size_pt, weight=weight, style=style)


def set_offset(artist, dx_pt: float, dy_pt: float) -> None:
    """Shift ``artist`` by (dx, dy) points on top of its own transform (axis titles, graph title); the canvas editor
    calls it again while dragging. (0, 0) restores the automatic place."""
    base = getattr(artist, "_base_transform", None)
    if base is None:
        base = artist._base_transform = artist.get_transform()
    artist.set_transform(base + ScaledTranslation(dx_pt / 72, dy_pt / 72, artist.get_figure().dpi_scale_trans))


def legend_outside(spec: PlotSpec) -> bool:
    """The legend sits right of the axes, so the right margin makes room for it (a dragged legend starting at x >= 1 too)."""
    return spec.legend_loc == "outside" or (spec.legend_loc == "manual" and (spec.legend_x or 0.0) >= 1.0)


def _margins_mm(spec: PlotSpec, extra_top_pt: float = 0.0, extra_mm=(0.0, 0.0, 0.0, 0.0),
                legend_labels=None) -> tuple[float, float, float, float]:
    """Fixed (left, bottom, right, top) margins in mm that keep tick labels, titles and legend inside.
    ``extra_mm`` (left, bottom, right, top) and ``legend_labels`` come from ``_extra`` for the P6-P8 kinds."""
    fs = spec.font_size_pt
    tick = _tick_geom(spec)[0] - 3.5  # longer ticks push the labels out
    ty, tx = (a.title_size_pt / fs if a.title_size_pt else 1.0 for a in (spec.y_axis, spec.x_axis))  # axis title sizes / fs
    left = fs * (3.6 + (1.4 * ty if spec.y_axis.title else 0)) + 3 + tick
    bottom = 8 + fs * (1.3 + (1.4 * tx if spec.x_axis.title else 0)) + tick
    if spec.x_axis.tick_rotation:  # a rotated label of ~4 characters hangs lower by about its sine
        bottom += fs * 1.1 * math.sin(math.radians(abs(spec.x_axis.tick_rotation))) * (1.05 if spec.tick_bold else 1.0)
    top = 4 + (max(fs * 1.6, spec.title_size_pt * 1.45) if spec.title else 0) + extra_top_pt
    right = 6.0
    if spec.legend and legend_outside(spec):
        labels = legend_labels if legend_labels is not None else [s.label for s in spec.series if s.visible]
        ls = spec.legend_size_pt or fs
        right += ls * (0.55 * max((len(str(s)) for s in labels), default=1) + 3.5 + max(0.0, (spec.legend_handle_len or 1.6) - 1.6))
    e = extra_mm
    return left * MM / 72 + e[0], bottom * MM / 72 + e[1], right * MM / 72 + e[2], top * MM / 72 + e[3]


def _tick_geom(spec: PlotSpec) -> tuple[float, float]:
    """(length, width) in pt of the major ticks."""
    return (3.5 if spec.tick_length_pt is None else spec.tick_length_pt,
            spec.tick_width_pt if spec.tick_width_pt is not None else spec.axis_width or spec.line_width_pt)


def _axes_rect(box_mm, margins, page_mm):
    """Axes rectangle in figure fractions for a box (x, y_from_top, w, h) minus margins."""
    (x, y, w, h), (l, b, r, t), (W, H) = box_mm, margins, page_mm
    aw, ah = w - l - r, h - b - t
    if aw <= 0 or ah <= 0:
        raise ValueError("graph area too small for its axis labels; enlarge it or reduce the font size")
    return [(x + l) / W, 1 - (y + h - b) / H, aw / W, ah / H]


def build_figure(spec: PlotSpec, table, results: dict[str, dict], tables: dict | None = None) -> Figure:
    """``tables``: table id -> Table of the whole project; a trace graph may draw data sets of several tables."""
    W, H = spec.width_mm, spec.height_mm
    fig = _Figure(figsize=(W / MM, H / MM), facecolor=_background(spec))
    fig.family = spec.font_family
    spec = _resolve(spec, results)
    ax = fig.add_axes(_axes_rect((0, 0, W, H), _margins(spec, table, results), (W, H)))
    draw(ax, spec, table, results, tables)
    return fig


def build_layout(layout: LayoutSpec, graphs: dict) -> Figure:
    """graphs: graph_id -> (PlotSpec, Table, results[, tables]). Panel boxes include their axis labels."""
    W, H = layout.width_mm, layout.height_mm
    fig = _Figure(figsize=(W / MM, H / MM), facecolor="white")
    first = None
    for p in layout.panels:
        spec, table, results, *tables = graphs[p.graph_id]
        first = first or spec
        spec = _resolve(spec, results)
        m = _margins(spec, table, results, layout.label_size_pt * 1.2)
        ax = fig.add_axes(_axes_rect((p.x_mm, p.y_mm, p.w_mm, p.h_mm), m, (W, H)))
        draw(ax, spec, table, results, *tables)
        if p.label:
            fig.text(p.x_mm / W, 1 - p.y_mm / H, p.label, ha="left", va="top", gid=f"panel:{p.graph_id}",
                     fontproperties=FontProperties(family=_families(spec.font_family)[:1], size=layout.label_size_pt, weight="bold"))
    if first is not None:
        fig.family = first.font_family
        fig.set_facecolor(_background(first))
    return fig


def _background(spec: PlotSpec) -> str:
    """Figure face colour: "transparent" keeps the alpha channel in PNG / TIFF and draws nothing in SVG / PDF."""
    return "none" if spec.background == "transparent" else spec.background


# --------------------------------------------------------------------------- helpers
def _lighten(color, keep: float = 0.3):
    return tuple(1 - (1 - c) * keep for c in mpl.colors.to_rgb(color))


def _set_gid(artists, gid: str) -> None:
    for a in artists:
        if a is not None:
            a.set_gid(gid)


def _points(table, ds_id):
    """x (None for category tables), y, row, excluded flag of every non-blank cell."""
    x, y, row = table.points(ds_id, include_excluded=True)
    ds = table.dataset(ds_id)
    ok = ~np.isnan(ds.values)
    if table.x is not None:
        ok &= ~np.isnan(table.x)[:, None]
    return x, y, row, ds.excluded[ok]


def _summary_of(y: np.ndarray) -> dict:
    n = len(y)
    sd = float(np.std(y, ddof=1)) if n > 1 else np.nan
    return {"mean": np.array([y.mean() if n else np.nan]), "sd": np.array([sd]),
            "sem": np.array([sd / math.sqrt(n) if n > 1 else np.nan]), "n": np.array([float(n)])}


def _err(m: dict, error: str) -> np.ndarray:
    if error == "sd":
        return m["sd"]
    if error == "sem":
        return m["sem"]
    if error == "ci95":
        with np.errstate(invalid="ignore"):
            return m["sem"] * _t.ppf(0.975, m["n"] - 1)
    return np.zeros_like(m["mean"])


def _jitter(spec: PlotSpec, idx: int, n: int, amp: float) -> np.ndarray:
    return np.random.default_rng([spec.jitter_seed, idx]).uniform(-amp, amp, n)


def _rgba(color, alpha: float = 1.0):
    """``color`` with ``alpha``; an opaque colour name stays as it is (artifacts then look like the pre-alpha ones)."""
    return color if alpha == 1.0 and isinstance(color, str) else mpl.colors.to_rgba(color, alpha)


def linestyle(name: str, dash: str = ""):
    """Matplotlib linestyle of a ``SeriesStyle.curve_style``; ``dash`` is the custom "on,off,..." pattern."""
    if name in _LS:
        return _LS[name]
    if name != "custom":
        raise ValueError(f"unknown line style {name!r}")
    try:
        seq = [float(v) for v in dash.split(",")]
    except ValueError:
        seq = []
    if not seq or not all(0 < v < 100 for v in seq):
        raise ValueError(f"invalid dash pattern {dash!r}; use positive numbers such as '4,2'")
    return (0, tuple(seq * 2 if len(seq) % 2 else seq))


def _line_kw(s: SeriesStyle, mult: float = 1.0) -> dict:
    """Colour, alpha, width and dashes of a series line (``mult`` scales the automatic width)."""
    return dict(color=_rgba(s.curve_color or s.color, s.curve_alpha), lw=s.curve_width or s.line_width * mult,
                ls=linestyle(s.curve_style, s.curve_dash))


def _marker_kw(s: SeriesStyle, excluded: bool = False, mew_auto: float | None = None) -> dict:
    """Marker shape, size, face and edge. Excluded cells are always hollow."""
    f = s.marker_fill
    if excluded:
        face = "white"
    elif f == "filled":
        face = s.color
    elif f == "hollow":
        face = "none"
    else:
        face = f or (s.color if s.fill else "white")
    return dict(marker=s.marker, ms=s.marker_size, mfc=face if face == "none" else _rgba(face, s.marker_alpha),
                mec=_rgba(s.marker_edge_color or s.color, s.marker_alpha),
                mew=s.marker_edge_width or (0.8 * s.line_width if mew_auto is None else mew_auto))


def _err_kw(s: SeriesStyle) -> dict:
    w = s.err_width or s.line_width
    return dict(ecolor=_rgba(s.err_color or s.color, s.err_alpha), elinewidth=w,
                capsize=s.err_cap if s.err_cap >= 0 else 2.5 + s.line_width, capthick=w)


def _yerr(e, s: SeriesStyle):
    """Error-bar lengths for the series' direction (the other side gets length 0; see ``_hide_caps``)."""
    if s.err_dir == "both":
        return e
    e = np.asarray(e, float)
    gap = np.zeros_like(e)
    return np.array([gap, e] if s.err_dir == "above" else [e, gap])


def _hide_caps(eb, s: SeriesStyle):
    """One-sided error bars: hide the cap of the zero-length side. Returns ``eb``."""
    if s.err_dir != "both" and len(eb[1]) == 2:
        eb[1][0 if s.err_dir == "above" else 1].set_visible(False)
    return eb


def _lohi(m, e, s: SeriesStyle):
    return m - e * (s.err_dir != "above"), m + e * (s.err_dir != "below")


def _fill_kw(s: SeriesStyle, face=None, alpha: float = 1.0, hollow="white") -> dict:
    """Patch style of bars / boxes / violins. ``face`` / ``alpha`` / ``hollow`` are the automatic face colour,
    its alpha and the colour of an unfilled patch."""
    base = s.fill_color or (_lighten(s.color) if face is None else face)
    if not s.fill and not s.fill_color:
        base = hollow
    a = alpha if s.fill_alpha is None else s.fill_alpha
    return dict(facecolor="none" if base == "none" else _rgba(base, a), edgecolor=_rgba(s.edge_color or s.color),
                linewidth=s.edge_width or s.line_width, hatch=s.hatch or None)


def _dots(ax, x, y, ex, s: SeriesStyle, gid: str) -> None:
    for sel, excluded in ((~ex, False), (ex, True)):
        if sel.any():
            (ln,) = ax.plot(x[sel], y[sel], ls="none", zorder=3, **_marker_kw(s, excluded))
            ln.set_gid(gid)


def _mean_err_marks(ax, xc, m, e, s: SeriesStyle, spec: PlotSpec, gid: str, half: float, ext: list) -> None:
    """Horizontal mean line (width 2*half) and error bar at xc."""
    ln = ax.plot([xc - half, xc + half], [m, m], zorder=4, solid_capstyle="butt", **_line_kw(s, 1.2))[0]
    parts = [ln]
    ext.append(np.array([m]))
    if spec.error != "none" and np.isfinite(e):
        eb = _hide_caps(ax.errorbar([xc], [m], yerr=_yerr(np.array([e]), s), fmt="none", zorder=4, **_err_kw(s)), s)
        parts += [*eb[1], *eb[2]]
        ext.append(np.array(_lohi(m, e, s)))
    _set_gid(parts, gid)


def _extent(ext: list) -> tuple[float, float]:
    v = np.concatenate([np.ravel(a) for a in ext]) if ext else np.array([])
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if len(v) else (0.0, 1.0)


# --------------------------------------------------------------------------- draw
def draw(ax, spec: PlotSpec, table, results: dict[str, dict], tables: dict | None = None) -> None:
    """Draw one graph into an existing Axes (whose size the caller has already fixed)."""
    new = spec.kind in _DRAW
    if new:
        want = _TABLE_KIND_OF.get(spec.kind)
        if want and table.kind != want:
            raise ValueError(f"plot kind {spec.kind!r} needs a {want} table, got {table.kind!r}")
    else:
        ok = (spec.kind in ("xy", "column", "grouped") and spec.kind == table.kind.lower()) or \
             (spec.kind == "xy" and table.kind == "MultipleVariables")
        if not ok:
            raise ValueError(f"plot kind {spec.kind!r} does not match table kind {table.kind!r}")
    if spec.kind in STYLES and spec.style not in STYLES[spec.kind]:
        raise ValueError(f"style {spec.style!r} is not valid for kind {spec.kind!r}")
    if spec.error not in ERRORS:
        raise ValueError(f"unknown error type {spec.error!r}")
    if spec.band not in BANDS or spec.band_style not in BAND_STYLES:
        raise ValueError(f"unknown band option {spec.band!r}/{spec.band_style!r}")
    ids = {ds.id for ds in table.datasets}
    styles = [s for s in spec.series if s.visible and s.dataset_id in ids]
    for s in spec.series:
        linestyle(s.curve_style, s.curve_dash)
        if s.err_dir not in ERR_DIRS:
            raise ValueError(f"unknown error bar direction {s.err_dir!r}")
    ext: list = []
    handles = None
    if spec.kind == "trace":
        xpos, handles = _draw_trace(ax, spec, table, tables or {table.id: table})
    elif new:
        xpos, handles = _DRAW[spec.kind](ax, spec, table, results, ext)
    elif spec.kind == "xy":
        _draw_xy(ax, spec, table, styles, results, ext)
        xpos = None
    elif spec.kind == "column":
        xpos = _draw_column(ax, spec, table, styles, ext)
    else:
        xpos = _draw_grouped(ax, spec, table, styles, ext)
    if xpos is not None:
        _category_limits(ax, spec, results, xpos, ext)
        if spec.kind == "contingency" and spec.value == "percent" and spec.style == "stacked" \
                and spec.y_axis.max is None and not spec.y_axis.log:
            ax.set_ylim(ax.get_ylim()[0], 100)
    _style_axes(ax, spec, table, xpos is not None or spec.kind in _HEAT)
    if spec.kind == "xy":
        _draw_interpolated(ax, spec, results, styles)
    elif spec.kind == "survival":
        _at_risk(ax, spec, results)
    elif spec.kind in _HEAT:
        _heat_finish(ax, spec)
    note = _annotation(spec, results)
    if note:
        ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top", gid="annotation",
                fontproperties=_fp(spec))
    for i, t in enumerate(spec.texts):
        ax.text(t["x"], t["y"], t["text"], transform=ax.transAxes, ha="left", va="center", rotation=t.get("rotation", 0.0),
                rotation_mode="anchor", color=t.get("color", "#000000"), gid=f"text:{i}", clip_on=False,
                fontproperties=_fp(spec, t.get("size_pt") or None, "bold" if t.get("bold") else "normal",
                                   "italic" if t.get("italic") else "normal"))
    leg = _legend(ax, spec, styles, handles) if spec.legend else None
    if spec.kind == "trace" and spec.trace.bli.sensors and spec.trace.bli.kinetics_show:
        _draw_kinetics(ax, spec, leg)


def _draw_curve(ax, spec, s: SeriesStyle, curve: dict) -> None:
    """Fitted / regression curve plus optional confidence and prediction bands."""
    cx, cy = np.asarray(curve["x"], float), np.asarray(curve["y"], float)
    for key, alpha in (("ci", 0.18), ("pi", 0.10)):
        lo, hi = curve.get(f"{key}_low"), curve.get(f"{key}_high")
        if lo is None or hi is None or spec.band not in (key, "both"):
            continue
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        col = s.band_color or s.color
        if spec.band_style == "fill":
            a = alpha if s.band_alpha is None else s.band_alpha * alpha / 0.18  # prediction band: 10/18 of it
            parts = [ax.fill_between(cx, lo, hi, facecolor=col, alpha=a, lw=0, zorder=1)]
        else:
            c = _rgba(col, 1.0 if s.band_alpha is None else s.band_alpha)
            parts = [ax.plot(cx, b, color=c, lw=s.line_width * 0.8, ls=":", zorder=2)[0] for b in (lo, hi)]
        _set_gid(parts, f"band:{s.dataset_id}")
    (ln,) = ax.plot(cx, cy, zorder=2, solid_capstyle="round", **_line_kw(s, 1.2))
    ln.set_gid(f"fit:{s.dataset_id}")


def _draw_interpolated(ax, spec, results, styles) -> None:
    """Interpolated unknowns: marker on the curve, dashed drop line to the X axis, optional X interval."""
    if not spec.show_interpolated:
        return
    found = next((r["interpolated"] for r in (results.get(spec.analysis_id), results.get(spec.fit_analysis_id))
                  if r and r.get("interpolated")), None)
    if not found:
        return
    lo, kw = ax.get_ylim()[0], dict(scalex=False, scaley=False)
    for s in styles:
        parts = []
        for p in found.get(s.dataset_id, []):
            x, y = p.get("x"), p.get("y")
            if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
                continue
            parts += ax.plot([x, x], [lo, y], color=s.color, ls="--", lw=s.line_width * 0.8, zorder=2, **kw)
            parts += ax.plot([x], [y], ls="none", marker=s.marker, ms=s.marker_size * 1.2, mfc="white",
                             mec=s.color, mew=1.2 * s.line_width, zorder=5, **kw)
            if p.get("x_ci_low") is not None and p.get("x_ci_high") is not None:
                parts += ax.plot([p["x_ci_low"], p["x_ci_high"]], [lo, lo], color=s.color, lw=2 * s.line_width,
                                 zorder=2, solid_capstyle="butt", clip_on=False, **kw)
        _set_gid(parts, f"interp:{s.dataset_id}")


def _mv_points(table, x_id, y_id):
    if x_id is None:
        raise ValueError("an xy graph on a MultipleVariables table needs x_dataset")
    xd, yd = table.dataset(x_id), table.dataset(y_id)
    x, y = xd.values[:, 0], yd.values[:, 0]
    ok = ~np.isnan(x) & ~np.isnan(y)
    return x[ok], y[ok], (xd.excluded[:, 0] | yd.excluded[:, 0])[ok]


def _draw_xy(ax, spec, table, styles, results, ext) -> None:
    res = results.get(spec.fit_analysis_id) if spec.fit_analysis_id else None
    curves = {k: v["curve"] for k, v in ((res or {}).get("datasets") or {}).items() if v.get("curve")}
    pending = bool(spec.fit_analysis_id) and res is None  # a linked fit still being computed: no connecting lines
    logx = spec.x_axis.log
    if logx:
        ax.set_xscale("log", nonpositive="mask")
    for s in styles:
        gid, curve = f"series:{s.dataset_id}", curves.get(s.dataset_id)
        if table.kind == "MultipleVariables":
            x, y, ex = _mv_points(table, spec.x_dataset, s.dataset_id)
            keep = x > 0 if logx else np.ones(len(x), bool)
            _dots(ax, x[keep], y[keep], ex[keep], s, gid)
            ext.append(y)
        elif spec.style == "replicates" and table.y_format == "replicates":
            x, y, _, ex = _points(table, s.dataset_id)
            keep = x > 0 if logx else np.ones(len(x), bool)
            _dots(ax, x[keep], y[keep], ex[keep], s, gid)
            ext.append(y)
        else:
            m = table.summary(s.dataset_id)
            e = _err(m, spec.error)
            keep = ~np.isnan(m["mean"]) & ~np.isnan(table.x) & ((table.x > 0) if logx else True)
            line = _line_kw(s)
            line["ls"] = line["ls"] if curve is None and not pending else "none"
            mk = _marker_kw(s)
            del mk["marker"]  # given as ``fmt``
            eb = _hide_caps(ax.errorbar(table.x[keep], m["mean"][keep],
                                        yerr=_yerr(e[keep], s) if spec.error != "none" else None,
                                        fmt=s.marker, zorder=3, **line, **mk, **_err_kw(s)), s)
            _set_gid([eb[0], *eb[1], *eb[2]], gid)
            ext.append(m["mean"][keep])
        if curve:
            _draw_curve(ax, spec, s, curve)


def _draw_column(ax, spec, table, styles, ext) -> dict:
    """Returns xpos(ds_id, row) -> x position."""
    pos = {}
    for i, s in enumerate(styles):
        pos[s.dataset_id] = i
        _, y, _, ex = _points(table, s.dataset_id)
        yin, gid = y[~ex], f"series:{s.dataset_id}"
        if len(y) == 0:
            continue
        m = _summary_of(yin)
        mean, e = float(m["mean"][0]), float(_err(m, spec.error)[0])
        jit = _jitter(spec, i, len(y), 0.18 if spec.style == "scatter" else 0.12)
        if spec.style in ("scatter", "bar"):
            if spec.style == "bar" and len(yin):
                bar = ax.bar([i], [mean], width=0.6, zorder=1, **_fill_kw(s))
                _set_gid(bar.patches, gid)
                ext.append(np.array([0.0, mean]))
            if len(yin):
                _mean_err_marks(ax, i, mean, e, s, spec, gid, 0.3 if spec.style == "scatter" else 0.0, ext)
            _dots(ax, i + jit, y, ex, s, gid)
            ext.append(y)
        elif spec.style == "box":
            if len(yin):
                _box(ax, spec, s, i, yin, gid)
                ext.append(yin)
            if spec.show_points:
                _dots(ax, i + jit, y, ex, s, gid)
                ext.append(y)
        else:  # violin
            if len(yin) > 2 and np.ptp(yin) > 0:
                _violin(ax, s, i, yin, gid)
            elif len(yin):
                ax.plot([i - 0.3, i + 0.3], [np.median(yin)] * 2, color=_rgba(s.edge_color or s.color),
                        lw=s.edge_width or s.line_width, gid=gid)
            ext.append(y)
            if spec.show_points:
                _dots(ax, i + jit, y, ex, s, gid)
    n = max(len(styles), 1)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(range(len(styles)), [s.label for s in styles])
    return lambda ds_id, row: pos.get(ds_id)


def _box(ax, spec, s, i, y, gid) -> None:
    tukey = spec.whisker == "tukey"
    ec, ew = _rgba(s.edge_color or s.color), s.edge_width or s.line_width
    bp = ax.boxplot([y], positions=[i], widths=0.5, whis=1.5 if tukey else (0, 100), patch_artist=True,
                    showfliers=tukey and not spec.show_points, manage_ticks=False,
                    boxprops=_fill_kw(s),
                    whiskerprops=dict(color=ec, linewidth=ew),
                    capprops=dict(color=ec, linewidth=ew),
                    medianprops=dict(color=ec, linewidth=ew * 1.4),
                    flierprops=dict(marker=s.marker, mec=s.color, mfc="white", ms=s.marker_size))
    for parts in bp.values():
        _set_gid(parts, gid)


def _violin(ax, s, i, y, gid) -> None:
    vp = ax.violinplot([y], positions=[i], widths=0.8, showextrema=False)
    for body in vp["bodies"]:
        body.set(**_fill_kw(s), alpha=None, zorder=1)  # drops Matplotlib's 0.3 body alpha: the colours carry it
        body.set_gid(gid)
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    for q, ls, half in ((q1, "--", 0.2), (med, "-", 0.3), (q3, "--", 0.2)):
        ax.plot([i - half, i + half], [q, q], color=_rgba(s.edge_color or s.color), lw=s.edge_width or s.line_width,
                ls=ls, zorder=2, gid=gid)


def _draw_grouped(ax, spec, table, styles, ext) -> dict:
    m, n_rows = max(len(styles), 1), table.n_rows
    w = 0.8 / m
    off = {s.dataset_id: (j - (m - 1) / 2) * w for j, s in enumerate(styles)}
    rows = np.arange(n_rows)
    for j, s in enumerate(styles):
        gid, xc = f"series:{s.dataset_id}", rows + off[s.dataset_id]
        sm = table.summary(s.dataset_id)
        e = _err(sm, spec.error)
        keep = ~np.isnan(sm["mean"])
        if spec.style == "bars":
            bar = ax.bar(xc[keep], sm["mean"][keep], width=w * 0.9, zorder=1, **_fill_kw(s))
            _set_gid(bar.patches, gid)
            ext.append(np.r_[0.0, sm["mean"][keep]])
        if spec.error != "none":
            eb = _hide_caps(ax.errorbar(xc[keep], sm["mean"][keep], yerr=_yerr(e[keep], s), fmt="none", zorder=4,
                                        **_err_kw(s)), s)
            _set_gid([*eb[1], *eb[2]], gid)
            lo, hi = _lohi(sm["mean"][keep], e[keep], s)
            ext.append(np.r_[lo, hi])
        if spec.style == "scatter":
            for xk, mk in zip(xc[keep], sm["mean"][keep]):
                _set_gid(ax.plot([xk - w * 0.35, xk + w * 0.35], [mk, mk], zorder=4, solid_capstyle="butt",
                                 **_line_kw(s, 1.2)), gid)
            ext.append(sm["mean"][keep])
        if table.y_format == "replicates":
            _, y, row, ex = _points(table, s.dataset_id)
            _dots(ax, row + off[s.dataset_id] + _jitter(spec, j, len(y), w * 0.3), y, ex, s, gid)
            ext.append(y)
    ax.set_xlim(-0.6, n_rows - 0.4)
    labels = table.row_titles if len(table.row_titles) == n_rows else [str(i + 1) for i in range(n_rows)]
    ax.set_xticks(rows, labels)
    return lambda ds_id, row: None if row is None or ds_id not in off else row + off[ds_id]


# --------------------------------------------------------------------------- limits and brackets
def _category_limits(ax, spec, results, xpos, ext) -> None:
    """Y limits of Column/Grouped graphs, with room for stacked significance brackets."""
    ya = spec.y_axis
    if ya.log:
        ax.set_yscale("log")
        ax.set_ylim(ya.min, ya.max)
        return
    dmin, dmax = _extent(ext)
    barlike = spec.style in ("bar", "bars")
    if ya.min is not None:
        lo = ya.min
    elif dmin >= 0 and (barlike or dmin <= 0.25 * dmax):
        lo = 0.0
    else:
        lo = dmin - 0.05 * ((dmax - dmin) or 1.0)
    items = _bracket_items(spec, results, xpos)
    if not items:
        raw = dmax + 0.05 * ((dmax - lo) or 1.0)
        if ya.max is not None:
            top = ya.max
        elif ya.major_tick:
            top = math.ceil(raw / ya.major_tick - 1e-9) * ya.major_tick
        else:
            top = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, raw)[-1]
        ax.set_ylim(lo, top)
        return
    # stack: smallest free level for each span, narrow spans first
    levels: list[list[tuple[float, float]]] = []
    placed = []
    for it in sorted(items, key=lambda i: (i[1] - i[0], i[0])):
        k = next((k for k, spans in enumerate(levels)
                  if all(it[0] > b1 or it[1] < b0 for b0, b1 in spans)), len(levels))
        if k == len(levels):
            levels.append([])
        levels[k].append((it[0], it[1]))
        placed.append((k, it))
    fs = spec.font_size_pt
    bs = spec.star_size_pt or fs  # bracket label size
    gap, leg, step, txt = 0.5 * fs, 0.4 * fs, 1.6 * max(fs, bs), 1.4 * max(fs, bs)
    fig_h_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    h_b = gap + leg + (len(levels) - 1) * step + txt
    r = min(h_b / fig_h_pt, 0.8)
    top = ya.max if ya.max is not None else (dmax - lo * r) / (1 - r)
    ax.set_ylim(lo, top)
    unit = (top - lo) / fig_h_pt  # data units per pt
    dy = {b.comparison_id: b.dy_pt for b in spec.brackets}
    for k, (x0, x1, label, cid) in placed:
        y = dmax + unit * (gap + leg + k * step + dy.get(cid, 0.0))
        (ln,) = ax.plot([x0, x0, x1, x1], [y - unit * leg, y, y, y - unit * leg], color=spec.bracket_color,
                        lw=spec.bracket_width or spec.line_width_pt * 0.8, clip_on=False, solid_joinstyle="miter")
        ln.set_gid(f"bracket:{cid}")
        star = label.strip("*") == ""
        t = ax.annotate(label, ((x0 + x1) / 2, y), xytext=(0, -0.3 * bs if star else 1), textcoords="offset points",
                        ha="center", va="bottom", annotation_clip=False, fontproperties=_fp(spec, bs),
                        color=spec.bracket_color)
        t.set_gid(f"bracket:{cid}")


def _bracket_items(spec, results, xpos) -> list[tuple[float, float, str, str]]:
    res = results.get(spec.stats_analysis_id) if spec.stats_analysis_id else None
    comps = {c["id"]: c for c in (res or {}).get("comparisons") or []}
    items = []
    for b in spec.brackets:
        c = comps.get(b.comparison_id)
        if c is None or (c.get("summary") == "ns" and not b.show_ns):
            continue
        xa, xb = xpos(c["a"], c.get("row")), xpos(c["b"], c.get("row"))
        if xa is None or xb is None or xa == xb:
            continue
        items.append((min(xa, xb), max(xa, xb), b.label if b.label is not None else c.get("summary", ""), b.comparison_id))
    return items


# --------------------------------------------------------------------------- axes style / legend
def _style_axes(ax, spec: PlotSpec, table, category: bool) -> None:
    lw, ac = spec.axis_width or spec.line_width_pt, spec.axis_color
    tick_len, tick_w = _tick_geom(spec)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name, gid in (("left", "yaxis"), ("bottom", "xaxis")):
        ax.spines[name].set_linewidth(lw)
        ax.spines[name].set_color(ac)
        ax.spines[name].set_gid(gid)
    ax.patch.set_visible(False)
    for axis, a, gid in ((ax.xaxis, spec.x_axis, "xaxis"), (ax.yaxis, spec.y_axis, "yaxis")):
        size = a.font_size or spec.font_size_pt
        ax.tick_params(axis=gid[0], direction="out", length=tick_len, width=tick_w, labelsize=size, color=ac,
                       labelcolor="black", labelfontfamily=_families(spec.font_family), pad=3)
        axis.label.set(text=a.title, color="black", fontproperties=_fp(
            spec, a.title_size_pt or size, "bold" if a.title_bold else "normal", "italic" if a.title_italic else "normal"))
        if a.title_dx or a.title_dy:
            set_offset(axis.label, a.title_dx, a.title_dy)
        if a.major_tick and not (axis is ax.xaxis and category):
            axis.set_major_locator(MultipleLocator(a.major_tick))
        if a.minor_ticks and not a.log and not (axis is ax.xaxis and category):
            axis.set_minor_locator(AutoMinorLocator(2 if a.log10_values == "values" else None))
            ax.tick_params(axis=gid[0], which="minor", direction="out", length=2, width=lw * 0.8, color=ac)
    if not category:
        xa = spec.x_axis
        if xa.log10_values != "off" and not xa.log:
            if not xa.major_tick:  # log10 values: a tick every 1 / 2 / 10 log units
                ax.xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 10] if xa.log10_values == "values" else None,
                                                       integer=True))
            if xa.log10_values == "power":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v + 0.0:g}".replace("-", "−")))
        ax.autoscale_view()
        if xa.min is not None or xa.max is not None:
            ax.set_xlim(xa.min, xa.max)
        if spec.y_axis.log:
            ax.set_yscale("log")
        if spec.y_axis.min is not None or spec.y_axis.max is not None:
            ax.set_ylim(spec.y_axis.min, spec.y_axis.max)
    ax.set_title(spec.title, color="black", gid="title", pad=4, fontproperties=_fp(
        spec, spec.title_size_pt or spec.font_size_pt * 1.1, "bold" if spec.title_bold else "normal",
        "italic" if spec.title_italic else "normal"))
    if spec.title_dx or spec.title_dy:
        set_offset(ax.title, spec.title_dx, spec.title_dy)
    for axis, a, gid in ((ax.xaxis, spec.x_axis, "xaxis"), (ax.yaxis, spec.y_axis, "yaxis")):
        if a.tick_decimals is not None and not a.log and not (axis is ax.xaxis and category):
            n = a.tick_decimals
            if isinstance(n, bool) or not isinstance(n, (int, float)) or n != int(n) or not 0 <= n <= 10:
                raise ValueError(f"{gid[0].upper()} axis: tick decimals must be a whole number from 0 to 10")
            axis.set_major_formatter(FuncFormatter(lambda v, _, n=int(n): f"{v + 0.0:.{n}f}".replace("-", "−")))
        labels = axis.get_ticklabels()
        if spec.tick_bold:
            for t in labels:
                t.set_fontweight("bold")
        if a.tick_rotation:
            for t in labels:
                t.set_rotation(a.tick_rotation)
                if axis is ax.xaxis:
                    t.set(ha="right", va="top", rotation_mode="anchor")
        _set_gid([axis, *labels], gid)
        axis.label.set_gid(gid[0] + "title")  # the axis title is selectable apart from the axis


def _legend(ax, spec: PlotSpec, styles, handles=None) -> None:
    if handles is None:
        handles = []
        for s in styles:
            if spec.kind == "xy":
                line = _line_kw(s)
                line["ls"] = line["ls"] if spec.style == "mean_error" else "none"
                handles.append(Line2D([], [], label=s.label, **line, **_marker_kw(s, mew_auto=1.0)))
            else:
                handles.append(Patch(label=s.label, **_fill_kw(s)))
    if handles:
        if spec.legend_loc == "outside":
            place = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            if spec.kind == "trace" and _has_right(spec):  # clear of the right Y axis
                place["bbox_transform"] = ax.transAxes + ScaledTranslation(_y2_pt(spec) / 72, 0, ax.figure.dpi_scale_trans)
        elif spec.legend_loc == "manual":  # dragged: the upper-left corner sits at (legend_x, legend_y) of the axes
            place = dict(loc="upper left", bbox_to_anchor=(spec.legend_x, spec.legend_y), bbox_transform=ax.transAxes,
                         borderaxespad=0) if spec.legend_x is not None and spec.legend_y is not None else dict(loc="best")
        else:
            place = dict(loc=spec.legend_loc)
        title = bli_legend(spec)[1] if spec.kind == "trace" and spec.trace.bli.sensors else ""
        leg = ax.legend(handles=handles, frameon=spec.legend_frame, fancybox=False, framealpha=1,
                        edgecolor=spec.axis_color, prop=_fp(spec, spec.legend_size_pt), handlelength=spec.legend_handle_len or 1.6,
                        labelspacing=0.5 if spec.legend_spacing is None else spec.legend_spacing, title=title or None, **place)
        if title:
            leg.set_alignment("left")
            leg.get_title().set_fontproperties(_fp(spec, spec.legend_size_pt))
        leg.set_gid("legend")
        by_label = {s.label or s.dataset_id: s.dataset_id for s in spec.series}
        for t in leg.get_texts():  # an entry's label is selectable and editable on the canvas
            if t.get_text() in by_label:
                t.set_gid(f"legend_entry:{by_label[t.get_text()]}")
        return leg


def _draw_kinetics(ax, spec: PlotSpec, leg) -> None:
    """The kinetics block (``BliSpec.kinetics``): left-aligned lines at the legend's left edge, their bottom on the
    axis line, or under the legend when that is longer. Each line hangs under the previous one."""
    size = spec.legend_size_pt or spec.font_size_pt
    rows = [(*t, k["name"]) for k in bli_kinetics(spec.trace.bli) if (t := bli_kinetics_text(k)) is not None]
    height = size * 1.05 * sum(0.8 if italic else 1.0 for _, italic, _ in rows)  # pt, estimated

    def top_left(renderer):
        axb, px = ax.get_window_extent(renderer), renderer.points_to_pixels
        x, y = (leg.get_window_extent(renderer).x0, axb.y0 + px(height)) if leg else (axb.x1 + 0.02 * axb.width, axb.y1)
        if leg:
            y = min(y, leg.get_window_extent(renderer).y0 - px(size))
        return Bbox.from_extents(x, y, x, y)
    prev = None
    for text, italic, name in rows:
        fp = _fp(spec, size * (0.8 if italic else 1.0), style="italic" if italic else "normal")
        first = prev is None
        prev = ax.annotate(text, xy=(0, 0), xycoords=top_left if first else prev, xytext=(0.4 * size if first else 0, 0 if first else -0.1 * size),
                           textcoords="offset points", ha="left", va="top", annotation_clip=False, color="black",
                           fontproperties=fp, gid=f"kinetics:{name}")


# --------------------------------------------------------------------------- P6-P8 kinds
# Every _draw_* below has the signature (ax, spec, table, results, ext) -> (xpos | None, legend handles).
# xpos is set only for the category graphs (contingency, nested), which then get the usual Y limits.
_HEAT = ("heatmap", "correlation_matrix")
_TABLE_KIND_OF = {"contingency": "Contingency", "nested": "Nested", "trace": "XY"}
_RESULT_FIELD = {"survival": "curves", "histogram": "histograms", "qq": "qq", "residuals": "residuals"}


def _need(spec, results, name: str) -> dict:
    """The result this graph reads; a missing, failed or empty one is a user-visible error."""
    res = results.get(spec.analysis_id) if spec.analysis_id else None
    if res is None:
        raise ValueError(f"a {spec.kind} graph needs a result: set analysis_id")
    if res.get("status") == "failed" or not res.get(name):
        raise ValueError(f"the result of {spec.analysis_id!r} has no {name!r} to plot")
    return res


def _value(spec, opts: tuple) -> str:
    v = spec.value or opts[0]
    if v not in opts:
        raise ValueError(f"value {v!r} is not valid for {spec.kind}; use one of {opts}")
    return v


def _keyed(spec, keys) -> list[SeriesStyle]:
    """Visible series styles for result keys (spec.series entry by dataset_id, else a palette default)."""
    by = {s.dataset_id: s for s in spec.series}
    out = [by.get(k) or SeriesStyle(k, str(k), PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])
           for i, k in enumerate(keys)]
    return [s for s in out if s.visible]


def _roc_curves(res: dict | None) -> dict:
    """ROC curves of a result: top-level ``roc`` (one curve, or {name: curve}) or per-data-set ``datasets[ds]["roc"]``."""
    res = res or {}
    r = res.get("roc") or {}
    if "fpr" in r:
        return {"roc": r}
    return r or {k: d["roc"] for k, d in (res.get("datasets") or {}).items() if d.get("roc")}


def _result_keys(spec, results) -> list:
    res = results.get(spec.analysis_id) if spec.analysis_id else None
    if not res:
        return []
    if spec.kind == "roc":
        return list(_roc_curves(res))
    return list(res.get(_RESULT_FIELD.get(spec.kind, "")) or {})


def _annotation(spec: PlotSpec, results: dict) -> str:
    """``spec.annotation``; "auto" on a survival graph is the log-rank P of its Kaplan-Meier result."""
    if spec.annotation != "auto":
        return spec.annotation
    p = (((results.get(spec.analysis_id) or {}).get("comparison") or {}).get("logrank") or {}).get("p")
    if spec.kind != "survival" or p is None:
        return ""
    return "Log-rank P < 0.0001" if p < 1e-4 else f"Log-rank P = {p:.3g}"


def _handle(s: SeriesStyle, how: str):
    lab = s.label or s.dataset_id
    if how == "patch":
        return Patch(label=lab, **_fill_kw(s))
    if how == "line":
        return Line2D([], [], label=lab, **_line_kw(s))
    return Line2D([], [], ls="none", label=lab, **_marker_kw(s, mew_auto=1.0))


def _explained(p: dict) -> np.ndarray:
    """Explained variance in percent (the result may hold fractions)."""
    ex = np.asarray(p["explained"], float)
    return ex * 100 if ex.sum() <= 1.0001 else ex


def _pc_label(p: dict, i: int) -> str:
    names = p.get("pc_names") or []
    name = names[i] if i < len(names) else f"PC{i + 1}"
    ex = _explained(p) if p.get("explained") is not None else []
    return f"{name} ({ex[i]:.1f}%)" if i < len(ex) else name


def _resolve(spec: PlotSpec, results: dict) -> PlotSpec:
    """PCA plots label an empty axis title with the PC name and its explained variance; a normalised trace graph
    titles its left axis ``TraceSpec.norm_title``."""
    if spec.kind == "trace":
        t = spec.trace
        return replace(spec, y_axis=replace(spec.y_axis, title=t.norm_title)) if t.normalize != "none" and t.norm_title else spec
    res = results.get(spec.analysis_id) if spec.analysis_id else None
    if spec.kind not in ("pca_scores", "pca_loadings", "biplot") or not res or not res.get("pca"):
        return spec
    p = res["pca"]
    return replace(spec,
                   x_axis=replace(spec.x_axis, title=spec.x_axis.title or _pc_label(p, spec.pc_x - 1)),
                   y_axis=replace(spec.y_axis, title=spec.y_axis.title or _pc_label(p, spec.pc_y - 1)))


def _dends(spec, dend):
    """(row linkage, column linkage) to draw, either may be None."""
    if not dend or not spec.show_dendrograms:
        return None, None
    return tuple((dend.get(k) or {}).get("linkage") for k in ("rows", "cols"))


def _extra(spec, table, results):
    """Extra margins (left, bottom, right, top; mm) and legend labels the P6-P8 kinds need."""
    fs, mm = spec.font_size_pt, MM / 72
    l = b = r = t = 0.0
    labels = None
    if spec.kind in _HEAT:
        _, rl, cl, dend = _heat_source(spec, table, results)
        dr, dc = _dends(spec, dend)
        width = lambda labs: 0.6 * fs * mm * max((len(str(x)) for x in labs), default=0) if len(labs) <= 60 else 0.0
        l, b = width(rl), width(cl)
        r = HEAT_GAP_MM + (HEAT_DEND_MM + HEAT_GAP_MM if dr is not None else 0) + HEAT_CB_MM + 3.2 * fs * mm
        t = HEAT_GAP_MM + HEAT_DEND_MM if dc is not None else 0.0
    elif spec.kind == "survival" and spec.show_at_risk:
        at = (results.get(spec.analysis_id) or {}).get("at_risk") or {}
        n = sum(k in at for k in _result_keys(spec, results))
        b = (4 + 1.4 * fs * (n + 1)) * mm if n else 0.0
    elif spec.kind in ("pca_scores", "biplot") and spec.color_by and table is not None:
        labels = table.dataset(spec.color_by).categories or []
    elif spec.kind in _RESULT_FIELD or spec.kind == "roc":  # legend entries come from the result keys
        labels = [s.label or s.dataset_id for s in _keyed(spec, _result_keys(spec, results))]
    elif spec.kind == "trace":  # right Y axis beside the axes, event marks under the X tick labels
        r = (_y2_pt(spec) if _has_right(spec) else 1.2 * fs) * mm  # room for the half of the last X tick label
        b = _event_strip_pt(spec) * mm
        if spec.trace.bli.sensors:  # the legend lists concentrations (and metrics), not the fitted curves
            handles, title = bli_legend(spec)
            labels = [_plain(h.get_label()) for h in handles] + [_plain(ln)[:int(0.72 * len(_plain(ln)))] for ln in title.split("\n")]
            if spec.trace.bli.kinetics_show:  # the block hangs under the legend: as wide as its longest line (italic, 0.8 x size)
                need = 0.4 + 0.36 * max((len(_plain(line[0])) for k in bli_kinetics(spec.trace.bli) if (line := bli_kinetics_text(k))), default=0)
                labels.append("x" * math.ceil(max(0.0, (need - 3.5 - max(0.0, (spec.legend_handle_len or 1.6) - 1.6)) / 0.55)))
    return (l, b, r, t), labels


def _margins(spec, table, results, extra_top_pt: float = 0.0):
    if spec.kind not in _DRAW:
        return _margins_mm(spec, extra_top_pt)
    extra, labels = _extra(spec, table, results)
    return _margins_mm(spec, extra_top_pt, extra, labels)


def _draw_survival(ax, spec, table, results, ext):
    res = _need(spec, results, "curves")
    styles, k, tmax = _keyed(spec, res["curves"]), 100.0 if spec.percent else 1.0, 0.0
    for s in styles:
        c = res["curves"][s.dataset_id]
        t = np.asarray(c["time"], float)
        lead = len(t) > 0 and t[0] > 0  # curves start at time 0 with 100 %
        tt = np.r_[0.0, t] if lead else t
        col = lambda v: (np.r_[1.0, np.asarray(v, float)] if lead else np.asarray(v, float)) * k
        if spec.show_ci and c.get("ci_low") is not None and c.get("ci_high") is not None:
            band = ax.fill_between(tt, col(c["ci_low"]), col(c["ci_high"]), step="post",
                                   facecolor=s.band_color or s.color, lw=0, zorder=1,
                                   alpha=0.15 if s.band_alpha is None else s.band_alpha)
            band.set_gid(f"band:{s.dataset_id}")
        line = _line_kw(s, 1.2)
        (ln,) = ax.plot(tt, col(c["surv"]), drawstyle="steps-post", zorder=3, **line)
        ln.set_gid(f"series:{s.dataset_id}")
        ct = np.asarray(c.get("censor_time") or [], float)
        if spec.show_censor and len(ct):
            (cm,) = ax.plot(ct, np.asarray(c["censor_surv"], float) * k, ls="none", marker="|", color=line["color"],
                            ms=s.marker_size * 1.4, mew=s.line_width, zorder=4, clip_on=False)
            cm.set_gid(f"series:{s.dataset_id}")
        tmax = max([tmax, *tt.tolist(), *ct.tolist()])
    ax.set_ylim(0, k)
    ax.set_xlim(0, tmax or 1.0)
    return None, [_handle(s, "line") for s in styles]


def _at_risk(ax, spec, results) -> None:
    """Number-at-risk table below the axes; rows coloured like their curves, columns at the given times."""
    if not spec.show_at_risk:
        return
    res = _need(spec, results, "curves")
    at = res.get("at_risk")
    if not at:
        return
    fs, xmax = spec.font_size_pt, ax.get_xlim()[1]
    base = 8 + fs * (1.3 + (1.4 if spec.x_axis.title else 0))
    kw = dict(textcoords="offset points", va="top", annotation_clip=False, fontproperties=_fp(spec), gid="at_risk")
    ax.annotate("Number at risk", xy=(0, 0), xycoords="axes fraction", xytext=(0, -base), ha="left", **kw)
    for row, s in enumerate((s for s in _keyed(spec, res["curves"]) if s.dataset_id in at), 1):
        for t, n in zip(at["time"], at[s.dataset_id]):
            if 0 <= t <= xmax:
                ax.annotate(f"{n:g}", xy=(t, 0), xycoords=("data", "axes fraction"),
                            xytext=(0, -(base + 1.4 * fs * row)), ha="center", color=s.color, **kw)


def _draw_histogram(ax, spec, table, results, ext):
    res = _need(spec, results, "histograms")
    val = _value(spec, ("counts", "relative", "cumulative"))
    styles = _keyed(spec, res["histograms"])
    m = len(styles)
    for j, s in enumerate(styles):
        h = res["histograms"][s.dataset_id]
        e, y = np.asarray(h["edges"], float), np.asarray(h[val], float)
        w = np.diff(e)
        c, gid = e[:-1] + w / 2, f"series:{s.dataset_id}"
        if spec.style == "points":
            (ln,) = ax.plot(c, y, zorder=3, **_line_kw(s), **_marker_kw(s, mew_auto=1.0))
            ln.set_gid(gid)
        else:
            over = spec.style == "overlay"
            fill = _fill_kw(s, s.color, 0.35, "none") if over else _fill_kw(s, hollow="none")
            bar = ax.bar(c if over else c + (j - (m - 1) / 2) * w / m, y, width=w if over else w / m * 0.9,
                         zorder=1, **fill)
            _set_gid(bar.patches, gid)
    ax.set_ylim(bottom=0)
    return None, [_handle(s, "marker" if spec.style == "points" else "patch") for s in styles]


def _draw_contingency(ax, spec, table, results, ext):
    ids = {d.id for d in table.datasets}
    styles = [s for s in spec.series if s.visible and s.dataset_id in ids]
    percent = _value(spec, ("counts", "percent")) == "percent"
    n, m = table.n_rows, max(len(styles), 1)
    tot = np.sum([np.nan_to_num(d.values[:, 0]) for d in table.datasets], axis=0)  # row total over all outcomes
    rows, bottom, w = np.arange(n), np.zeros(n), 0.8 / m
    stacked = spec.style == "stacked"
    for j, s in enumerate(styles):
        v = np.nan_to_num(table.dataset(s.dataset_id).values[:, 0])
        if percent:
            v = np.divide(100 * v, tot, out=np.zeros(n), where=tot > 0)
        kw = dict(**_fill_kw(s), zorder=1)
        if stacked:
            bar = ax.bar(rows, v, width=0.6, bottom=bottom, **kw)
            bottom = bottom + v
            ext.append(bottom)
        else:
            bar = ax.bar(rows + (j - (m - 1) / 2) * w, v, width=w * 0.9, **kw)
            ext.append(v)
        _set_gid(bar.patches, f"series:{s.dataset_id}")
    ext.append(np.zeros(1))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(rows, table.row_titles if len(table.row_titles) == n else [str(i + 1) for i in range(n)])
    return (lambda ds_id, row: None), [_handle(s, "patch") for s in styles]


def _draw_nested(ax, spec, table, results, ext):
    """All technical replicates, clustered by subject inside each group; group mean +- error of subject means."""
    ids = {d.id for d in table.datasets}
    styles = [s for s in spec.series if s.visible and s.dataset_id in ids]
    pos = {}
    for i, s in enumerate(styles):
        pos[s.dataset_id] = i
        ds, gid = table.dataset(s.dataset_id), f"series:{s.dataset_id}"
        k = ds.values.shape[1]
        offs = np.linspace(-0.25, 0.25, k) if k > 1 else np.zeros(1)
        means = []
        for j in range(k):
            col, ok = ds.values[:, j], ~np.isnan(ds.values[:, j])
            if not ok.any():
                continue
            sub = replace(s, marker=MARKERS[j % len(MARKERS)],
                          color=PALETTE[j % len(PALETTE)] if spec.style == "subject_colors" else s.color)
            _dots(ax, i + offs[j] + _jitter(spec, i * 1000 + j, int(ok.sum()), 0.04), col[ok],
                  ds.excluded[:, j][ok], sub, gid)
            ext.append(col[ok])
            inc = ok & ~ds.excluded[:, j]
            if inc.any():  # subject mean tick
                means.append(col[inc].mean())
                _set_gid(ax.plot([i + offs[j] - 0.07, i + offs[j] + 0.07], [means[-1]] * 2, color=sub.color,
                                 lw=s.line_width * 1.5, zorder=4, solid_capstyle="butt"), gid)
        if means:
            sm = _summary_of(np.array(means))
            _mean_err_marks(ax, i, float(sm["mean"][0]), float(_err(sm, spec.error)[0]),
                            replace(s, color="#000000"), spec, gid, 0.36, ext)
    ax.set_xlim(-0.6, max(len(styles), 1) - 0.4)
    ax.set_xticks(range(len(styles)), [s.label for s in styles])
    return (lambda ds_id, row: pos.get(ds_id)), []


def _draw_qq(ax, spec, table, results, ext):
    res = _need(spec, results, "qq")
    styles, lim = _keyed(spec, res["qq"]), []
    for s in styles:
        q = res["qq"][s.dataset_id]
        th, sa = np.asarray(q["theoretical"], float), np.asarray(q["sample"], float)
        _dots(ax, th, sa, np.zeros(len(th), bool), s, f"series:{s.dataset_id}")
        lim += [th, sa]
    lo, hi = _extent(lim)
    ax.plot([lo, hi], [lo, hi], color="#7F7F7F", ls="--", lw=0.8 * spec.line_width_pt, zorder=1, gid="identity")
    return None, [_handle(s, "marker") for s in styles]


def _draw_residuals(ax, spec, table, results, ext):
    res = _need(spec, results, "residuals")
    styles = _keyed(spec, res["residuals"])
    for s in styles:
        q = res["residuals"][s.dataset_id]
        x, r = np.asarray(q["x"], float), np.asarray(q["r"], float)
        _dots(ax, x, r, np.zeros(len(x), bool), s, f"series:{s.dataset_id}")
    ax.axhline(0, color="#7F7F7F", ls="--", lw=0.8 * spec.line_width_pt, zorder=1, gid="zero_line")
    return None, [_handle(s, "marker") for s in styles]


def _draw_roc(ax, spec, table, results, ext):
    curves = _roc_curves(results.get(spec.analysis_id))
    if not curves:
        raise ValueError(f"the result of {spec.analysis_id!r} has no 'roc' to plot")
    styles = _keyed(spec, curves)
    ax.plot([0, 1], [0, 1], color="#7F7F7F", ls="--", lw=0.8 * spec.line_width_pt, zorder=1, gid="identity")
    for i, s in enumerate(styles):
        c = curves[s.dataset_id]
        (ln,) = ax.plot(c["fpr"], c["tpr"], zorder=3, **_line_kw(s, 1.2))
        ln.set_gid(f"series:{s.dataset_id}")
        if c.get("auc") is not None:
            lab = f"AUC = {c['auc']:.3f}" if len(styles) == 1 else f"{s.label or s.dataset_id}: AUC = {c['auc']:.3f}"
            ax.text(0.97, 0.04 + 0.1 * i, lab, transform=ax.transAxes, ha="right", va="bottom", color=s.color,
                    fontproperties=_fp(spec), gid=f"auc:{s.dataset_id}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return None, [_handle(s, "line") for s in styles]


# ---- PCA
def _pca(spec, results):
    p = _need(spec, results, "pca")["pca"]
    sc, ld = np.asarray(p["scores"], float), np.asarray(p["loadings"], float)
    ix, iy = spec.pc_x - 1, spec.pc_y - 1
    if not (0 <= min(ix, iy) and max(ix, iy) < min(sc.shape[1], ld.shape[1])):
        raise ValueError(f"principal components {spec.pc_x} and {spec.pc_y} are not in the PCA result")
    return p, sc, ld, ix, iy


def _zero_lines(ax) -> None:
    for f in (ax.axhline, ax.axvline):
        f(0, color="#BBBBBB", lw=0.5, ls=":", zorder=0, gid="zero_line")


def _score_points(ax, spec, table, sc, ix, iy) -> list:
    """Scores, one group per category of ``color_by`` (gid ``cluster:<i>``), else one group ``series:scores``."""
    n = len(sc)
    if spec.color_by:
        if table is None:
            raise ValueError("color_by needs the source table")
        ds = table.dataset(spec.color_by)
        if ds.categories is None:
            raise ValueError(f"{ds.title!r} is not a categorical variable")
        if len(ds.values) != n:
            raise ValueError(f"the table has {len(ds.values)} observations but the PCA has {n} scores")
        groups = [(c, ds.values[:, 0] == i, PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)], f"cluster:{i}")
                  for i, c in enumerate(ds.categories)]
    else:
        groups = [(None, np.ones(n, bool), PALETTE[0], MARKERS[0], "series:scores")]
    ms = spec.series[0].marker_size if spec.series else 5.0
    handles = []
    for label, m, col, mk, gid in groups:
        if m.any():
            (ln,) = ax.plot(sc[m, ix], sc[m, iy], ls="none", marker=mk, ms=ms, mfc=col, mec=col, mew=0.8, zorder=3)
            ln.set_gid(gid)
        if label is not None:
            handles.append(Line2D([], [], ls="none", marker=mk, ms=ms, mfc=col, mec=col, label=label))
    return handles


def _arrows(ax, spec, names, lx, ly, scale: float) -> None:
    col = PALETTE[1]
    for nm, x, y in zip(names, lx * scale, ly * scale):
        gid = f"loading:{nm}"
        ax.annotate("", xy=(x, y), xytext=(0, 0), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8 * spec.line_width_pt, shrinkA=0, shrinkB=0)
                    ).set_gid(gid)
        ax.annotate(str(nm), (x, y), xytext=(3, 3), textcoords="offset points", color=col, annotation_clip=False,
                    fontproperties=_fp(spec)).set_gid(gid)
    ax.plot(lx * scale, ly * scale, ls="none")  # lets autoscale see the arrow tips


def _draw_pca_scores(ax, spec, table, results, ext):
    p, sc, ld, ix, iy = _pca(spec, results)
    _zero_lines(ax)
    return None, _score_points(ax, spec, table, sc, ix, iy)


def _draw_pca_loadings(ax, spec, table, results, ext):
    p, sc, ld, ix, iy = _pca(spec, results)
    _zero_lines(ax)
    _arrows(ax, spec, p["variables"], ld[:, ix], ld[:, iy], 1.0)
    ax.margins(0.15)
    return None, []


def _draw_biplot(ax, spec, table, results, ext):
    p, sc, ld, ix, iy = _pca(spec, results)
    _zero_lines(ax)
    handles = _score_points(ax, spec, table, sc, ix, iy)
    smax, lmax = np.abs(sc[:, [ix, iy]]).max(0), np.abs(ld[:, [ix, iy]]).max(0)
    scale = 0.9 * float(np.min(smax / np.where(lmax > 0, lmax, np.inf)))  # arrows span ~ the score cloud
    _arrows(ax, spec, p["variables"], ld[:, ix], ld[:, iy], scale)
    return None, handles


def _draw_scree(ax, spec, table, results, ext):
    p = _need(spec, results, "pca")["pca"]
    y = np.asarray(p["eigenvalues"] if _value(spec, ("eigenvalues", "explained")) == "eigenvalues"
                   else _explained(p), float)
    (ln,) = ax.plot(np.arange(1, len(y) + 1), y, marker="o", ms=5, color=PALETTE[0], lw=spec.line_width_pt, zorder=3)
    ln.set_gid("series:scree")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return None, []


# ---- heatmap / correlation matrix
def _heat_source(spec, table, results):
    """(values, row labels, column labels, dendrogram dict | None) from a result or the table."""
    if spec.kind == "correlation_matrix":
        m = _need(spec, results, "matrix")["matrix"]
        return np.asarray(m["r"], float), list(m["labels"]), list(m["labels"]), None
    if spec.analysis_id:
        res = _need(spec, results, "matrix")
        m = res["matrix"]
        return (np.asarray(m["values"], float), list(m["row_labels"]), list(m["col_labels"]),
                res.get("dendrogram"))
    if table is None or table.kind not in ("MultipleVariables", "Grouped"):
        raise ValueError("a heatmap needs a clustering result or a MultipleVariables / Grouped table")
    if table.kind == "Grouped":
        cols = [(d.title, table.summary(d.id)["mean"]) for d in table.datasets]
    else:
        cols = [(d.title, d.values[:, 0]) for d in table.datasets if d.categories is None]
    vals = np.column_stack([c for _, c in cols])
    if _value(spec, ("raw", "zscore")) == "zscore":
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = (vals - np.nanmean(vals, 0)) / np.nanstd(vals, 0, ddof=1)
    rows = table.row_titles if len(table.row_titles) == len(vals) else [str(i + 1) for i in range(len(vals))]
    return vals, list(rows), [t for t, _ in cols], None


def _order(dend, key: str, n: int) -> np.ndarray:
    node = (dend or {}).get(key)
    o = np.asarray(node["order"] if node else range(n), int)
    if sorted(o.tolist()) != list(range(n)):
        raise ValueError(f"dendrogram order for {key} does not match the matrix ({n} entries)")
    return o


def _dendro(dax, link, order, rows: bool, lw: float, gid: str) -> None:
    """Draw a dendrogram whose leaves sit at the positions of ``order`` (leaf i at its rank, like image pixels)."""
    link = np.asarray(link, float)
    n = len(link) + 1
    if len(order) != n:
        raise ValueError("dendrogram linkage and order disagree")
    pos, hgt, segs = {int(leaf): float(r) for r, leaf in enumerate(order)}, {i: 0.0 for i in range(n)}, []
    for k, (i, j, d, _) in enumerate(link):
        i, j = int(i), int(j)
        pts = [(pos[i], hgt[i]), (pos[i], d), (pos[j], d), (pos[j], hgt[j])]
        segs.append([(h, p) for p, h in pts] if rows else pts)
        pos[n + k], hgt[n + k] = (pos[i] + pos[j]) / 2, float(d)
    lc = LineCollection(segs, colors="black", linewidths=lw)
    dax.add_collection(lc)
    top = float(link[:, 2].max()) * 1.02 or 1.0
    if rows:
        dax.set_xlim(0, top)
        dax.set_ylim(n - 0.5, -0.5)
    else:
        dax.set_xlim(-0.5, n - 0.5)
        dax.set_ylim(0, top)
    dax.axis("off")
    lc.set_gid(gid)


def _draw_heat(ax, spec, table, results, ext):
    vals, rl, cl, dend = _heat_source(spec, table, results)
    ro, co = _order(dend, "rows", vals.shape[0]), _order(dend, "cols", vals.shape[1])
    v = vals[np.ix_(ro, co)]
    corr = spec.kind == "correlation_matrix"
    vmin, vmax = (-1.0, 1.0) if corr else (float(np.nanmin(v)), float(np.nanmax(v)))
    cmap = mpl.colormaps[spec.colormap].with_extremes(bad="#DDDDDD")
    im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    im.set_gid("heatmap")
    fig, pos = ax.figure, ax.get_position()
    aw, ah = pos.width * fig.get_figwidth() * MM, pos.height * fig.get_figheight() * MM
    # labels only when they do not overlap (one text line per row / column)
    fit = lambda n, size_mm: n * spec.font_size_pt * MM / 72 <= size_mm
    ax.set_yticks(range(len(ro)), [rl[i] for i in ro] if fit(len(ro), ah) else [])
    ax.set_xticks(range(len(co)), [cl[i] for i in co] if fit(len(co), aw) else [])
    if corr and spec.show_values and len(ro) <= 12:
        for (i, j), x in np.ndenumerate(v):
            if np.isfinite(x):
                r, g, b = cmap((x - vmin) / ((vmax - vmin) or 1.0))[:3]
                ax.text(j, i, f"{x:.2f}", ha="center", va="center", gid="heatmap", fontproperties=_fp(spec),
                        color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    dr, dc = _dends(spec, dend)
    lw, x = 0.6 * spec.line_width_pt, 1 + HEAT_GAP_MM / aw
    if dr is not None:
        _dendro(ax.inset_axes([x, 0, HEAT_DEND_MM / aw, 1]), dr, ro, True, lw, "dendrogram:rows")
        x += (HEAT_DEND_MM + HEAT_GAP_MM) / aw
    if dc is not None:
        _dendro(ax.inset_axes([0, 1 + HEAT_GAP_MM / ah, 1, HEAT_DEND_MM / ah]), dc, co, False, lw,
                "dendrogram:cols")
    cax = ax.inset_axes([x, 0, HEAT_CB_MM / aw, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(spec.line_width_pt * 0.6)
    cax.tick_params(labelsize=spec.font_size_pt, width=spec.line_width_pt * 0.6, labelfontfamily=_families(spec.font_family))
    _set_gid(cax.findobj(), "colorbar")
    return None, []


def _heat_finish(ax, spec) -> None:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelrotation=90)


# ---- trace graphs: dense instrument curves (chromatography UV / conductivity / gradient) drawn from the raw samples
def trace_envelope(x, y, x0: float, x1: float, n_px: int):
    """The samples of the ascending curve (x, y) that draw the same line at ``n_px`` pixel columns over [x0, x1]:
    per column its first, minimum, maximum and last sample (in X order), plus one sample beyond each end. Returned
    unchanged (cut to the window) while there are at most DECIMATE_FACTOR samples per column."""
    i0 = max(int(np.searchsorted(x, x0, "left")) - 1, 0)
    i1 = min(int(np.searchsorted(x, x1, "right")) + 1, len(x))
    xs, ys = x[i0:i1], y[i0:i1]
    if len(xs) <= DECIMATE_FACTOR * n_px or not x1 > x0:
        return xs, ys
    col = np.floor((xs - x0) * (n_px / (x1 - x0))).astype(np.int64)  # the two outer samples fall outside 0..n_px-1
    starts = np.r_[0, np.flatnonzero(col[1:] != col[:-1]) + 1]
    ends = np.r_[starts[1:], len(xs)]
    n = ends - starts
    where = lambda v: (lambda hit: hit[np.searchsorted(hit, starts)])(np.flatnonzero(ys == np.repeat(v, n)))  # noqa: E731
    keep = np.unique(np.r_[starts, ends - 1, where(np.minimum.reduceat(ys, starts)), where(np.maximum.reduceat(ys, starts))])
    return xs[keep], ys[keep]


class _TraceLine(Line2D):
    """A line that keeps all its samples. Drawn for the screen / preview it shows only the min/max envelope of every
    pixel column when it has more than DECIMATE_FACTOR samples per column (same picture, far fewer vertices); while the
    figure is exported (``_Figure.exact``) or when X is not ascending it draws every sample."""

    def __init__(self, x, y, **kw):
        super().__init__(x, y, **kw)
        self.full = (np.asarray(x, float), np.asarray(y, float))
        self._ascending = bool(np.all(np.diff(self.full[0]) >= 0))
        self._shown = "full"

    def draw(self, renderer):
        ax, fig = self.axes, self.get_figure()
        key = "full"
        if self._ascending and ax is not None and not getattr(fig, "exact", False):
            x0, x1 = sorted(ax.get_xlim())
            key = (max(1, int(ax.bbox.width)), x0, x1)
        if key != self._shown:
            x, y = self.full if key == "full" else trace_envelope(*self.full, x0, x1, key[0])
            self.set_data(x, y)
            self._shown = key
        super().draw(renderer)


def _trace_xy(table, ds_id: str):
    """X and Y of one data set as plotted: the replicate mean (the value itself for one replicate; the first
    sub-column of mean / error layouts), blank and excluded rows dropped."""
    v = np.where(table.dataset(ds_id).excluded, np.nan, table.dataset(ds_id).values)
    if table.y_format == "replicates":
        n = (~np.isnan(v)).sum(1)
        y = np.where(n > 0, np.nansum(v, 1) / np.maximum(n, 1), np.nan)
    else:
        y = v[:, 0]
    ok = np.isfinite(table.x) & np.isfinite(y)
    return table.x[ok], y[ok]


def _window(x, w, what: str):
    if len(w) != 2 or not w[0] < w[1]:
        raise ValueError(f"{what} window: give [x0, x1] with x0 < x1, got {list(w)}")
    m = (x >= w[0]) & (x <= w[1])
    if not m.any():
        raise ValueError(f"the {what} window {list(w)} contains no samples")
    return m


def trace_transform(x, y, tr: TraceSpec, name: str = "") -> np.ndarray:
    """Baseline subtraction, then normalisation, of one trace (see ``TraceSpec``)."""
    if tr.baseline == "min":
        y = y - y.min()
    elif tr.baseline == "window":
        y = y - y[_window(x, tr.baseline_window, "baseline")].mean()
    if tr.normalize != "none":
        top = y.max() if tr.normalize == "max" else y[_window(x, tr.norm_window, "normalisation")].max()
        if not top > 0:
            raise ValueError(f"cannot normalise {name!r}: its maximum is not positive (subtract a baseline first?)")
        y = y / top
    return y


def trace_peaks(x, y, prominence: float, x0: float = -math.inf, x1: float = math.inf):
    """Apex (x, y) of every peak of the curve inside [x0, x1] whose prominence is at least ``prominence`` x the
    Y span of that part (``scipy.signal.find_peaks``; apexes closer than 1 % of the X range count once), strongest first."""
    m = (x >= x0) & (x <= x1)
    xs, ys = x[m], y[m]
    span = float(ys.max() - ys.min()) if len(ys) else 0.0
    if len(xs) < 3 or not span > 0:
        return np.empty(0), np.empty(0)
    dx = float(np.median(np.diff(xs)))
    idx, props = find_peaks(ys, prominence=prominence * span,
                            distance=max(1, int(0.01 * (xs[-1] - xs[0]) / dx)) if dx > 0 else 1)
    order = np.argsort(-props["prominences"])
    return xs[idx[order]], ys[idx[order]]


def _has_right(spec: PlotSpec) -> bool:
    return any(s.visible and s.dataset_id in spec.trace.right for s in spec.series)


def _y2_pt(spec: PlotSpec) -> float:
    """Width (pt) the right Y axis takes beside the axes: tick labels, title and gap."""
    return spec.font_size_pt * (3.6 + (1.4 if spec.trace.y2_axis.title else 0)) + 3


def _shown_events(spec: PlotSpec) -> list[dict]:
    tr = spec.trace
    return [e for e in tr.events if tr.show_events and (not tr.event_run or e.get("run") == tr.event_run)
            and (not tr.event_kinds or e.get("kind") in tr.event_kinds)]


def _event_strip_pt(spec: PlotSpec) -> float:
    """Height (pt) of the event strip under the X tick labels: tick, gap and the longest (rotated) label."""
    ev = _shown_events(spec)
    return 7.0 + 0.62 * 0.8 * spec.font_size_pt * min(max((len(str(e["label"])) for e in ev), default=0), 12) if ev else 0.0


def _style_y2(ax2, spec: PlotSpec) -> None:
    lw, a = spec.axis_width or spec.line_width_pt, spec.trace.y2_axis
    for name in ("left", "top", "bottom"):
        ax2.spines[name].set_visible(False)
    sp = ax2.spines["right"]
    sp.set_visible(True)
    sp.set_linewidth(lw)
    sp.set_color(spec.axis_color)
    sp.set_gid("y2axis")
    ax2.patch.set_visible(False)
    size = a.font_size or spec.font_size_pt
    tick_len, tick_w = _tick_geom(spec)
    ax2.tick_params(axis="y", direction="out", length=tick_len, width=tick_w, labelsize=size, color=spec.axis_color,
                    labelcolor="black", labelfontfamily=_families(spec.font_family), pad=3)
    if spec.tick_bold:
        for t in ax2.yaxis.get_ticklabels():
            t.set_fontweight("bold")
    ax2.yaxis.label.set(text=a.title, color="black", fontproperties=_fp(
        spec, a.title_size_pt or size, "bold" if a.title_bold else "normal", "italic" if a.title_italic else "normal"))
    ax2.yaxis.label.set_gid("y2title")
    if a.major_tick:
        ax2.yaxis.set_major_locator(MultipleLocator(a.major_tick))
    if a.minor_ticks:
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis="y", which="minor", direction="out", length=2, width=lw * 0.8, color=spec.axis_color)


def _trace_ylim(series, x0: float, x1: float, a: AxisSpec, pad_top: float, noise: float = 0.0) -> tuple[float, float]:
    """Y limits of the traces drawn against one axis: the samples inside [x0, x1], a little room above, zero as the
    lower limit when the data start at (about) zero (``noise``: also when the lowest sample is below zero by up to this
    fraction of the highest one); ``a.min`` / ``a.max`` win."""
    v = [y[(x >= x0) & (x <= x1)] for x, y in series]
    v = np.concatenate([seg for seg in v if len(seg)]) if any(len(seg) for seg in v) else np.array([0.0, 1.0])
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or abs(hi) or 1.0
    bottom = 0.0 if -noise * hi <= lo <= 0.12 * hi else lo - 0.03 * span
    bottom, top = (bottom if a.min is None else a.min), (hi + pad_top * span if a.max is None else a.max)
    if not bottom < top:
        raise ValueError("Y axis: the minimum must be below the maximum")
    return bottom, top


def _draw_events(ax, spec: PlotSpec, x0: float, x1: float, w_pt: float) -> None:
    """Fraction / event marks: a small tick and a rotated label per event below the X tick labels; a label that would
    touch an earlier one is left out (marks that are not fractions are placed first), its tick stays."""
    ev = [e for e in _shown_events(spec) if x0 <= e["x"] <= x1]
    if not ev:
        return
    size, fs = 0.8 * spec.font_size_pt, spec.x_axis.font_size or spec.font_size_pt
    off = 6.5 + 1.25 * fs + 1.0  # axis -> bottom of the X tick labels
    shift = ScaledTranslation(0, -off / 72, ax.figure.dpi_scale_trans)
    (tk,) = ax.plot([e["x"] for e in ev], [0.0] * len(ev), ls="none", marker=3, ms=3.0, scalex=False, scaley=False,
                    mew=0.8 * (spec.axis_width or spec.line_width_pt), color=spec.axis_color, clip_on=False,
                    transform=ax.get_xaxis_transform() + shift)
    tk.set_gid("event")
    placed: list[float] = []
    ppu = w_pt / (x1 - x0)
    for e in sorted(ev, key=lambda e: (str(e.get("kind", "")).casefold().startswith("fraction"), e["x"])):
        if any(abs(e["x"] - p) * ppu < 1.3 * size for p in placed):
            continue
        placed.append(e["x"])
        ax.annotate(str(e["label"])[:12], xy=(e["x"], 0), xycoords=("data", "axes fraction"), xytext=(0, -(off + 4)),
                    textcoords="offset points", rotation=90, ha="center", va="top", annotation_clip=False,
                    fontproperties=_fp(spec, size), color="black", gid="event")


def _draw_peak_labels(ax, spec: PlotSpec, data, x0, x1, ylim) -> None:
    """Apex volume above every detected peak of the left-axis traces; overlapping labels are dropped."""
    tr, fig = spec.trace, ax.figure
    pos = ax.get_position()
    w_pt, h_pt = pos.width * fig.get_figwidth() * 72, pos.height * fig.get_figheight() * 72
    boxes: list[tuple] = []
    size = 0.8 * spec.font_size_pt
    left = [d for d in data if not d[3] and len(d[1])]
    for s, x, y, _ in [d for d in left if d[0].dataset_id in tr.peak_series] if tr.peak_series else left[:1]:
        for px, py in zip(*trace_peaks(x, y, tr.peak_prominence, x0, x1)):
            text = f"{px:.{tr.peak_decimals}f}"
            cx, by = (px - x0) / (x1 - x0) * w_pt, (py - ylim[0]) / (ylim[1] - ylim[0]) * h_pt + 2.5
            half = 0.3 * size * len(text) + 1.0
            box = (cx - half, by, cx + half, by + 1.2 * size)
            if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                continue
            boxes.append(box)
            ax.annotate(text, (px, py), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                        fontproperties=_fp(spec, size), color=s.curve_color or s.color, annotation_clip=False,
                        gid=f"peak:{s.dataset_id}")


def _check_bli(b: BliSpec) -> None:
    if b.show not in BLI_SHOW:
        raise ValueError(f"BLI show {b.show!r} is not valid; use one of {BLI_SHOW}")
    if not 0 <= b.fit_alpha <= 1 or not b.fit_width >= 0:
        raise ValueError("BLI fit alpha must be between 0 and 1 and the fit width must not be negative")
    linestyle(b.fit_style, b.fit_dash)
    if b.fit_color:
        mpl.colors.to_rgba(b.fit_color)
    if b.raw_alpha is not None and not 0 <= b.raw_alpha <= 1 or b.raw_width is not None and not b.raw_width >= 0:
        raise ValueError("BLI raw alpha must be between 0 and 1 and the raw width must not be negative")
    if b.raw_style:
        linestyle(b.raw_style, b.raw_dash)
    if b.phase_width is not None and not b.phase_width >= 0:
        raise ValueError("BLI phase line width must not be negative")
    mpl.colors.to_rgba(b.phase_color)
    if not isinstance(b.kinetics, list) or not all(
            isinstance(k, dict) and isinstance(k.get("name"), str) and isinstance(k.get("unit", ""), str)
            and (k.get("value") is None or isinstance(k["value"], (int, float)) and not isinstance(k["value"], bool)
                 and math.isfinite(k["value"])) for k in b.kinetics):
        raise ValueError('BLI kinetics must be a list of {"name": text, "value": number or null, "unit": text}')


def _bli_raw_kw(b: BliSpec, s: SeriesStyle) -> dict:
    """Line keywords of a raw curve: its own series style, with the BLI group overrides (width, alpha, line style)."""
    kw = _line_kw(s)
    if b.raw_width is not None:
        kw["lw"] = b.raw_width
    if b.raw_alpha is not None:
        kw["color"] = _rgba(s.curve_color or s.color, b.raw_alpha)
    if b.raw_style:
        kw["ls"] = linestyle(b.raw_style, b.raw_dash)
    return kw


def _bli_fit_kw(b: BliSpec, base: SeriesStyle) -> dict:
    """Line keywords of a fitted curve; ``base`` = the style of its raw curve (the colour of "" = by concentration)."""
    return dict(color=_rgba(b.fit_color or base.curve_color or base.color, b.fit_alpha), lw=b.fit_width,
                ls=linestyle(b.fit_style, b.fit_dash))


def bli_metric_key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def bli_metric_names(bli: BliSpec) -> list[str]:
    """File keys of the metrics the sensors carry (their "Error ..." companions left out), in file order."""
    return list(dict.fromkeys(k for r in bli.sensors for k in r["metrics"] if not bli_metric_key(k).startswith("error")))


def _bli_metric(rec: dict, name):
    """(value, error or None) of the metric ``name`` of one sensor record (case, spaces and punctuation ignored), or None."""
    get = lambda key: next((v for k, v in rec["metrics"].items() if bli_metric_key(k) == key), None)  # noqa: E731
    v = get(bli_metric_key(name))
    return None if v is None else (v, get("error" + bli_metric_key(name)))


_BLI_SYMBOL = {"kobs": r"$k_\mathrm{obs}$", "koff": r"$k_\mathrm{off}$", "kon": r"$k_\mathrm{on}$", "kd": r"$K_\mathrm{D}$",
               "rmax": r"$R_\mathrm{max}$"}
_BLI_UNIT = {"kobs": r" s$^{-1}$", "koff": r" s$^{-1}$", "kon": r" M$^{-1}$ s$^{-1}$", "kd": " M", "rmax": " nm"}


def _bli_metric_text(name: str, ve) -> str:
    """``koff = (9.65 +- 0.22) x 10^-5 s^-1`` as mathtext: 3 significant digits, the error with 2."""
    v, e = ve
    ex = math.floor(math.log10(abs(v))) if v else 0
    if -2 <= ex <= 2:
        val = f"{v:.3g}" + (f" ± {e:.2g}" if e is not None else "")
    else:
        m = f"({v / 10 ** ex:.3g} \\pm {e / 10 ** ex:.2g})" if e is not None else f"{v / 10 ** ex:.3g}"
        val = f"${m}\\times10^{{{ex}}}$"
    key = bli_metric_key(name)
    return f"{_BLI_SYMBOL.get(key) or str(name).replace('$', '')} = {val}{_BLI_UNIT.get(key, '')}"


_KINETICS = {"kd": ("KD", "M", ("kd",)), "kon": ("kon", "1/Ms", ("kon",)), "koff": ("koff", "1/s", ("koff", "kdis"))}


def bli_kinetics(bli: BliSpec) -> list[dict]:
    """The rows of the kinetics block: the stored ones, or (none stored) KD / kon / koff whose value is the one every
    sensor of the file reports under that name (KD, KOn / kon, KOff / kdis; case and punctuation ignored), else None
    (to be typed in). Nothing is ever computed from the data."""
    if bli.kinetics:
        return [{"name": k["name"], "value": k.get("value"), "unit": k.get("unit", "")} for k in bli.kinetics]
    rows = []
    for name, unit, keys in _KINETICS.values():
        vals = [next((v for k, v in r["metrics"].items() if bli_metric_key(k) in keys), None) for r in bli.sensors]
        shared = vals[0] if vals and vals[0] is not None and all(v == vals[0] for v in vals) else None
        rows.append({"name": name, "value": shared, "unit": unit})
    return rows


def bli_kinetics_text(k: dict) -> tuple[str, bool] | None:
    """(mathtext line, italic) of one kinetics row, None while it has no value. KD (molar) is scaled to pM / nM / µM /
    mM / M with 3 significant digits (a unit other than M prints the number as given); kon and koff use E notation as
    Prism does (``k_on= 6.38E+04 1/Ms``, italic with a subscript); other names print 3 significant digits."""
    v, unit = k["value"], str(k["unit"]).strip().replace("$", r"\$")
    if v is None:
        return None
    key = bli_metric_key(k["name"])
    tail = f" {unit}" if unit else ""
    if key == "kd" and unit in ("", "M", "auto"):
        c = float(f"{abs(v):.3g}")
        u, f = next(((u, f) for u, f in (("M", 1.0), ("mM", 1e-3), ("µM", 1e-6), ("nM", 1e-9)) if c >= f), ("pM", 1e-12))
        return f"KD= {float(f'{v / f:.3g}'):g} {u}", False
    if key in ("kon", "koff"):
        return rf"$\mathit{{k}}_{{\mathit{{{key[1:]}}}}}$= {v:.2E}{tail}", True
    name = str(k["name"]).replace("$", r"\$")
    return f"{name}= {v:{'.2E' if v and not 1e-3 <= abs(v) < 1e4 else '.3g'}}{tail}", False


def bli_legend(spec: PlotSpec) -> tuple[list, str]:
    """(legend handles, legend title) of a BLI graph: one entry per sensor, concentrations from high to low, drawn like
    the raw curve (like the fit when only that is shown); the title lists the chosen metrics that are the same for every
    sensor, the entries append the per-sensor ones."""
    bli, by = spec.trace.bli, {s.dataset_id: s for s in spec.series}
    _check_bli(bli)  # the legend is built before the curves: report bad options in plain words here too
    shared, own = [], []
    for name in bli.metrics:
        per = [_bli_metric(r, name) for r in bli.sensors]
        if all(p is not None for p in per) and all(p == per[0] for p in per):
            shared.append(_bli_metric_text(name, per[0]))
        elif any(p is not None for p in per):
            own.append(name)
    handles = []
    for r in sorted(bli.sensors, key=lambda r: (r["conc"] is None, -(r["conc"] or 0.0))):
        raw, fit = by.get(r["data"]), by.get(r["fit"])
        lab = (raw.label if raw else (fit.label[:-4] if fit.label.endswith(" fit") else fit.label) if fit else "") or r["data"]
        text = ", ".join([lab] + [_bli_metric_text(n, m) for n in own if (m := _bli_metric(r, n)) is not None])
        if raw is not None and raw.visible and bli.show != "fit":
            handles.append(Line2D([], [], label=text, **_bli_raw_kw(bli, raw)))
        elif fit is not None and fit.visible and bli.show != "raw":
            handles.append(Line2D([], [], label=text, **_bli_fit_kw(bli, raw or fit)))
    return handles, "\n".join(shared)


def _plain(text: str) -> str:
    """Mathtext reduced to about the characters it prints (for legend width estimates)."""
    return re.sub(r"[$\\{}^_]|mathrm|mathit", "",text.replace("\\times", "x").replace("\\pm", "±"))


def _draw_trace(ax, spec: PlotSpec, table, tables: dict):
    """Lines drawn straight from the samples of the series' tables; see ``TraceSpec``. Returns (None, legend handles)."""
    tr, fig = spec.trace, ax.figure
    fig.has_traces = True
    for value, valid, what in ((tr.normalize, TRACE_NORMALIZE, "normalize"), (tr.baseline, TRACE_BASELINE, "baseline")):
        if value not in valid:
            raise ValueError(f"trace {what} {value!r} is not valid; use one of {valid}")
    bli = tr.bli if tr.bli.sensors else None
    if bli:
        _check_bli(bli)
    raw_of = {r["data"]: r for r in bli.sensors if r["data"]} if bli else {}
    fit_of = {r["fit"]: r for r in bli.sensors if r["fit"]} if bli else {}
    by_ds = {s.dataset_id: s for s in spec.series}
    data, k = [], 0  # (series style, x, y as drawn, right axis)
    for s in spec.series:
        if not s.visible or (bli and (s.dataset_id in fit_of and bli.show == "raw" or s.dataset_id in raw_of and bli.show == "fit")):
            continue
        tid = tr.sources.get(s.dataset_id, table.id)
        t = tables.get(tid)
        name = s.label or s.dataset_id
        if t is None or all(d.id != s.dataset_id for d in t.datasets):
            raise ValueError(f"trace {name!r}: data set {s.dataset_id!r} is not in table {tid!r} (hide or remove the series)")
        x, y = _trace_xy(t, s.dataset_id)
        right = s.dataset_id in tr.right
        if len(x) and not right and not bli:
            y = trace_transform(x, y, tr, name) + k * tr.stack_offset
            k += 1
        data.append((s, x, y, right))
    off = 0.0
    if bli and bli.rebase:  # association start = 0
        inside = [x for s, x, _, _ in data if len(x) and (s.dataset_id in raw_of or s.dataset_id in fit_of)]
        off = bli.t0 if bli.t0 is not None else min((float(x.min()) for x in inside), default=0.0)
        data = [(s, x - off if s.dataset_id in raw_of or s.dataset_id in fit_of else x, y, r) for s, x, y, r in data]
    xa = spec.x_axis
    xs = [x for _, x, _, _ in data if len(x)]
    lo, hi = (min(x.min() for x in xs), max(x.max() for x in xs)) if xs else (0.0, 1.0)
    x0, x1 = (lo if xa.min is None else xa.min), ((hi if hi > lo else lo + 1.0) if xa.max is None else xa.max)
    if not x0 < x1:
        raise ValueError("X axis: the minimum must be below the maximum")
    ax.set_xlim(x0, x1)
    phase = bli is not None and bli.boundary is not None and x0 < bli.boundary - off < x1
    pad = max(0.12 if tr.peaks else 0.05, 0.14 if phase and bli.phase_labels else 0.0)
    ylim = _trace_ylim([(x, y) for _, x, y, r in data if not r], x0, x1, spec.y_axis, pad, 0.03 if bli else 0.0)
    ax.set_ylim(*ylim)
    ax2 = None
    if any(r for *_, r in data):
        ax2 = ax.twinx()
        ax2.set_ylim(*_trace_ylim([(x, y) for _, x, y, r in data if r], x0, x1, tr.y2_axis, 0.05))
        _style_y2(ax2, spec)
    dots = spec.style == "dots"
    for s, x, y, right in data:
        if s.dataset_id in fit_of:  # fitted curve on top of its raw curve
            line = _TraceLine(x, y, zorder=4, **_bli_fit_kw(bli, by_ds.get(fit_of[s.dataset_id]["data"]) or s))
        elif dots:
            mk = _marker_kw(s, mew_auto=0.0)
            mk["ms"] = max(0.5, 0.3 * s.marker_size)
            line = _TraceLine(x, y, ls="none", zorder=3, **mk)
        else:
            line = _TraceLine(x, y, zorder=3, **(_bli_raw_kw(bli, s) if s.dataset_id in raw_of else _line_kw(s)))
        (ax2 if right else ax).add_line(line)
        line.set_gid(f"series:{s.dataset_id}")
    if tr.stack_offset:
        ax.tick_params(axis="y", labelleft=False)
    pos = ax.get_position()
    if tr.peaks:
        _draw_peak_labels(ax, spec, data, x0, x1, ylim)
    if _shown_events(spec):
        ax.xaxis.labelpad = 4.0 + _event_strip_pt(spec)
        _draw_events(ax, spec, x0, x1, pos.width * fig.get_figwidth() * 72)
    if phase:
        bx = bli.boundary - off
        if bli.phase_line:
            ax.axvline(bx, color=bli.phase_color, ls=(0, (4, 3)), zorder=2, gid="phase_line",
                       lw=0.8 * (spec.axis_width or spec.line_width_pt) if bli.phase_width is None else bli.phase_width)
        if bli.phase_labels:
            for a, b, text in ((x0, bx, "Association"), (bx, x1, "Dissociation")):
                ax.text((a + b) / 2, 0.98, text, transform=ax.get_xaxis_transform(), ha="center", va="top", color="black",
                        fontproperties=_fp(spec, 0.9 * spec.font_size_pt), gid="phase_label")
    return None, bli_legend(spec)[0] if bli else [_handle(s, "marker" if dots else "line") for s, *_ in data]


# draw() calls _draw_trace itself (it needs the other tables); the entry makes "trace" a result-less kind with margins
_DRAW = {"survival": _draw_survival, "histogram": _draw_histogram, "contingency": _draw_contingency,
         "nested": _draw_nested, "qq": _draw_qq, "residuals": _draw_residuals, "roc": _draw_roc,
         "pca_scores": _draw_pca_scores, "pca_loadings": _draw_pca_loadings, "biplot": _draw_biplot,
         "scree": _draw_scree, "heatmap": _draw_heat, "correlation_matrix": _draw_heat, "trace": _draw_trace}


# --------------------------------------------------------------------------- export
def export(fig: Figure, path, *, dpi: int = 600, text_as_paths: bool = False, transparent: bool = False) -> None:
    """Write svg / pdf / png / tif(f) / eps chosen by the extension. ``dpi`` applies to png/tif only.

    Text stays editable: SVG text elements, TrueType (type 42) fonts in PDF/EPS. With
    ``text_as_paths`` SVG text becomes outlines and PDF/EPS use type 3 glyph outlines.
    Trace graphs are exported with every sample (the preview's pixel-column envelope is off; vector formats also
    skip Matplotlib's path simplification, so each sample is a vertex)."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in ("svg", "pdf", "png", "tif", "tiff", "eps"):
        raise ValueError(f"unsupported export format {path.suffix!r}")
    rc = {"svg.fonttype": "path" if text_as_paths else "none",
          "pdf.fonttype": 3 if text_as_paths else 42, "ps.fonttype": 3 if text_as_paths else 42}
    if getattr(fig, "has_traces", False):
        rc["agg.path.chunksize"] = 20000  # very long lines overflow Agg's cell buffer otherwise
        if fmt in ("svg", "pdf", "eps"):
            rc["path.simplify"] = False
    kw: dict = {}
    if fmt in ("png", "tif", "tiff"):
        kw["dpi"] = dpi
    if fmt in ("tif", "tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    was, fig.exact = getattr(fig, "exact", False), True
    try:
        with mpl.rc_context(rc):
            fig.savefig(path, format={"tif": "tiff"}.get(fmt, fmt), transparent=transparent, **kw)
    except PermissionError as exc:
        raise PermissionError(
            f"Windows refused to write {path}. Usual causes: the file is open in another program (close it there), the "
            f"folder is read-only or protected, or this bioapp window was started by another tool with restricted "
            f"rights (start bioapp yourself from the Start menu or Explorer). [{exc.strerror}]") from exc
    finally:
        fig.exact = was
