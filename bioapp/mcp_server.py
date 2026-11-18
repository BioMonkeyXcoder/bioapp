"""Local stdio MCP server for bioapp: thin tools over ``api.Session``.

Run:  python -X utf8 -m bioapp.mcp_server
Every tool that changes the project saves the .biofig file immediately (the GUI can reload it).
Env BIOAPP_MCP_ROOTS = ";"-separated folders: when set, files may only be read or written below them.
"""
import functools
import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .api import Session

INSTRUCTIONS = """bioapp: GraphPad-Prism-style tables, analyses and publication graphs stored in a .biofig project file.
Workflow: project_new / project_open -> table_add or table_import -> analysis_run (analysis_kinds lists the kinds and
options; model_search finds curve-fit models) -> graph_create -> graph_preview (look at it!) -> graph_update (patch)
-> layout_create -> graph_export / layout_export. Instrument traces (chromatography UV / conductivity): trace_import
-> graph_preview -> graph_update. BLI sensorgrams (Octet / ForteBio "Kinetic Data" CSV): trace_import -> graph_look
"bold" for the publication look (optional) -> graph_update patch {"trace": {"bli": {...}}} to style ALL raw curves at
once (raw_width / raw_alpha / raw_style) and ALL fitted curves at once (fit_color / fit_width / fit_alpha / fit_style),
set rebase, phase line, palette and the typed-in kinetics block -> graph_preview -> graph_export. Ids (tbl_..., ds_..., an_..., gr_..., lay_...) come from the
tool results; data sets may also be addressed by title or 0-based index. Every change is saved at once;
project_undo restores the previous saved version. Data edits (table_set_values) re-run dependent analyses."""

server = MCPServer("bioapp", instructions=INSTRUCTIONS)
SESSION = Session()
READ_ONLY = ToolAnnotations(read_only_hint=True)


def _roots() -> list[Path]:
    return [Path(r).resolve() for r in os.environ.get("BIOAPP_MCP_ROOTS", "").split(";") if r.strip()]


def _path(p: str) -> Path:
    """Resolve a user path; refuse anything outside BIOAPP_MCP_ROOTS when that is set."""
    path = Path(p).expanduser().resolve()
    roots = _roots()
    if roots and not any(path.is_relative_to(r) for r in roots):
        raise ValueError(f"{p!r} is outside the allowed folders (BIOAPP_MCP_ROOTS): {[str(r) for r in roots]}")
    return path


def tool(mutates: bool = False, needs_project: bool = True):
    """Register a tool: pick up a file saved meanwhile by the GUI, run, save when it mutates, report errors as text."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                if needs_project:
                    if SESSION.path is None:
                        raise ValueError("no project: call project_new or project_open first")
                    SESSION.reload_if_changed()
                out = fn(*args, **kwargs)
                if mutates:
                    SESSION.save()
                return out
            except ToolError:
                raise
            except Exception as e:
                raise ToolError(str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}") from None
        return server.tool(annotations=None if mutates else READ_ONLY)(wrapper)
    return deco


def _expect(item_id: str, section: str) -> None:
    if item_id not in getattr(SESSION.project, section):
        raise ValueError(f"{item_id!r} is not a {section[:-1]} of this project; {section}: {sorted(getattr(SESSION.project, section))}")


# ---------------------------------------------------------------- project

@tool(needs_project=False)
def project_open(path: str) -> dict[str, Any]:
    """Open an existing .biofig project (stale analyses are re-run). Returns the project summary."""
    SESSION.open(_path(path))
    return SESSION.summary()


@tool(needs_project=False)
def project_new(path: str, title: str = "Untitled") -> dict[str, Any]:
    """Create an empty project and save it to `path` (.biofig). Replaces the open project in this server."""
    p = _path(path)
    if p.exists():
        raise ValueError(f"{p} already exists: use project_open, or choose a new file name")
    SESSION.new(title)
    SESSION.save(p)
    return SESSION.summary()


@tool()
def project_save(path: str | None = None) -> dict[str, Any]:
    """Save in place, or 'save as' to `path`. The replaced file is kept as a snapshot (last 10) for project_undo."""
    return {"saved": SESSION.save(_path(path) if path else None)}


@tool()
def project_undo() -> dict[str, Any]:
    """Undo the last saved change: restore the previous saved version of the file (repeatable, up to 10 steps)."""
    SESSION.undo()
    return SESSION.summary()


@tool()
def project_summary() -> dict[str, Any]:
    """List tables (with data set ids), analyses (status, stale), graphs and layouts of the open project."""
    return SESSION.summary()


@tool(mutates=True)
def item_delete(item_id: str) -> dict[str, Any]:
    """Delete a table (with its analyses, output tables and graphs), an analysis, a graph or a layout by id."""
    return SESSION.delete(item_id)


# ---------------------------------------------------------------- tables

@tool(mutates=True)
def table_add(kind: str, datasets: list[dict[str, Any]], title: str | None = None, x: list[float | None] | None = None,
              row_titles: list[str] | None = None, y_format: str = "replicates", x_title: str | None = None) -> dict[str, Any]:
    """Create a data table.
    kind: XY | Column | Grouped | Contingency | Survival | Nested | MultipleVariables.
    datasets: [{title, values, sub_titles?, categories?}]. values: a flat list (one column) or rows of replicates
    [[r1, r2, r3], ...]; null = blank cell. XY: x = one X per row (e.g. log10 doses), rows of values = replicates.
    Column: one flat list per group (lengths may differ). Grouped: values rows = row groups (row_titles), replicate columns.
    y_format: replicates (raw) | mean_sd | mean_sem | mean_sd_n | mean_sem_n (then each row is [mean, error(, n)]).
    Survival: x = times, values 1 = event / 0 = censored in the subject's group column. Contingency: integer counts.
    Returns the table id and data set ids."""
    return SESSION.add_table(kind, title, x, datasets, row_titles, y_format, x_title)


@tool(mutates=True)
def table_import(path: str, kind: str = "XY", options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Import a .csv/.tsv/.txt/.xlsx file as one table of `kind`, or every supported table of a .pzfx file.
    options (CSV/XLSX): header_rows=1, x_col=0, row_title_col, replicates=1, y_format, decimal='.', title, sheet."""
    return SESSION.import_file(_path(path), kind, **(options or {}))


@tool(mutates=True)
def trace_import(paths: list[str], traces: list[str] | None = None, right_axis: list[str] | None = None,
                 title: str | None = None) -> dict[str, Any]:
    """Import chromatography / instrument exports (.csv/.tsv/.txt; UTF-16, UTF-8 or GB18030; tab, semicolon or comma;
    decimal point or comma; 1-3 header rows such as UNICORN's run / curve-name / unit rows; one shared X column or one
    volume column per curve) as one XY table per trace (X = volume or time) and one "trace" graph.
    The graph draws each trace as a thin line straight from the raw samples (no averaging, no fitting); text curves
    (Fractions, Injection) become tick marks with labels along the X axis. Several files overlay several runs.
    paths: files. traces: names of the curves to keep (default all; "run/name" picks one run's curve).
    right_axis: names drawn against the right Y axis (default: curves whose unit differs from the first one's; [] = none).
    Octet / ForteBio "Kinetic Data" exports (metadata rows, then Time1,Data1,Fit1,... columns) are read as BLI
    sensorgrams: per sensor a raw curve ("200 nM") and a fit curve ("200 nM fit") in s / nm, plus the file's metrics;
    the graph gets the BLI preset (gradient by concentration, thin black fits, dashed association / dissociation line).
    Returns graph_id and per trace its table_id, data set id, run and axis (BLI: role data|fit and a "bli" summary with
    boundary and the metric names). Style it with graph_update (patch.trace: x range via x_axis, normalize, baseline,
    stack_offset, peaks, events ...; BLI: patch.trace.bli), look with graph_preview."""
    return SESSION.import_traces([_path(p) for p in paths], traces, right_axis, title)


@tool()
def table_get(table_id: str, max_rows: int = 50) -> dict[str, Any]:
    """Read a table back: x, row titles and per-data-set values (null = blank), excluded cells."""
    return SESSION.get_table(table_id, max_rows)


@tool(mutates=True)
def table_set_values(table_id: str, dataset: str | int | None = None, values: list[Any] | None = None,
                     cells: list[dict[str, Any]] | None = None, start_row: int = 0,
                     x: list[float | None] | None = None, row_titles: list[str] | None = None) -> dict[str, Any]:
    """Edit data and re-run every dependent analysis (transform outputs and chains too).
    dataset: id, title or 0-based index. values: rows written into that data set from start_row (the table grows).
    cells: [{row, sub?, value?, excluded?}] edit single cells (value null = blank; excluded true/false = exclude).
    x / row_titles: written from start_row. Returns the new revision and the status of the dependent analyses."""
    return SESSION.set_values(table_id, dataset, values, cells, start_row, x, row_titles)


# ---------------------------------------------------------------- analyses

@tool()
def analysis_kinds(table_kind: str | None = None) -> dict[str, Any]:
    """List analysis kinds (id, label, category, table kinds, params with defaults and choices),
    optionally only those that accept `table_kind` (XY, Column, ...). Long choice lists are cut: use model_search."""
    return {"kinds": SESSION.analysis_kinds(table_kind)}


@tool()
def model_search(query: str | None = None, limit: int = 25) -> dict[str, Any]:
    """Find curve-fit models for analysis_run(kind='nonlin_fit', params={'model': id}).
    query: words that must all appear in id, name, family or parameter names (e.g. 'inhibitor', 'ec50', 'michaelis').
    Returns id, name, family, parameters, x_scale (log = X is log10 dose) and the equation text."""
    return SESSION.models(query, limit)


@tool(mutates=True)
def analysis_run(table_id: str, kind: str, dataset_ids: list[str | int] | None = None,
                 params: dict[str, Any] | None = None, title: str | None = None) -> dict[str, Any]:
    """Run an analysis now and keep it in the project (it re-runs when the table changes).
    dataset_ids: ids, titles or indices (default: all data sets). params: see analysis_kinds.
    Returns status, messages, result tables (4 significant digits) and, for tests, the comparison ids
    that graph brackets use. Transforms (normalize, ...) create an output table (output_table_id)."""
    return SESSION.run_analysis(table_id, kind, dataset_ids, params, title)


@tool(mutates=True)
def analysis_update(analysis_id: str, params: dict[str, Any] | None = None, dataset_ids: list[str | int] | None = None,
                    title: str | None = None) -> dict[str, Any]:
    """Change an analysis (params are merged into the old ones) and re-run it."""
    return SESSION.update_analysis(analysis_id, params, dataset_ids, title)


@tool()
def analysis_result(analysis_id: str, raw: bool = False) -> dict[str, Any]:
    """Result of an analysis: compact tables, or raw=true for the full structured result (curves, comparisons...)."""
    return SESSION.result(analysis_id, raw)


# ---------------------------------------------------------------- graphs

@tool()
def graph_choices(table_id: str) -> dict[str, Any]:
    """The graphs a table offers: label, kind and analysis_id for graph_create."""
    return {"choices": SESSION.graph_choices(table_id)}


@tool(mutates=True)
def graph_create(table_id: str, kind: str | None = None, analysis_id: str | None = None) -> dict[str, Any]:
    """Make a graph like the GUI's New graph. Default kind follows the table (xy, column, grouped, ...);
    an XY graph draws the table's latest fit, a Column/Grouped graph the brackets of its latest comparison test.
    Result-driven kinds (survival, pca_scores, heatmap, qq, roc, ...) need analysis_id; see graph_choices.
    kind="trace" on an XY table draws its data sets as thin lines from the raw samples (for several tables use trace_import).
    Returns the id and the full spec."""
    return SESSION.create_graph(table_id, kind, analysis_id)


@tool(mutates=True)
def graph_from_result(analysis_id: str) -> dict[str, Any]:
    """Make the natural graph(s) of an analysis result (curves, survival, PCA, ...). Returns graph ids ([] if none)."""
    return {"graph_ids": SESSION.graphs_for_result(analysis_id)}


@tool()
def graph_get(graph_id: str) -> dict[str, Any]:
    """The graph's full spec (PlotSpec as JSON): size, fonts, axes, series styles, brackets, legend."""
    return SESSION.get_graph(graph_id)


@tool(mutates=True)
def graph_update(graph_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a patch into the graph spec; unknown keys are rejected with the list of valid keys.
    Top-level scalars replace (title, width_mm, height_mm, font_size_pt, legend, style, error, band, ...);
    x_axis / y_axis merge key by key ({"min": 0, "title": "..."});
    series: [{"dataset_id": id-or-label, "color": "#D55E00", "marker": "s", ...}];
    brackets: [{"comparison_id": id, "label": "P < 0.05"}] or [{"comparison_id": id, "remove": true}];
    trace (trace graphs): {"normalize": "max", "baseline": "min", "stack_offset": 0.5, "peaks": true, "right": [series],
    "y2_axis": {...}, "event_run": "Run 1", "events": [...]}.
    BLI sensorgrams: patch {"trace": {"bli": {...}}}, fields:
      what is drawn: "show": "both|raw|fit", "rebase": true (time 0 = association start), "boundary": 570 (s),
        "phase_line": true, "phase_labels": true, "phase_width": 2 (pt, null = auto), "phase_color": "#8C8C8C";
      raw curves as ONE group (null / "" = each curve keeps its own series setting; colours stay per concentration):
        "raw_width": 4 (pt), "raw_alpha": 0.5 (0-1), "raw_style": "solid|dashed|dotted|dashdot|custom", "raw_dash": "6,3" (on,off pt with custom);
        "palette": "Blues|Greens|Oranges|Purples|Reds|Greys|viridis_r|magma_r|Vivid blues" recolours them by
        concentration (highest darkest); one curve's colour: series [{"dataset_id": "200 nM", "color": ...}];
      fitted curves as ONE group, drawn on top, not in the legend: "fit_color": "#707070" ("" = colour of its
        concentration), "fit_width": 1 (pt), "fit_alpha": 1, "fit_style": "solid|dashed|dotted|dashdot|custom", "fit_dash": "4,2";
      legend text: "metrics": ["KOff"] (file metrics; shared value in the legend title, per-sensor after each entry);
      kinetics block under the legend: "kinetics_show": true, "kinetics": [{"name": "KD", "value": 6.68e-9, "unit": "M"},
        {"name": "kon", "value": 6.38e4, "unit": "1/Ms"}, {"name": "koff", "value": 4.26e-4, "unit": "1/s"}] (typed-in
        values: KD in molar, shown as nM etc., kon / koff in E notation; [] = the file's own KD / kon / koff metrics;
        never computed from the data, an Octet export usually only has koff / kobs).
    Look fields (any graph): tick_bold, tick_length_pt, tick_width_pt, axis_width, legend_handle_len, legend_size_pt,
    legend_spacing, x_axis / y_axis {"tick_rotation": 45, "tick_decimals": 1, "major_tick": 150, "title_size_pt": 14};
    graph_look applies them as a preset."""
    return SESSION.update_graph(graph_id, patch)


@tool(mutates=True)
def graph_palette(graph_id: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Recolour all series of a graph with a named palette; without graph_id / name only lists the palettes."""
    if graph_id is None or name is None:
        return {"palettes": SESSION.palettes()}
    return SESSION.apply_palette(graph_id, name)


@tool(mutates=True)
def graph_look(graph_id: str, name: str = "bold") -> dict[str, Any]:
    """Apply a look preset to a graph: "bold" = the heavy publication look (bold 12 pt tick labels and axis titles, thick
    axes with long outward ticks, round X ticks at 45 degrees, one-decimal Y ticks, frameless legend with long handles,
    square plot area; BLI sensorgrams also get thick opaque "Vivid blues" raw curves, grey fits on top, a thick dashed
    phase line and the kinetics block), "default" = those fields back to a new graph's values. Axis limits, titles and
    data stay; refine afterwards with graph_update (tick_rotation, tick_decimals, kinetics values, sizes ...)."""
    return SESSION.apply_look(graph_id, name)


@tool()
def graph_preview(graph_id: str, dpi: int = 150) -> Image:
    """Render a graph as a PNG you can look at (dpi 50-400)."""
    _expect(graph_id, "graphs")
    return Image(data=SESSION.preview(graph_id, max(50, min(400, dpi))), format="png")


@tool()
def graph_export(graph_id: str, path: str, dpi: int = 600, text_as_paths: bool = False) -> dict[str, Any]:
    """Export a graph to svg / pdf / png / tif / eps (by file extension). dpi applies to png/tif;
    text_as_paths turns SVG/PDF text into outlines (default: editable text)."""
    _expect(graph_id, "graphs")
    return {"exported": SESSION.export(graph_id, _path(path), dpi, text_as_paths)}


# ---------------------------------------------------------------- layouts

@tool(mutates=True)
def layout_create(graph_ids: list[str], rows: int = 1, cols: int | None = None, width_mm: float = 180.0,
                  height_mm: float | None = None, labels: bool = True, title: str = "") -> dict[str, Any]:
    """Arrange graphs on a page in a rows x cols grid (left to right, top to bottom), panel labels A, B, ...
    height_mm defaults to the aspect ratio of the first graph."""
    return SESSION.create_layout(graph_ids, rows, cols, width_mm, height_mm, labels, title)


@tool()
def layout_preview(layout_id: str, dpi: int = 150) -> Image:
    """Render a layout as a PNG you can look at (dpi 50-400)."""
    _expect(layout_id, "layouts")
    return Image(data=SESSION.preview(layout_id, max(50, min(400, dpi))), format="png")


@tool()
def layout_export(layout_id: str, path: str, dpi: int = 600, text_as_paths: bool = False) -> dict[str, Any]:
    """Export a layout to svg / pdf / png / tif / eps (by file extension)."""
    _expect(layout_id, "layouts")
    return {"exported": SESSION.export(layout_id, _path(path), dpi, text_as_paths)}


def main() -> None:
    import logging
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF export would log font-subsetting chatter to stderr
    server.run("stdio")


if __name__ == "__main__":
    main()
