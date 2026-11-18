"""Instrument trace import (chromatography UV / conductivity / gradient exports) -> traces, events, XY tables, graph.

``read_traces(path)`` reads one CSV / TSV / TXT export (UTF-16 with BOM, UTF-8 with or without BOM, GB18030 fallback;
comma, tab or semicolon separated; decimal point or decimal comma) and returns ``(traces, events)``:

* a trace is a numeric curve with its X (volume / time) and Y arrays, units and run name;
* an event is a text-valued curve entry (fraction label, injection mark, ...): an X position and a label.

Layouts understood: one shared X column followed by several curves, or one X column per curve (x1, y1, x2, y2, ...);
each curve is bound to the nearest X column on its left. 0-3 header rows: run name row, curve name row and unit
row (ÄKTA UNICORN style) or a single row such as ``Volume (mL)``. A trailing empty column or row is ignored.
A text column is an event column when most of its cells are text or its name says so (Fractions, Injection, ...).

``traces_to_graph`` turns chosen traces into one XY table per trace plus a "trace" PlotSpec that carries the events.

BLI sensorgrams: an Octet / ForteBio Data Analysis "Kinetic Data" export is recognised by its header row
``Time<k>,Data<k>[,Fit<k>]`` (one block of columns per sensor). The rows above it are metadata: an optional row of
well ids (``t1A1,t1A1,t1A1,...``) and any number of ``<key>,<value>`` rows repeated per sensor block (Conc1 in M, Start1,
Stop1 in s, KObs, KOff, Error KOff, Start2, KD, ... - every numeric pair is kept). Every sensor gives a raw trace
(``200 nM``) and, when the export has a Fit column, a fit trace (``200 nM fit``), unit_x ``s``, unit_y ``nm``; the value
-99999 (and blanks) mean "missing" and are dropped. Both traces share one ``BliSensor`` record. The association /
dissociation boundary is ``Start2`` when the file has it, otherwise the turning point of the fitted curves (the
sample where a fit is farthest from its first value, then falls back again), the median over the sensors.
"""
from __future__ import annotations

import codecs
import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import DataSet, Table, new_id

X_UNITS = {"ml", "l", "ul", "µl", "min", "s", "h", "cv", "cm"}
KNOWN_UNITS = X_UNITS | {"au", "mau", "mv", "v", "ms/cm", "µs/cm", "s/cm", "%", "°c", "c", "mpa", "bar", "psi", "kpa", "ph",
                         "ml/min", "mm", "nm", "mabs", "abs", "a", "ma", "mg/ml", "mm/ml", "g/l", "ng/ml"}
X_NAME = re.compile(r"^(volume|vol|time|elution|retention|x)\b", re.I)
EVENT_NAME = re.compile(r"fraction|inject|mark|event|waste|\blog\b|comment|note|label", re.I)
_NUM = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_COMMA_NUM = re.compile(r"[+-]?\d+,\d+([eE][+-]?\d+)?")
_PAREN = re.compile(r"^(.*?)\s*[\(\[]([^\)\]]*)[\)\]]\s*$")


class TraceImportError(ValueError):
    """A file the importer cannot read; the message names the file, line and column."""


@dataclass(eq=False)
class BliSensor:
    """Metadata of one BLI sensor: ``conc`` in M (None = the file has none), ``start`` / ``stop`` = Start1 / Stop1 (s),
    ``metrics`` = every other numeric ``key,value`` entry as written in the file, ``boundary`` = association /
    dissociation boundary in s (None = not found)."""
    well: str = ""
    conc: float | None = None
    start: float | None = None
    stop: float | None = None
    metrics: dict = field(default_factory=dict)
    boundary: float | None = None


@dataclass(eq=False)
class Trace:
    name: str
    unit_x: str
    unit_y: str
    x: np.ndarray
    y: np.ndarray
    run: str = ""
    bli: BliSensor | None = None  # BLI sensorgram: the sensor's record (shared by its raw and fit trace)
    role: str = ""  # BLI sensorgram: "data" | "fit"


@dataclass
class Event:
    x: float
    label: str
    kind: str = ""
    run: str = ""


# ---------------------------------------------------------------- reading
def decode_bytes(raw: bytes, name: str = "file") -> str:
    """UTF-16 (BOM, or NUL pattern without one), UTF-8 (BOM or not), else GB18030."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw[3:].decode("utf-8")
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")  # consumes the BOM and takes its byte order
    head = raw[:4096]
    if len(head) >= 4 and head.count(0) > len(head) // 4:  # UTF-16 text without BOM: ASCII characters carry NUL bytes
        return raw.decode("utf-16-le" if head[1::2].count(0) > head[0::2].count(0) else "utf-16-be")
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    raise TraceImportError(f"{name}: cannot decode the text (tried UTF-16, UTF-8, GB18030)")


def sniff_delimiter(lines: list[str]) -> str:
    """Tab, semicolon or comma, in that order, whichever occurs the same number of times on most lines."""
    lines = [ln for ln in lines[:60] if ln.strip()]
    for d in ("\t", ";", ","):
        counts = Counter(n for n in (ln.count(d) for ln in lines) if n > 0)
        if counts and counts.most_common(1)[0][1] >= max(1, 0.7 * len(lines)):
            return d
    raise TraceImportError("no column separator found (expected tab, semicolon or comma)")


def _number(c: str, decimal: str) -> float | None:
    t = c.replace(",", ".") if decimal == "," else c
    return float(t) if _NUM.fullmatch(t) else None


def _detect_decimal(body: list[list[str]], delim: str) -> str:
    if delim == ",":
        return "."
    cells = [c for r in body[:300] for c in r if c]
    comma = sum(bool(_COMMA_NUM.fullmatch(c)) for c in cells)
    dot = sum(bool(_NUM.fullmatch(c)) and "." in c for c in cells)
    return "," if comma and not dot else "."


def _unit_row(row: list[str]) -> bool:
    cells = [c.strip().casefold().replace("μ", "µ") for c in row if c.strip()]
    return bool(cells) and sum(c in KNOWN_UNITS for c in cells) >= 0.6 * len(cells)


def _name_unit(name: str, unit: str) -> tuple[str, str]:
    """``Volume (mL)`` -> (``Volume``, ``mL``) when the unit row gives nothing."""
    m = _PAREN.match(name)
    if m and not unit:
        return m.group(1).strip(), m.group(2).strip()
    return name, unit


def read_traces(path, *, header_rows: int | None = None, decimal: str | None = None) -> tuple[list[Trace], list[Event]]:
    """Read one export file. ``header_rows`` (0-3) and ``decimal`` ('.' or ',') override the detection."""
    path = Path(path)
    text = decode_bytes(path.read_bytes(), path.name)
    lines = text.splitlines()
    if not any(ln.strip() for ln in lines):
        raise TraceImportError(f"{path.name}: the file is empty")
    try:
        delim = sniff_delimiter(lines)
        reader = csv.reader(io.StringIO(text), delimiter=delim)
        rows = [(reader.line_num, [c.strip() for c in r]) for r in reader if any(c.strip() for c in r)]
    except (TraceImportError, csv.Error) as e:
        raise TraceImportError(f"{path.name}: {e}") from None
    width = max(len(r) for _, r in rows)
    rows = [(n, r + [""] * (width - len(r))) for n, r in rows]
    dec = decimal or _detect_decimal([r for _, r in rows], delim)
    if dec not in (".", ","):
        raise ValueError("decimal must be '.' or ','")
    bli = _bli_blocks(rows)
    if bli is not None:
        return _read_bli(path, rows, *bli, dec), []
    h = header_rows
    if h is None:  # header = the leading rows without any number
        h = next((i for i, (_, r) in enumerate(rows) if any(_number(c, dec) is not None for c in r)), len(rows))
    if not 0 <= h <= 3 or h >= len(rows):
        raise TraceImportError(f"{path.name}: found {h} header rows before the first number; expected 0 to 3 "
                               "and at least one data row")
    head, body = [r for _, r in rows[:h]], rows[h:]
    units_row = head[-1] if head and _unit_row(head[-1]) else None
    rest = head[:-1] if units_row is not None else head
    names_row = rest[-1] if rest else [""] * width
    run_row = rest[0] if len(rest) >= 2 else [""] * width
    units_row = units_row or [""] * width

    def column(c: int) -> tuple[str, str]:
        return _name_unit(names_row[c], units_row[c])

    used = [c for c in range(width) if names_row[c] or units_row[c] or any(r[c] for _, r in body)]
    is_x = {}
    for c in used:
        name, unit = column(c)
        is_x[c] = unit.casefold().replace("μ", "µ") in X_UNITS or bool(X_NAME.match(name))
    xs = [c for c in used if is_x[c]] or used[:1]
    if not xs:
        raise TraceImportError(f"{path.name}: no data columns")

    def where(i: int, c: int) -> str:
        name = names_row[c] or units_row[c] or f"#{c + 1}"
        return f"{path.name}: line {body[i][0]}, column {c + 1} ({name!r})"

    def floats(c: int) -> np.ndarray:
        cells = [r[c] for _, r in body]
        try:
            t = [x.replace(",", ".") if dec == "," else x for x in cells]
            return np.array([x if x else "nan" for x in t], dtype=float)
        except ValueError:
            i, bad = next((i, x) for i, x in enumerate(cells) if x and _number(x, dec) is None)
            raise TraceImportError(f"{where(i, c)}: {bad!r} is not a number") from None

    xcache: dict[int, np.ndarray] = {}
    traces: list[Trace] = []
    events: list[Event] = []
    seen: Counter = Counter()
    for c in used:
        if c in xs:
            continue
        left = [x for x in xs if x < c]
        if not left:
            raise TraceImportError(f"{path.name}: column {c + 1} has no volume / time column on its left")
        xc = left[-1]
        name, unit = column(c)
        xname, xunit = column(xc)
        name = name or xname or f"Trace {c + 1}"
        run = run_row[c] or run_row[xc] or next((v for v in run_row if v), "") or path.stem
        cells = [r[c] for _, r in body]
        nonblank = [i for i, v in enumerate(cells) if v]
        if not nonblank:
            continue
        bad = [i for i in nonblank if _number(cells[i], dec) is None]
        if EVENT_NAME.search(name) or len(bad) > 0.5 * len(nonblank):  # text column: marks along X
            if xc not in xcache:
                xcache[xc] = floats(xc)
            for i in nonblank:
                if np.isnan(xcache[xc][i]):
                    raise TraceImportError(f"{where(i, xc)}: label {cells[i]!r} has no position")
                events.append(Event(float(xcache[xc][i]), cells[i], name, run))
            continue
        y = floats(c)  # a few stray non-numbers raise a clear error here
        if xc not in xcache:
            xcache[xc] = floats(xc)
        x = xcache[xc]
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        seen[name] += 1
        traces.append(Trace(name if seen[name] == 1 else f"{name} ({seen[name]})", xunit, unit, x[ok], y[ok], run))
    if not traces and not events:
        raise TraceImportError(f"{path.name}: no numeric trace columns found")
    return traces, events


# ---------------------------------------------------------------- BLI sensorgrams (Octet "Kinetic Data" export)
BLI_MISSING = -99999.0
_BLI_CONC, _BLI_START, _BLI_STOP = (re.compile(p, re.I) for p in (r"conc\d*", r"start1?", r"stop1?"))


def _bli_blocks(rows):
    """(header row index, [(time col, data col, fit col or None)] per sensor) when a row reads Time<k>,Data<k>[,Fit<k>]..."""
    for hi, (_, r) in enumerate(rows):
        blocks, c = [], 0
        while c < len(r):
            m = re.fullmatch(r"time(\d*)", r[c], re.I)
            if m and c + 1 < len(r) and re.fullmatch(f"data{m.group(1)}", r[c + 1], re.I):
                fit = c + 2 if c + 2 < len(r) and re.fullmatch(f"fit{m.group(1)}", r[c + 2], re.I) else None
                blocks.append((c, c + 1, fit))
                c += 3 if fit else 2
            else:
                c += 1
        if blocks:
            return hi, blocks
    return None


def conc_label(c: float) -> str:
    """Concentration in M as text with a unit that keeps the number in 1-999: 2e-7 -> ``200 nM``, 1.25e-8 -> ``12.5 nM``."""
    if c <= 0:
        return "0 nM"
    c = float(f"{c:.3g}")  # three digits first, so 9.9999999e-7 is 1 µM and not 1000 nM
    unit, f = next(((u, f) for u, f in (("M", 1.0), ("mM", 1e-3), ("µM", 1e-6), ("nM", 1e-9)) if c >= f), ("pM", 1e-12))
    return f"{float(f'{c / f:.3g}'):g} {unit}"


def bli_turning_point(x, y) -> float | None:
    """X where the fitted curve ``y`` is farthest from its first value, when it falls back afterwards (association ->
    dissociation); None for a curve that is flat, monotone, or whose maximum is at an end or within 0.1 % of its end value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5:
        return None
    order = np.argsort(x, kind="stable")
    x, dev = x[order], np.abs(y[order] - y[order][0])
    i = int(dev.argmax())
    if not 0 < i < len(x) - 1 or not dev[i] > 0 or dev[i] - dev[-1] < 1e-3 * dev[i]:
        return None
    return float(x[i])


def _read_bli(path: Path, rows, hi: int, blocks, dec: str) -> list[Trace]:
    head, pre, body = rows[hi][1], [r for _, r in rows[:hi]], rows[hi + 1:]
    num = lambda s: _number(s, dec)  # noqa: E731
    well_row = next((r for r in pre[-1:] if all(r[t] and r[t] == r[d] for t, d, _ in blocks)), None)
    meta = [r for r in pre if r is not well_row]
    sensors, cols = [], []
    for t, d, f in blocks:
        metrics = {r[t]: num(r[d]) for r in meta if r[t] and num(r[d]) is not None}
        pick = lambda pat: next((metrics.pop(k) for k in list(metrics) if pat.fullmatch(k)), None)  # noqa: E731
        sensors.append(BliSensor(well_row[t] if well_row else "", pick(_BLI_CONC), pick(_BLI_START), pick(_BLI_STOP), metrics))

    def column(c: int) -> np.ndarray:
        out = np.full(len(body), np.nan)
        for i, (n, r) in enumerate(body):
            if r[c]:
                v = num(r[c])
                if v is None:
                    raise TraceImportError(f"{path.name}: line {n}, column {c + 1} ({head[c]!r}): {r[c]!r} is not a number")
                out[i] = np.nan if v == BLI_MISSING else v
        return out

    names, seen = [], Counter()
    for s, (t, d, f) in zip(sensors, blocks):
        base = conc_label(s.conc) if s.conc is not None else s.well or head[d]
        seen[base] += 1
        names.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    traces: list[Trace] = []
    kept: list[tuple[BliSensor, np.ndarray, np.ndarray]] = []
    for s, (t, d, f), name in zip(sensors, blocks, names):
        x = column(t)
        for role, c, label in (("data", d, name), ("fit", f, f"{name} fit")):
            if c is None:
                continue
            y = column(c)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                traces.append(Trace(label, "s", "nm", x[ok], y[ok], path.stem, s, role))
                if role == "fit":
                    kept.append((s, x[ok], y[ok]))
    if not traces:
        raise TraceImportError(f"{path.name}: no numeric BLI curves found")
    starts = [s.metrics["Start2"] for s in sensors if "Start2" in s.metrics]
    pts = [p for p in (bli_turning_point(x, y) for _, x, y in kept) if p is not None]
    boundary = float(np.median(starts or pts)) if starts or pts else None
    for s in sensors:
        s.boundary = boundary
    return traces


def read_files(paths, errors: list | None = None) -> tuple[list[Trace], list[Event]]:
    """``read_traces`` over several files, traces and events in file order. A run name that an earlier file already
    used becomes ``run (file stem)``. An unreadable file raises ``TraceImportError``, or - when ``errors`` is a
    list - appends its message there and is skipped."""
    traces, events, runs = [], [], set()
    for p in map(Path, paths):
        try:
            if not p.is_file():
                raise TraceImportError(f"{p} does not exist")
            tr, ev = read_traces(p)
        except TraceImportError as e:
            if errors is None:
                raise
            errors.append(str(e))
            continue
        if {o.run for o in (*tr, *ev)} & runs:
            for o in (*tr, *ev):
                o.run = f"{o.run} ({p.stem})"
        runs |= {o.run for o in (*tr, *ev)}
        traces += tr
        events += ev
    return traces, events


# ---------------------------------------------------------------- tables and graph
_PRETTY = {"ml": "mL", "ul": "µL", "µl": "µL", "μl": "µL", "l": "L", "cv": "CV", "au": "AU", "mau": "mAU", "ms/cm": "mS/cm",
           "µs/cm": "µS/cm", "us/cm": "µS/cm"}


def pretty_unit(unit: str) -> str:
    return _PRETTY.get(unit.casefold(), unit)


def _x_title(unit: str) -> str:
    u = unit.casefold()
    return ("Volume" if u in ("ml", "l", "ul", "µl") else "Time" if u in ("min", "s", "h") else
            "Column volumes" if u == "cv" else "X") + (f" ({pretty_unit(unit)})" if unit else "")


def trace_table(t: Trace, title: str | None = None) -> Table:
    """One XY table: X = volume / time, one data set (title ``name (unit)``) with one replicate."""
    ds = DataSet(new_id("ds"), f"{t.name} ({pretty_unit(t.unit_y)})" if t.unit_y else t.name, t.y[:, None])
    return Table(new_id("tbl"), title or (f"{t.run} - {t.name}" if t.run else t.name), "XY", "replicates", [ds], t.x,
                 _x_title(t.unit_x))


def _axis_title(ts: list[Trace]) -> str:
    names = list(dict.fromkeys(t.name for t in ts))
    units = list(dict.fromkeys(t.unit_y for t in ts))
    if len(names) == 1:
        return f"{names[0]} ({pretty_unit(units[0])})" if units[0] else names[0]
    first = {re.split(r"[ _]", n)[0] for n in names}
    lead = first.pop() if len(first) == 1 else "Signal"
    if len(units) == 1:
        return f"{lead} ({pretty_unit(units[0])})" if units[0] else lead
    return " / ".join(dict.fromkeys(f"{t.name} ({pretty_unit(t.unit_y)})" if t.unit_y else t.name for t in ts))


def auto_right(ts: list[Trace]) -> list[bool]:
    """Default axis assignment: traces whose unit differs from the first trace's go to the right axis."""
    return [t.unit_y.casefold() != ts[0].unit_y.casefold() for t in ts] if ts else []


def traces_to_graph(traces: list[Trace], events: list[Event] = (), right: list[bool] | None = None,
                    labels: list[str] | None = None, title: str = "", graph_id: str | None = None):
    """Tables (one per trace, same order) and the "trace" PlotSpec drawing them with the events.
    ``right``: per trace, True = right Y axis (default ``auto_right``). ``labels``: legend labels (default: the trace
    name, or the run name when every run contributes one trace, or ``run - name``)."""
    from . import plot
    if not traces:
        raise ValueError("no traces selected")
    bli = [t for t in traces if t.bli]
    if bli and (len(bli) < len(traces) or len({t.run for t in bli}) > 1):
        raise ValueError("a BLI sensorgram graph takes the curves of one BLI export; do not mix it with other traces")
    right = auto_right(traces) if right is None else list(right)
    if len(right) != len(traces):
        raise ValueError("right needs one flag per trace")
    runs = list(dict.fromkeys(t.run for t in traces))
    if labels is None:
        per_run = Counter(t.run for t in traces)
        labels = [t.name if len(runs) == 1 else t.run if max(per_run.values()) == 1 else f"{t.run} - {t.name}"
                  for t in traces]
    tables = [trace_table(t) for t in traces]
    spec = plot.trace_graph(graph_id or new_id("gr"), [(tb, tb.datasets[0].id, lab, r) for tb, lab, r in zip(tables, labels, right)],
                            events=[{"x": e.x, "label": e.label, "kind": e.kind, "run": e.run} for e in events], title=title)
    spec.x_axis.title = _x_title(traces[0].unit_x)
    spec.y_axis.title = _axis_title([t for t, r in zip(traces, right) if not r] or traces)
    if any(right):
        spec.trace.y2_axis.title = _axis_title([t for t, r in zip(traces, right) if r])
    if len(runs) > 1:  # overlays: fraction marks of several runs would collide
        spec.trace.show_events = False
    if bli:
        _bli_graph(traces, tables, spec)
    return tables, spec


def _bli_graph(traces: list[Trace], tables, spec) -> None:
    """The BLI preset: one sensor record per raw / fit pair, titles, legend and the concentration gradient."""
    from . import plot
    recs, by = [], {}
    for t, tb in zip(traces, tables):
        s = t.bli
        if id(s) not in by:
            by[id(s)] = {"data": "", "fit": "", "conc": s.conc, "well": s.well, "start": s.start, "stop": s.stop,
                         "metrics": dict(s.metrics)}
            recs.append(by[id(s)])
        by[id(s)][t.role] = tb.datasets[0].id
    tmin = min(float(t.x.min()) for t in traces)
    starts = [s.start for s in (t.bli for t in traces) if s.start is not None]
    boundary = traces[0].bli.boundary
    t0 = min(starts) if starts and tmin <= min(starts) and (boundary is None or min(starts) < boundary) else tmin
    spec.trace.bli = plot.BliSpec(sensors=recs, boundary=boundary, t0=t0)
    spec.x_axis.title, spec.y_axis.title = "Time (s)", "Response (nm)"
    spec.legend, spec.width_mm = True, 90.0
    for s in spec.series:
        s.line_width = 0.9
    plot.apply_bli_palette(spec)
