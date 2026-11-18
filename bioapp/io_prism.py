"""Prism XML files: ``.pzfx`` and the Prism sample ``.xml`` files (GraphPadPrismFile inside an XSL wrapper).

Format facts (verified against the Prism 10.4.1 sample files):
- ``.pzfx`` has no XML namespace; the ``.xml`` samples use http://graphpad.com/prism/Prism.htm. Tags are matched by local name.
- A table is ``Table`` with attributes TableType (XY / OneWay = Column / TwoWay = Grouped), XFormat, YFormat
  (``replicates`` + Replicates=n, or SDN for mean/SD/N; absent for OneWay), ExtTableType (Nested, MultipleVariables).
- Children: Title, [RowTitlesColumn], [XColumn], YColumn*; each column holds Subcolumn elements of ``d`` cells,
  ``<d/>`` is blank, ``<d Excluded="1">v</d>`` is an excluded value (attribute documented in the sample stylesheet).
- Contingency / Survival are plain TableType values (no YFormat; Survival has an XColumn of times and 1/0 codes in the
  group's YColumn). Nested is TwoWay + ExtTableType="Nested": Replicates subcolumns per group, titles in
  ``SubColumnTitles OwnSet="1"`` (one Subcolumn per subcolumn index, one ``d`` per data set). MultipleVariables is OneWay +
  ExtTableType: one YColumn per variable with MVType (Continues / Categorical / Label, optionally prefixed ``Manual|``);
  categorical columns carry a ``CategoryDictionary`` (``d id=code``) and cells ``<d CellType="Text" num=code>label</d>``;
  the first Label column holds observation names (read as row titles).
- Columns are padded to the table's row count; shorter columns simply have fewer ``d`` elements.
- The sample .xml files store decimals with a comma (locale of the author); such cells are read as decimal points.
The zipped ``.prism`` format is not supported.
"""
from __future__ import annotations

import datetime as dt
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .data import DataSet, Table, Y_FORMATS, new_id
from .io_tabular import fmt_num
from .project import atomic_write_bytes

_KIND = {"XY": "XY", "OneWay": "Column", "TwoWay": "Grouped", "Contingency": "Contingency", "Survival": "Survival"}
_EXT = ("Nested", "MultipleVariables")  # ExtTableType values that map to a kind of their own
_TYPE_OUT = {"XY": "XY", "Column": "OneWay", "Grouped": "TwoWay", "Contingency": "Contingency", "Survival": "Survival",
             "Nested": "TwoWay", "MultipleVariables": "OneWay"}
_XKINDS = ("XY", "Survival")  # kinds with an X column of numbers
_FMT_IN = {"SD": "mean_sd", "SDN": "mean_sd_n", "SE": "mean_sem", "SEM": "mean_sem", "SEN": "mean_sem_n"}
_FMT_OUT = {"mean_sd": "SD", "mean_sd_n": "SDN", "mean_sem": "SE", "mean_sem_n": "SEN"}
_NUM = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


class _Skip(Exception):
    """A table that cannot be represented; the message is the reason."""


def _loc(el) -> str:
    return el.tag.rpartition("}")[2]


def _kids(el, name: str) -> list:
    return [c for c in el if _loc(c) == name]


def _text(el) -> str:
    return "".join(el.itertext())


def _cell(d, state: dict) -> tuple[float, bool]:
    if len(d):
        raise _Skip("formatted text cell")
    s = (d.text or "").strip()
    if not s:
        return np.nan, False
    if not _NUM.fullmatch(s):
        t = s.replace(",", ".")
        if s.count(",") != 1 or "." in s or not _NUM.fullmatch(t):
            raise _Skip(f"text cell {s[:20]!r}")
        state["comma"] = True
        s = t
    excluded = d.get("Excluded", "0") not in ("", "0")
    state["excluded"] += excluded
    return float(s), excluded


def _subcolumns(col, state: dict) -> list[list[tuple[float, bool]]]:
    return [[_cell(d, state) for d in _kids(sub, "d")] for sub in _kids(col, "Subcolumn")]


def _title(el, default: str) -> str:
    t = _kids(el, "Title")
    return _text(t[0]) if t else default


def _row_titles(t) -> list[str]:
    rt = _kids(t, "RowTitlesColumn")
    return [_text(d) for sub in _kids(rt[0], "Subcolumn")[:1] for d in _kids(sub, "d")] if rt else []


def _padded(titles: list[str], n: int) -> list[str]:
    return titles + [""] * (n - len(titles)) if titles else []


def _categorical(y, cells, state: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Codes (NaN = blank), excluded flags and labels of a categorical column. The dictionary ids give the level
    order and are the codes; cells without a known id are matched by label, unseen labels are appended."""
    cd = _kids(y, "CategoryDictionary")
    entries = sorted((int(d.get("id")), _text(d)) for d in _kids(cd[0], "d")) if cd else []
    labels = [lab for _, lab in entries]
    code_of_id = {i: k for k, (i, _) in enumerate(entries)}
    codes, excl = np.full(len(cells), np.nan), np.zeros(len(cells), dtype=bool)
    for r, d in enumerate(cells):
        s, num = _text(d), d.get("num")
        if num is not None and int(num) in code_of_id:
            codes[r] = code_of_id[int(num)]
        elif s.strip():
            if s not in labels:
                labels.append(s)
            codes[r] = labels.index(s)
        else:
            continue
        excl[r] = d.get("Excluded", "0") not in ("", "0")
    state["excluded"] += int(excl.sum())
    return codes, excl, labels


def _build_mv(t, state: dict) -> Table:
    titles, cols = _row_titles(t), []
    for y in _kids(t, "YColumn"):
        subs = _kids(y, "Subcolumn")
        if len(subs) != 1:
            raise _Skip(f"variable {_title(y, '')!r} has {len(subs)} subcolumns, expected 1")
        cols.append((y, _kids(subs[0], "d")))
    sets = []
    for y, cells in cols:
        role = (y.get("MVType") or "").rpartition("|")[2]
        if role == "Label" and not any(titles):  # the first label column holds the observation names
            titles = [d.get("UserText") or _text(d) for d in cells]
            continue
        data = None
        if role != "Categorical" and not _kids(y, "CategoryDictionary"):
            try:
                trial = dict(state)
                data = [_cell(d, trial) for d in cells]
                state.update(trial)
            except _Skip:
                if role == "Continues":
                    raise
        if data is not None:
            values, excl, cats = np.array([c[0] for c in data]), np.array([c[1] for c in data]), None
        else:
            values, excl, cats = _categorical(y, cells, state)
        sets.append((_title(y, ""), values, excl.astype(bool), cats))
    n = max([len(titles)] + [len(s[1]) for s in sets])
    out = []
    for title, values, excl, cats in sets:
        v, e = np.full((n, 1), np.nan), np.zeros((n, 1), dtype=bool)
        v[:len(values), 0], e[:len(values), 0] = values, excl
        out.append(DataSet(new_id("ds"), title, v, e, categories=cats))
    return Table(new_id("tbl"), _title(t, "Untitled"), "MultipleVariables", "replicates", out,
                 row_titles=_padded(titles, n))


def _sub_titles(t, sets: list[DataSet], nsub: int) -> None:
    """Nested subcolumn titles: ``SubColumnTitles`` holds one Subcolumn per subcolumn index; with OwnSet each of its
    ``d`` belongs to one data set (in order), otherwise the first ``d`` is shared by all data sets."""
    st = _kids(t, "SubColumnTitles")
    if not st:
        return
    grid = [[_text(d).strip() if len(d) else _text(d) for d in _kids(s, "d")] for s in _kids(st[0], "Subcolumn")]
    own = st[0].get("OwnSet") == "1"
    for j, ds in enumerate(sets):
        names = [(g[j] if own else g[0]) if len(g) > (j if own else 0) else "" for g in grid[:nsub]]
        names += [""] * (nsub - len(names))
        if any(names):
            ds.sub_titles = names


def _build(t, state: dict) -> Table:
    a = t.attrib
    ext = a.get("ExtTableType")
    if ext:
        if ext not in _EXT:
            raise _Skip(f"{ext} table is not supported")
        kind = ext
    elif a.get("TableType") in _KIND:
        kind = _KIND[a["TableType"]]
    else:
        raise _Skip(f"{a.get('TableType')} tables are not supported")
    if kind == "MultipleVariables":
        return _build_mv(t, state)
    yfmt = a.get("YFormat") or "replicates"
    if yfmt == "replicates":
        y_format, nsub = "replicates", int(a.get("Replicates") or 1)
    elif yfmt in _FMT_IN:
        y_format = _FMT_IN[yfmt]
        nsub = len(Y_FORMATS[y_format])
    else:
        raise _Skip(f"Y format {yfmt!r} is not supported")
    ycols = _kids(t, "YColumn")
    xs = None
    if kind in _XKINDS:
        if a.get("XFormat") != "numbers":
            raise _Skip(f"X format {a.get('XFormat')!r} (dates / elapsed times) is not supported")
        xcols = _kids(t, "XColumn")
        if len(xcols) != 1:
            raise _Skip(f"{kind} table without an X column")
        xsubs = _subcolumns(xcols[0], state)
        if len(xsubs) != 1:
            raise _Skip("X column with error values is not supported")
        xs = [v for v, _ in xsubs[0]]
    ysubs = []
    for y in ycols:
        subs = _subcolumns(y, state)
        if len(subs) != nsub:
            raise _Skip(f"data set {_title(y, '')!r} has {len(subs)} subcolumns, expected {nsub}")
        ysubs.append(subs)
    titles = _row_titles(t)
    n = max([len(xs or [])] + [len(s) for subs in ysubs for s in subs] + [len(titles)])
    sets = []
    for y, subs in zip(ycols, ysubs):
        values = np.full((n, nsub), np.nan)
        excl = np.zeros((n, nsub), dtype=bool)
        for j, sub in enumerate(subs):
            for i, (v, e) in enumerate(sub):
                values[i, j], excl[i, j] = v, e
        sets.append(DataSet(new_id("ds"), _title(y, ""), values, excl))
    x = x_title = None
    if kind in _XKINDS:
        x = np.full(n, np.nan)
        x[:len(xs)] = xs
        x_title = _title(_kids(t, "XColumn")[0], "X")
    if kind == "Nested":
        _sub_titles(t, sets, nsub)
    return Table(new_id("tbl"), _title(t, "Untitled"), kind, y_format, sets, x, "X" if x_title is None else x_title,
                 _padded(titles, n))


def _entry(t) -> dict:
    a = t.attrib
    e = {"title": _title(t, a.get("ID", "Untitled")),
         "prism_type": a.get("TableType", "") + (f"/{a['ExtTableType']}" if a.get("ExtTableType") else ""),
         "y_format": a.get("YFormat", ""),
         "rows": max((len(_kids(s, "d")) for s in t.iter() if _loc(s) == "Subcolumn"), default=0),
         "sets": len(_kids(t, "YColumn")), "table": None, "reason": None, "notes": []}
    if _loc(t) != "Table":
        e["reason"] = f"{_loc(t)} is not supported"
        return e
    state = {"comma": False, "excluded": 0}
    try:
        e["table"] = _build(t, state)
    except _Skip as s:
        e["reason"] = str(s)
    except ValueError as s:  # bad attribute or a layout the data model rejects
        e["reason"] = f"malformed table ({s})"
    if e["table"] is not None:
        if state["comma"]:
            e["notes"].append("decimal commas read as decimal points")
        if state["excluded"]:
            e["notes"].append(f"{state['excluded']} excluded values")
    return e


def scan_pzfx(path) -> list[dict]:
    """One entry per table in the file: title, prism_type, y_format, rows, sets, table (Table or None),
    reason (why skipped, else None), notes. A file-level problem yields a single entry named after the file."""
    path = Path(path)
    data = path.read_bytes()
    file_entry = {"title": path.name, "prism_type": "", "y_format": "", "rows": 0, "sets": 0, "table": None,
                  "notes": []}
    if data[:2] == b"PK":
        return [{**file_entry, "reason": "zipped Prism format (.prism / .prismt) is not supported"}]
    if b"<!ENTITY" in data:
        raise ValueError(f"{path}: XML entity declarations are not accepted")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(f"{path}: not valid XML ({e})") from e
    g = next((el for el in root.iter() if _loc(el) == "GraphPadPrismFile"), None)
    if g is None:
        return [{**file_entry, "reason": "no GraphPadPrismFile element"}]
    entries = [_entry(t) for t in g if _loc(t) in ("Table", "HugeTable")]
    return entries or [{**file_entry, "reason": "no tables"}]


def read_pzfx(path) -> tuple[list[Table], list[str]]:
    """Import every supported table; every skipped table gets a report line naming it and the reason."""
    tables, report = [], []
    for e in scan_pzfx(path):
        if e["table"] is not None:
            tables.append(e["table"])
            report += [f"{e['title']}: {n}" for n in e["notes"]]
        else:
            report.append(f"{e['title']}: skipped, {e['reason']}")
    return tables, report


def _col(parent, tag: str, title: str | None, subs: list[list[tuple[float, bool]]], **attrs):
    col = ET.SubElement(parent, tag, {**attrs, "Subcolumns": str(len(subs))})
    if title is not None:
        ET.SubElement(col, "Title").text = title
    for cells in subs:
        sub = ET.SubElement(col, "Subcolumn")
        for v, e in cells:
            d = ET.SubElement(sub, "d", {"Excluded": "1"} if e else {})
            d.text = fmt_num(v) or None
    return col


def _mv_col(parent, ds: DataSet) -> None:
    """One MultipleVariables variable, laid out like Prism's own files (cell types, category dictionary, stats)."""
    v, ex = ds.values[:, 0], ds.excluded[:, 0]
    ok = ~np.isnan(v)
    cats = ds.categories
    whole = ok & (np.where(ok, v, 0) == np.round(np.where(ok, v, 0)))
    n_dec = 0 if cats is not None else int((ok & ~whole).sum())
    col = ET.SubElement(parent, "YColumn", {
        "Width": "141", "Decimals": "3" if n_dec else "0", "Subcolumns": "1",
        "MVType": "Continues" if cats is None else "Categorical", "statDecimals": str(n_dec),
        "statNumbers": str(int(ok.sum()) - n_dec if cats is None else 0), "statTexts": str(int(ok.sum()) if cats else 0)})
    ET.SubElement(col, "Title").text = ds.title
    if cats is not None:
        cd = ET.SubElement(col, "CategoryDictionary")
        for k, label in enumerate(cats):
            rows = np.flatnonzero(ok & (v == k))
            ET.SubElement(cd, "d", {"id": str(k), "firstRowUsage": str(rows[0] if len(rows) else 0),
                                    "numUsage": str(int((~ex[rows]).sum())), "numExcluded": str(int(ex[rows].sum()))}).text = label
    sub = ET.SubElement(col, "Subcolumn")
    for r in range(len(v)):
        if not ok[r]:
            ET.SubElement(sub, "d")
            continue
        s = cats[int(v[r])] if cats is not None else fmt_num(v[r])
        attrs = {"CellType": "Text" if cats is not None else "Integer" if whole[r] else "Decimal", "UserText": s}
        if cats is not None:
            attrs["num"] = str(int(v[r]))
        if ex[r]:
            attrs["Excluded"] = "1"
        ET.SubElement(sub, "d", attrs).text = s


def _write_sub_titles(el, t: Table, nsub: int) -> None:
    if not any(d.sub_titles for d in t.datasets):
        return
    st = ET.SubElement(el, "SubColumnTitles", OwnSet="1")
    for s in range(nsub):
        sub = ET.SubElement(st, "Subcolumn")
        for d in t.datasets:
            ET.SubElement(sub, "d").text = (d.sub_titles[s] if d.sub_titles else "") or None


def _write_table(root, t: Table, i: int) -> None:
    t.validate()
    widths = {d.values.shape[1] for d in t.datasets}
    if len(widths) > 1:
        raise ValueError(f"{t.title}: data sets have different subcolumn counts")
    nsub = widths.pop() if widths else 1
    a = {"ID": f"Table{i}", "XFormat": "numbers" if t.kind in _XKINDS else "none"}
    if t.y_format != "replicates":
        a["YFormat"] = _FMT_OUT[t.y_format]
    elif t.kind in ("XY", "Grouped", "Nested"):
        a.update(YFormat="replicates", Replicates=str(nsub))
    a["TableType"] = _TYPE_OUT[t.kind]
    if t.kind in _EXT:
        a["ExtTableType"] = t.kind
    a["EVFormat"] = "AsteriskAfterNumber"
    el = ET.SubElement(root, "Table", a)
    ET.SubElement(el, "Title").text = t.title
    if t.row_titles:
        titles = list(t.row_titles) + [""] * (t.n_rows - len(t.row_titles))
        rt = ET.SubElement(el, "RowTitlesColumn", Width="81")
        sub = ET.SubElement(rt, "Subcolumn")
        for s in titles:
            ET.SubElement(sub, "d").text = s or None
    if t.kind in _XKINDS:
        _col(el, "XColumn", t.x_title, [[(v, False) for v in t.x]], Width="81", Decimals="3")
    if t.kind == "Nested":
        _write_sub_titles(el, t, nsub)
    for ds in t.datasets:
        if t.kind == "MultipleVariables":
            _mv_col(el, ds)
            continue
        subs = [[(v, bool(e) and not np.isnan(v)) for v, e in zip(ds.values[:, j], ds.excluded[:, j])]
                for j in range(ds.values.shape[1])]
        _col(el, "YColumn", ds.title, subs, Width="81" if nsub == 1 else str(81 * nsub), Decimals="3")


def write_pzfx(tables, path, title: str = "bioapp export") -> None:
    """Data-only .pzfx. Round trip with read_pzfx keeps values, NaN, exclusions, titles, layout, Nested subcolumn
    titles and the categories of MultipleVariables variables."""
    tables = list(tables)
    if not tables:
        raise ValueError("no tables to write")
    root = ET.Element("GraphPadPrismFile", PrismXMLVersion="5.00")
    created = ET.SubElement(root, "Created")
    ET.SubElement(created, "OriginalVersion", CreatedByProgram="bioapp", CreatedByVersion="0.1", Login="",
                  DateTime=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    ET.SubElement(ET.SubElement(root, "InfoSequence"), "Ref", ID="Info0", Selected="1")
    info = ET.SubElement(root, "Info", ID="Info0")
    ET.SubElement(info, "Title").text = title
    ET.SubElement(info, "Notes")
    seq = ET.SubElement(root, "TableSequence", Selected="1")
    for i in range(len(tables)):
        ET.SubElement(seq, "Ref", {"ID": f"Table{i}", **({"Selected": "1"} if i == 0 else {})})
    for i, t in enumerate(tables):
        _write_table(root, t, i)
    ET.indent(root)
    atomic_write_bytes(path, ET.tostring(root, encoding="utf-8", xml_declaration=True))
