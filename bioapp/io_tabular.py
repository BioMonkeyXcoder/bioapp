"""Tabular import/export: CSV / TSV / TXT / XLSX / clipboard text <-> Table."""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
from openpyxl.utils import get_column_letter

from .data import DataSet, ONE_SUBCOLUMN_KINDS, TABLE_KINDS, Table, Y_FORMATS, new_id
from .project import atomic_write_bytes

_NUM = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def fmt_num(v: float) -> str:
    """Shortest text that parses back to the same float; NaN -> blank. Infinity is rejected."""
    v = float(v)
    if np.isnan(v):
        return ""
    if np.isinf(v):
        raise ValueError("infinite values cannot be exported")
    s = repr(v)
    return s[:-2] if s.endswith(".0") else s


def _rect(rows: list[list[str]]) -> list[list[str]]:
    """Drop trailing fully blank rows and pad every row to the same width."""
    while rows and not any(c.strip() for c in rows[-1]):
        rows.pop()
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def parse_text_grid(text: str) -> list[list[str]]:
    """Clipboard text (tab separated, optional quoting as written by spreadsheets)."""
    return _rect([list(r) for r in csv.reader(io.StringIO(text), delimiter="\t")])


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")  # legacy ANSI export; never fails


def _xlsx_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return fmt_num(v) if np.isfinite(v) else str(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def read_grid(path, sheet: str | None = None) -> list[list[str]]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv", ".txt"):
        text = _read_text(path)
        if ext == ".tsv":
            delim = "\t"
        else:
            try:
                delim = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                delim = ","
        return _rect([list(r) for r in csv.reader(io.StringIO(text), delimiter=delim)])
    if ext == ".xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet is not None and sheet not in wb.sheetnames:
                raise ValueError(f"{path.name}: no sheet {sheet!r}; sheets are {wb.sheetnames}")
            ws = wb[sheet] if sheet is not None else wb.active
            return _rect([[_xlsx_cell(v) for v in row] for row in ws.iter_rows(values_only=True)])
        finally:
            wb.close()
    raise ValueError(f"unsupported file type {ext!r}; use .csv, .tsv, .txt or .xlsx")


def _number(raw: str, decimal: str, addr: str) -> float:
    s = raw.strip()
    if not s:
        return np.nan
    t = s.replace(",", ".") if decimal == "," else s
    if not _NUM.fullmatch(t):
        raise ValueError(f"{addr}: {raw!r}")
    return float(t)


def grid_to_table(grid, kind: str, *, header_rows: int = 1, x_col: int = 0, row_title_col: int | None = None,
                  replicates: int = 1, y_format: str = "replicates", decimal: str = ".",
                  title: str = "Imported") -> tuple[Table, list[str]]:
    """Convert a text grid to a Table.

    Y columns are every column except the X column (XY only) and the row-title column; consecutive Y
    columns form one data set of ``replicates`` (or of the mean/error fields of ``y_format``) subcolumns.
    Header text of the first column of each group names the data set. Blank cells become NaN;
    non-numeric cells raise ValueError naming the sheet cell, e.g. ``C7: 'n.d.'``. Kinds:
    Survival: the X column holds the times, every other column is one group of 1 (event) / 0 (censored) codes.
    Contingency: counts, validated as non-negative integers. Nested: ``header_rows`` >= 2; row 1 names the groups,
    row 2 the subcolumns (subjects), ``replicates`` consecutive columns per group.
    MultipleVariables: one variable per column; a column with any non-numeric text becomes categorical (labels coded
    in order of first appearance, blank stays NaN).
    """
    if kind not in TABLE_KINDS:
        raise ValueError(f"unknown table kind {kind!r}")
    if y_format not in Y_FORMATS:
        raise ValueError(f"unknown y_format {y_format!r}")
    if decimal not in (".", ","):
        raise ValueError("decimal must be '.' or ','")
    fields = Y_FORMATS[y_format]
    nsub = replicates if fields is None else len(fields)
    if nsub < 1 or (kind == "Column" and fields is None and nsub != 1):
        raise ValueError("Column tables with raw values have exactly one subcolumn per data set")
    if kind in ONE_SUBCOLUMN_KINDS and (y_format != "replicates" or nsub != 1):
        raise ValueError(f"{kind} tables have one raw-value column per data set")
    if kind == "Nested" and (y_format != "replicates" or header_rows < 2):
        raise ValueError("Nested tables need raw values and two header rows (group titles, subcolumn titles)")
    rows = _rect([list(map(str, r)) for r in grid])
    if len(rows) <= header_rows:
        raise ValueError("no data rows")
    head, body = rows[:header_rows], rows[header_rows:]
    width = len(rows[0])
    xc = x_col if kind in ("XY", "Survival") else None
    for c in (xc, row_title_col):
        if c is not None and not 0 <= c < width:
            raise ValueError(f"column index {c} is outside the {width} columns of the grid")
    used = {c for c in (xc, row_title_col) if c is not None}
    ycols = [c for c in range(width) if c not in used]
    if not ycols or len(ycols) % nsub:
        raise ValueError(f"{len(ycols)} Y columns cannot be grouped into data sets of {nsub} subcolumns")

    def addr(i: int, c: int) -> str:
        return f"{get_column_letter(c + 1)}{header_rows + i + 1}"

    def cell(i: int, c: int) -> float:
        v = _number(body[i][c], decimal, addr(i, c))
        if not np.isnan(v):
            if kind == "Contingency" and (v < 0 or v != round(v)):
                raise ValueError(f"{addr(i, c)}: {body[i][c]!r} is not a non-negative integer count")
            if kind == "Survival" and c != xc and v not in (0, 1):
                raise ValueError(f"{addr(i, c)}: {body[i][c]!r} is not a survival code (1 = event, 0 = censored)")
        return v

    n = len(body)
    sets, cat_names = [], []
    for k in range(0, len(ycols), nsub):
        cols = ycols[k:k + nsub]
        name = next((head[0][c].strip() for c in cols if head and head[0][c].strip()), "")
        name = name or f"Data set {k // nsub + 1}"
        if kind == "MultipleVariables":
            try:
                values, cats = np.array([[cell(i, cols[0])] for i in range(n)], dtype=float), None
            except ValueError:  # text in the column: categorical variable
                labels = [body[i][cols[0]].strip() for i in range(n)]
                cats = list(dict.fromkeys(s for s in labels if s))
                values = np.array([[cats.index(s) if s else np.nan] for s in labels], dtype=float)
                cat_names.append(name)
            sets.append(DataSet(new_id("ds"), name, values, categories=cats))
            continue
        values = np.array([[cell(i, c) for c in cols] for i in range(n)], dtype=float)
        subs = [head[1][c].strip() for c in cols] if kind == "Nested" else []
        sets.append(DataSet(new_id("ds"), name, values, sub_titles=subs if any(subs) else []))
    x = np.array([cell(i, xc) for i in range(n)], dtype=float) if xc is not None else None
    x_title = (head[0][xc].strip() or "X") if xc is not None and head else "X"
    row_titles = [body[i][row_title_col].strip() for i in range(n)] if row_title_col is not None else []
    table = Table(new_id("tbl"), title, kind, y_format, sets, x, x_title, row_titles)
    blanks = sum(int(np.isnan(d.values).sum()) for d in sets)
    report = [f"{title}: {n} rows, {len(sets)} data sets, {blanks} blank cells"]
    if cat_names:
        report.append(f"{title}: categorical variables (text): {', '.join(cat_names)}")
    return table, report


def table_to_grid(table: Table) -> list[list[str]]:
    """Grid for CSV export. Excluded values are exported like any other value (nothing is dropped).
    Categorical variables are exported as their labels; Nested tables get a second header row of subcolumn titles."""
    fields = Y_FORMATS[table.y_format]
    head, head2, cols = [], [], []
    if table.row_titles:
        head.append("")
        head2.append("")
        cols.append(list(table.row_titles) + [""] * (table.n_rows - len(table.row_titles)))
    if table.x is not None:
        head.append(table.x_title)
        head2.append("")
        cols.append([fmt_num(v) for v in table.x])
    for ds in table.datasets:
        for j in range(ds.values.shape[1]):
            head.append(ds.title if fields is None or j == 0 else f"{ds.title} ({fields[j]})")
            head2.append(ds.sub_titles[j] if ds.sub_titles else "")
            cols.append([ds.categories[int(v)] if ds.categories is not None and not np.isnan(v) else fmt_num(v)
                         for v in ds.values[:, j]])
    heads = [head, head2] if table.kind == "Nested" else [head]
    return heads + [list(r) for r in zip(*cols)] if cols else heads


def write_csv(table: Table, path) -> None:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(table_to_grid(table))
    atomic_write_bytes(path, buf.getvalue().encode("utf-8-sig"))
