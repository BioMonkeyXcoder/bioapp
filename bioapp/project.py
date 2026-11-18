"""Native project file (.biofig): a zip with project.json and tables/<id>.npz.

Saving never damages an existing file: the bytes are built in memory, written to a temp
file in the target directory, fsynced and moved over the target with ``os.replace``.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .data import Analysis, DataSet, Project, Table

SCHEMA_VERSION = 1
FORMAT_NAME = "bioapp-project"


def atomic_write_bytes(path, data: bytes) -> None:
    """Replace ``path`` with ``data`` atomically; on any error the old file stays and the temp file is removed."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.resolve().parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_default(o):
    # Results may carry numpy scalars/arrays; anything else is a contract violation.
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _table_meta(t: Table) -> dict:
    return {"id": t.id, "title": t.title, "kind": t.kind, "y_format": t.y_format, "x_title": t.x_title,
            "row_titles": list(t.row_titles), "revision": t.revision, "source": t.source,
            "has_x": t.x is not None,
            "datasets": [{"id": d.id, "title": d.title, "sub_titles": list(d.sub_titles),
                          "categories": None if d.categories is None else list(d.categories)} for d in t.datasets]}


def _table_npz(t: Table) -> bytes:
    arrays = {}
    if t.x is not None:
        arrays["x"] = t.x
    for i, d in enumerate(t.datasets):
        arrays[f"values_{i}"] = d.values
        arrays[f"excluded_{i}"] = d.excluded
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _analysis_meta(a: Analysis) -> dict:
    return {"id": a.id, "title": a.title, "kind": a.kind, "table_id": a.table_id, "dataset_ids": list(a.dataset_ids),
            "params": a.params, "params_revision": a.params_revision, "result": a.result,
            "result_key": None if a.result_key is None else list(a.result_key),
            "status": a.status, "output_table_id": a.output_table_id}


def save_project(project: Project, path) -> None:
    meta = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "title": project.title,
        "tables": [_table_meta(t) for t in project.tables.values()],
        "analyses": [_analysis_meta(a) for a in project.analyses.values()],
        "graphs": {k: v.to_dict() for k, v in project.graphs.items()},
        "layouts": {k: v.to_dict() for k, v in project.layouts.items()},
    }
    text = json.dumps(meta, default=_json_default, allow_nan=True, ensure_ascii=False, indent=1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", text.encode("utf-8"))
        for t in project.tables.values():
            z.writestr(f"tables/{t.id}.npz", _table_npz(t))
    atomic_write_bytes(path, buf.getvalue())


def _load_table(m: dict, z: zipfile.ZipFile) -> Table:
    with np.load(io.BytesIO(z.read(f"tables/{m['id']}.npz")), allow_pickle=False) as npz:
        x = npz["x"] if m["has_x"] else None
        # sub_titles / categories are absent in files written before the Nested / MultipleVariables kinds
        sets = [DataSet(d["id"], d["title"], npz[f"values_{i}"], npz[f"excluded_{i}"],
                        list(d.get("sub_titles", [])), d.get("categories"))
                for i, d in enumerate(m["datasets"])]
    return Table(id=m["id"], title=m["title"], kind=m["kind"], y_format=m["y_format"], datasets=sets, x=x,
                 x_title=m["x_title"], row_titles=list(m["row_titles"]), revision=m["revision"], source=m["source"])


def load_project(path) -> Project:
    try:
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("project.json").decode("utf-8"))
            if not isinstance(meta, dict) or meta.get("format") != FORMAT_NAME:
                raise ValueError(f"{path}: not a bioapp project file")
            if meta["schema_version"] > SCHEMA_VERSION:
                raise ValueError(f"{path}: project schema {meta['schema_version']} is newer than the supported "
                                 f"version {SCHEMA_VERSION}; update bioapp to open it")
            tables = {m["id"]: _load_table(m, z) for m in meta["tables"]}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"{path}: not a readable bioapp project ({type(e).__name__}: {e})") from e
    analyses = {}
    for a in meta["analyses"]:
        a = dict(a, result_key=None if a["result_key"] is None else tuple(a["result_key"]))
        analyses[a["id"]] = Analysis(**a)
    graphs, layouts = {}, {}
    if meta["graphs"] or meta["layouts"]:
        plot = importlib.import_module("bioapp.plot")
        graphs = {k: plot.PlotSpec.from_dict(v) for k, v in meta["graphs"].items()}
        layouts = {k: plot.LayoutSpec.from_dict(v) for k, v in meta["layouts"].items()}
    return Project(title=meta["title"], tables=tables, analyses=analyses, graphs=graphs, layouts=layouts,
                   schema_version=meta["schema_version"])
