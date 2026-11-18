"""Entry point: ``python -m bioapp`` (GUI), ``--selftest`` (worker pool only), ``--selftest-gui`` (offscreen friendly)."""
from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
import time
from pathlib import Path


def synthetic_tables():
    """A dose-response XY table (3 replicates, 2 data sets) and a 3-group Column table; fixed seed."""
    import numpy as np
    from bioapp.data import DataSet, Table
    rng = np.random.default_rng(7)
    x = np.linspace(-9, -4, 11)

    def curve(top, ec50):
        return 5 + (top - 5) / (1 + 10 ** ((ec50 - x) * 1.1))
    sets = [DataSet(f"ds_{n}", n, curve(top, ec) [:, None] + rng.normal(0, 3, (len(x), 3)))
            for n, top, ec in (("Control", 100, -6.5), ("Treated", 85, -6.0))]
    xy = Table("tbl_xy", "Dose response", "XY", datasets=sets, x=x, x_title="log[agonist], M")
    groups = [DataSet(f"ds_g{i}", f"Group {i + 1}", rng.normal(10 + 3 * i, 2, (8, 1))) for i in range(3)]
    col = Table("tbl_col", "Three groups", "Column", datasets=groups)
    return xy, col


def synthetic_survival():
    """A two-group Survival table (20 subjects each, 1 = event, 0 = censored); fixed seed."""
    import numpy as np
    from bioapp.data import DataSet, Table
    rng = np.random.default_rng(11)
    t = np.round(np.r_[rng.exponential(10, 20), rng.exponential(20, 20)] + 0.5, 1)
    e = (rng.random(40) < 0.8).astype(float)
    sets = [DataSet(f"ds_s{g}", name, np.where(np.arange(40) // 20 == g, e, np.nan))
            for g, name in enumerate(("Control", "Treated"))]
    return Table("tbl_surv", "Survival", "Survival", datasets=sets, x=t, x_title="Days")


def log_path(name: str) -> Path:
    base = os.environ.get("BIOAPP_LOG_DIR") or str(Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "bioapp" / "logs")
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(base) / name


class Say:
    """print + log file (a windowed build has no console)."""

    def __init__(self, name):
        self.f = open(log_path(name), "w", encoding="utf-8")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        self.f.write(line + "\n")
        self.f.flush()
        if sys.stdout is not None:
            print(line, flush=True)


def selftest() -> int:
    from PySide6.QtCore import QCoreApplication
    from bioapp import jobs
    say = Say("selftest.log")
    t0 = time.time()
    qt = QCoreApplication.instance() or QCoreApplication([])
    pool = jobs.JobPool(2)
    xy, col = synthetic_tables()
    out: dict = {}
    pool.done.connect(lambda tid, ok, res: out.__setitem__(tid, (ok, res)))
    pool.submit("fit", "analysis", ("nonlin_fit", xy, ["ds_Control", "ds_Treated"],
                                    {"model": "log_agonist_variable"}))
    pool.submit("ttest", "analysis", ("t_test", col, ["ds_g0", "ds_g1"], {}))
    end = time.time() + 120
    while len(out) < 2 and time.time() < end:
        qt.processEvents()
        time.sleep(0.01)
    pool.shutdown()
    code = 0
    for tid in ("fit", "ttest"):
        ok, res = out.get(tid, (False, "timed out"))
        status = res.get("status") if ok else "error"
        say(f"{tid}: ok={ok} status={status}")
        if ok and tid == "fit":
            for ds, r in res["datasets"].items():
                say(f"  {ds}: " + ", ".join(f"{k}={v['value']:.4g}" for k, v in r["params"].items()))
        if ok and tid == "ttest":
            for d in res["display"][:1]:
                say("  " + d["title"])
                for row in d["rows"][:6]:
                    say("   ", row)
        if not ok:
            say(str(res))
        if not ok or status == "failed":
            code = 1
    say(f"selftest {'passed' if code == 0 else 'FAILED'} in {time.time() - t0:.1f} s")
    return code


def selftest_gui() -> int:
    from PySide6.QtWidgets import QApplication
    from bioapp.app import App
    from bioapp.ui import theme
    from bioapp.ui.main_window import MainWindow
    say = Say("selftest_gui.log")
    t0 = time.time()
    qapp = QApplication.instance() or QApplication(sys.argv)
    theme.apply(qapp, theme.saved_style())
    os.environ.setdefault("BIOAPP_RECOVERY_DIR", tempfile.mkdtemp(prefix="bioapp_rec_"))
    win = MainWindow(App())
    win.show()
    xy, col = synthetic_tables()
    sv = synthetic_survival()
    win.app.add_tables([xy, col, sv])
    win.app.add_analysis(xy.id, "nonlin_fit", ["ds_Control", "ds_Treated"], {"model": "log_agonist_variable"})
    win.app.add_analysis(col.id, "one_way_anova", [d.id for d in col.datasets], {"multiple_comparisons": "tukey"})
    km = win.app.add_analysis(sv.id, "survival", [d.id for d in sv.datasets], {})  # a P6-P8 analysis and graph kind
    if not win.app.wait_idle(120):
        say("analyses did not finish")
        return 1
    bad = [(a.title, a.status) for a in win.app.project.analyses.values() if a.status not in ("ok", "warning")]
    if bad:
        say(f"analyses failed: {bad}")
        return 1
    sgids = win.app.graph_result(km.id)
    if len(sgids) != 1:
        say(f"graph this result gave {sgids}")
        return 1
    win.select_item("graphs", sgids[0])
    qapp.processEvents()
    gid = win.app.new_graph(xy.id)
    win.select_item("graphs", gid)
    qapp.processEvents()
    gid2 = win.app.new_graph(col.id)
    win.select_item("graphs", gid2)
    qapp.processEvents()
    lid = win.app.new_layout()
    win.select_item("layouts", lid)
    qapp.processEvents()
    folder = Path(tempfile.mkdtemp(prefix="bioapp_gui_"))
    for ext in ("svg", "pdf", "png", "tiff", "eps"):
        win.app.export_figure("graph", gid, folder / f"graph.{ext}", dpi=150)
    win.app.export_figure("layout", lid, folder / "layout.pdf")
    win.app.export_figure("graph", sgids[0], folder / "survival.png", dpi=150)
    path = folder / "selftest.biofig"
    win.app.save(path)
    from bioapp import project
    back = project.load_project(path)
    ok = (len(back.tables) == len(win.app.project.tables) and len(back.analyses) == 3 and len(back.graphs) == 3
          and all(a.result for a in back.analyses.values()))
    say(f"saved {path} and reloaded: tables={len(back.tables)} analyses={len(back.analyses)} graphs={len(back.graphs)}")
    win.close()
    say(f"selftest-gui {'passed' if ok else 'FAILED'} in {time.time() - t0:.1f} s")
    return 0 if ok else 1


def run_gui() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from bioapp.app import App
    from bioapp.ui import theme
    from bioapp.ui.main_window import MainWindow
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("bioapp")
    theme.apply(qapp, theme.saved_style())
    app = App()
    win = MainWindow(app)
    win.show()
    QTimer.singleShot(0, win.offer_recovery)
    QTimer.singleShot(200, app.pool.warm)
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".biofig"):
        win.open_path(sys.argv[1])
    return qapp.exec()


def main() -> int:
    multiprocessing.freeze_support()  # first: a frozen worker re-enters this exe
    for flag, fn in (("--selftest", selftest), ("--selftest-gui", selftest_gui)):
        if flag in sys.argv:
            try:
                return fn()
            except BaseException:  # a windowed build would otherwise show an error dialog and hang
                import traceback
                Say(flag.strip("-").replace("-", "_") + "_error.log")(traceback.format_exc())
                return 1
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())



