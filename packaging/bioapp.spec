# PyInstaller one-folder windowed build "bioapp".
# Run from the project root:
#   pyinstaller --noconfirm --workpath <workspace>\intermediate\pyinstaller --distpath <workspace>\output\dist packaging\bioapp.spec
from pathlib import Path
import sys

sys.path.insert(0, SPECPATH)
from windows_runtime import prepare_collection

root = Path(SPECPATH).parent

# Qt modules bioapp never imports (matplotlib's Qt backend needs QtCore/QtGui/QtWidgets/QtSvg).
unused_qt = ["Qt3D*", "QtBluetooth", "QtCharts", "QtDataVisualization", "QtMultimedia", "QtNetwork", "QtNfc",
             "QtPdf", "QtPositioning", "QtQml", "QtQuick", "QtRemoteObjects", "QtScxml", "QtSensors",
             "QtSerialPort", "QtSql", "QtTest", "QtWebChannel", "QtWebEngine", "QtWebSockets", "QtDesigner",
             "QtHelp", "QtSvgWidgets", "QtOpenGL", "QtOpenGLWidgets", "QtPrintSupport", "QtXml"]
excludes = ["tkinter", "_tkinter", "pytest", "_pytest", "IPython", "jupyter", "notebook", "sphinx", "setuptools",
            "pip", "wx", "PyQt5", "PyQt6", "PySide2", "matplotlib.tests", "numpy.tests", "pandas.tests",
            "scipy.tests", "statsmodels.tests"]
excludes += [f"PySide6.{m}" for m in unused_qt if "*" not in m] + ["PySide6.Qt3DCore", "PySide6.Qt3DRender"]

a = Analysis(
    [str(root / "bioapp" / "__main__.py")],
    pathex=[str(root)],
    datas=[(str(root / "bioapp" / "analysis" / "spearman_null.npz"), "bioapp/analysis"),
           (str(root / "bioapp" / "ui" / "bioapp.ico"), "bioapp/ui")],  # window icon: the 2026-11-17 Blender cover, cropped to a rounded tile
    hiddenimports=["bioapp.plot", "bioapp.analysis.fit", "bioapp.analysis.stats", "bioapp.analysis.transforms",
                   "bioapp.analysis.models", "bioapp.analysis.designs", "bioapp.analysis.categorical",
                   "bioapp.analysis.regression", "bioapp.analysis.multivariate",
                   "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_agg",
                   "matplotlib.backends.backend_svg", "matplotlib.backends.backend_pdf",
                   "matplotlib.backends.backend_ps", "openpyxl"],
    excludes=excludes,
    noarchive=False,
)
# Drop bundled test suites of third-party packages.
a.datas = [d for d in a.datas if "/tests/" not in d[0].replace("\\", "/")]
a.pure = [p for p in a.pure if ".tests." not in p[0]]

a.binaries = prepare_collection(a.binaries, workpath)
a.datas = prepare_collection(a.datas, workpath)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="bioapp", console=False, upx=False,
          icon=str(root / "bioapp" / "ui" / "bioapp.ico"))
coll = COLLECT(exe, a.binaries, a.datas, name="bioapp", upx=False)


