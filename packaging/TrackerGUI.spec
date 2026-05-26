# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_data_files


project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(project_root, "LICENSE"), "."),
    (os.path.join(project_root, "packaging", "cv2", "config.py"), "cv2"),
    (os.path.join(project_root, "packaging", "cv2", "config-3.12.py"), "cv2"),
]
binaries = []
hiddenimports = [
    "PIL.Image",
    "PIL.ImageTk",
    "matplotlib.backends.backend_tkagg",
    "crop.SelectVideoIntervals",
    "crop.CropVideosFromIntervals",
    "freeze.AutoFreezeCalibration",
    "freeze.BuildFreezeConfigGUI",
    "freeze.FreezeAnalysis_Functions",
    "freeze.RunFreezeAnalysisFromYAML",
]

# Keep the release folder small: collect only GUI/data assets that PyInstaller
# cannot infer reliably. Native scientific libraries are handled by hooks.
datas += collect_data_files("customtkinter")
datas += collect_data_files("matplotlib", includes=["mpl-data/**"])


a = Analysis(
    [os.path.join(project_root, "TrackerGUI.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        "matplotlib": {
            "backends": ["TkAgg"],
        },
    },
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "bokeh",
        "holoviews",
        "panel",
        "IPython",
        "scipy",
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "scipy.tests",
        "IPython.testing",
        "holoviews.tests",
        "bokeh.sampledata",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TrackerGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TrackerGUI",
)
