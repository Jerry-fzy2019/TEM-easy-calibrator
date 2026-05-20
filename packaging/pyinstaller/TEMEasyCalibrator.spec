# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve().parents[1]
INCLUDE_DM3 = os.environ.get("INCLUDE_DM3", "0") == "1"

datas = [(str(ROOT / "src"), "src")]
binaries = []
hiddenimports = []

for package_name in ("streamlit", "plotly", "altair", "pydeck"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
    datas += copy_metadata(package_name)

if INCLUDE_DM3:
    package_datas, package_binaries, package_hiddenimports = collect_all("hyperspy")
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
    datas += copy_metadata("hyperspy")

hiddenimports += collect_submodules("skimage")


a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[] if INCLUDE_DM3 else ["hyperspy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TEM Easy Calibrator",
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
    name="TEM Easy Calibrator",
)
