# -*- mode: python ; coding: utf-8 -*-


import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = []
hiddenimports = []
binaries = []
for package in ("sonora", "anyascii", "pycountry", "ftfy", "syncedlyrics", "music_metadata_filter", "filelock", "ijson"):
    datas += collect_data_files(package)
    hiddenimports += collect_submodules(package)
for package in ("soundfile", "shazamio", "shazamio_core", "msgpack"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports


a = Analysis(
    ['src/sonora/cli/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Fingerprinting uses fpcalc directly; pyacoustid's optional decoder pulls in GUI stacks.
    excludes=['audioread'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='sonora-windows-x86_64' if sys.platform == 'win32' else 'sonora-linux-x86_64',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
