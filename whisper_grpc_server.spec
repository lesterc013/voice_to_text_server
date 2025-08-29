# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\whisper_grpc_server.py'],
    pathex=[],
    binaries=[],
    datas=[('whisper_downloads/whisper_medium_model', 'whisper_downloads/whisper_medium_model'), ('whisper_downloads/whisper_medium_processor', 'whisper_downloads/whisper_medium_processor')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whisper_grpc_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='whisper_grpc_server',
)
