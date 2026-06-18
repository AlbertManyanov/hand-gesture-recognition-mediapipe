# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('model/hand_landmarker.task', 'model/'),
        
        # Датасет
        ('landmarks.csv', '.'),
        
        # ВСЯ папка MediaPipe (ОБЯЗАТЕЛЬНО для работы на других ПК)
        (r'C:\Users\Albert\AppData\Local\Programs\Python\Python314\Lib\site-packages\mediapipe', 'mediapipe/'),],
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
    a.binaries,
    a.datas,
    [],
    name='GestureRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
