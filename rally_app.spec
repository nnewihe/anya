# rally_app.spec
# PyInstaller spec for the Rally Detector cross-platform app.
#
# Before building, copy model weights into the models/ directory:
#   models/ball_best.pt   ← from weights/ball/weights/best.pt
#   models/yolo26n.pt     ← your player detection model
#
# Build commands:
#   macOS  : pyinstaller rally_app.spec
#   Windows: pyinstaller rally_app.spec
#
# Output: dist/RallyDetector.app  (macOS)
#         dist/RallyDetector/      (Windows one-folder)

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[str(Path('.')  .resolve())],
    binaries=[],
    datas=[
        # Bundle model weights — destination is the 'models' subfolder inside the app
        ('models/ball_best.pt',  'models'),
        ('models/yolo26n.pt',    'models'),
    ],
    hiddenimports=[
        # ultralytics dynamic imports
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.utils',
        'ultralytics.models',
        # filterpy
        'filterpy',
        'filterpy.kalman',
        # sklearn
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster._dbscan',
        # PyQt6 plugins (Fusion style)
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy packages not needed for rally_detector path
        'torch',
        'torch_geometric',
        'mediapipe',
        'tensorflow',
        'matplotlib',
        'IPython',
        'notebook',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RallyDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # set to 'icon.icns' / 'icon.ico' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RallyDetector',
)

# macOS .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='RallyDetector.app',
        icon=None,           # set to 'icon.icns' if you have one
        bundle_identifier='com.rallydetector.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleName': 'Rally Detector',
        },
    )
