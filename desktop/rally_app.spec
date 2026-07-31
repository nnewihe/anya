# rally_app.spec
# PyInstaller spec for Anya Tennis (pipeline.rally_reel).
#
# Weights are pulled straight from the repo (pipeline/models, walking/outputs)
# — nothing to copy by hand.  Destinations mirror the source tree because the
# pipeline resolves its weights relative to its own __file__, which under
# PyInstaller lands under sys._MEIPASS with the same layout.
#
# Build:
#   macOS  : pyinstaller rally_app.spec   -> dist/Anya Tennis.app
#   Windows: pyinstaller rally_app.spec   -> dist/AnyaTennis/  (one folder)
#   Linux  : pyinstaller rally_app.spec   -> dist/AnyaTennis/  (one folder)
#
# ffmpeg is NOT bundled: rally_reel shells out to it for the final cut, so it
# must be on PATH on the target machine (brew install ffmpeg / apt install
# ffmpeg / winget install ffmpeg).
#
# Expect a large bundle (~2 GB): torch + ultralytics are required by the
# telemetry and pose stages and cannot be excluded.

import sys
from pathlib import Path

block_cipher = None

# app.py imports the pipeline as a package (`from pipeline.X import …`), so the
# repo ROOT (parent of desktop/) must be on the analysis path, not just desktop/.
_REPO_ROOT = str(Path('.').resolve().parent)

a = Analysis(
    ['app.py'],
    pathex=[str(Path('.').resolve()), _REPO_ROOT],
    binaries=[],
    datas=[
        # Weights, at the paths the pipeline resolves relative to its own
        # __file__ (pipeline/models, walking/outputs).
        ('../pipeline/models/ball_best.pt',      'pipeline/models'),
        ('../pipeline/models/yolo26n.pt',        'pipeline/models'),
        ('../pipeline/models/yolov8n-pose.pt',   'pipeline/models'),
        ('../walking/outputs/walking_model.joblib', 'walking/outputs'),
        # Brand logo (shared with the mobile app) — resolved by app._logo_path()
        ('../mobile/assets/images/anya_logo_black.svg', 'assets'),
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
        # sklearn — cluster for exclusion zones, ensemble for the walking model
        'sklearn',
        'sklearn.cluster',
        'sklearn.cluster._dbscan',
        'sklearn.ensemble',
        'sklearn.ensemble._hist_gradient_boosting',
        'sklearn.ensemble._hist_gradient_boosting.gradient_boosting',
        # walking classifier: loaded via joblib, imported dynamically in
        # rally_reel.reel._walk_intervals so the analyzer cannot see it
        'joblib',
        'scipy',
        'walking',
        'walking.predict',
        'walking.evaluate',
        'walking.features',
        'walking.court',
        'walking.select_near',
        'walking.extract_pose',
        # rally_reel stages
        'pipeline.rally_reel',
        'pipeline.extract_far_pose',
        'pipeline.anya_far_serve',
        'pipeline.anya_near_serve',
        'pipeline.anya_telemetry',
        # PyQt6 plugins (Fusion style)
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        # SVG logo rendering in the header
        'PyQt6.QtSvg',
        'PyQt6.QtSvgWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # NOTE torch is NOT excluded: anya_telemetry imports it to pick the
        # mps/cuda/cpu device, and ultralytics needs it for every model call.
        # Excluding it (as this spec used to) builds cleanly and then fails
        # at runtime on the first stage.
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
    name='AnyaTennis',
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
    name='AnyaTennis',
)

# macOS .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Anya Tennis.app',
        icon=None,           # set to 'icon.icns' if you have one
        bundle_identifier='com.anyatennis.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleName': 'Anya Tennis',
        },
    )
