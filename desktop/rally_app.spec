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

# Windows/Linux use the .ico (EXE.icon below); macOS uses the .icns (BUNDLE.icon).
_ICON_ICO = 'assets/icon/AnyaTennis.ico'
_ICON_ICNS = 'assets/icon/AnyaTennis.icns'

# version.py is the single source of truth for the app version — imported
# here (rather than duplicating the string) so the bundle's Info.plist can't
# drift from what's shown in the app header.
sys.path.insert(0, str(Path('.').resolve()))
from version import APP_VERSION

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
        ('../mobile/assets/images/anya_logo.png', 'assets'),
        # Montserrat, burned into the Scoreboard tab's rendered overlay —
        # resolved by pipeline.scoreboard_reel.render.find_font()
        ('assets/fonts/Montserrat-SemiBold.ttf', 'assets/fonts'),
        ('assets/fonts/Montserrat-Bold.ttf', 'assets/fonts'),
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
        # scoreboard_reel (Scoreboard tab: scoring engine + ffmpeg burn-in)
        'pipeline.scoreboard_reel',
        # PyQt6 plugins (Fusion style)
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        # Scoreboard tab: in-app video playback while tagging points
        'PyQt6.QtMultimedia',
        'PyQt6.QtMultimediaWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_cv2.py'],
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
    # Left unsigned here on purpose: PyInstaller's own --deep-equivalent
    # signing is unreliable on a bundle this size (torch/opencv/ultralytics —
    # hundreds of nested dylibs) and has been known to sign out of order or
    # skip binaries, which passes locally but fails notarization. macOS
    # signing + notarization + stapling happens entirely in build_macos.sh,
    # run after this spec.
    codesign_identity=None,
    entitlements_file=None,
    icon=_ICON_ICO,         # used on Windows; PyInstaller ignores it on macOS/Linux
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
        icon=_ICON_ICNS,
        bundle_identifier='com.anyatennis.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundleName': 'Anya Tennis',
        },
    )
