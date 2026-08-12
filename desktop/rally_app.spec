# rally_app.spec
# PyInstaller spec for Anya Tennis (pipeline.rally_reel).
#
# Weights are pulled straight from the repo (pipeline/models, walking/outputs)
# — nothing to copy by hand.  Destinations mirror the source tree because the
# pipeline resolves its weights relative to its own __file__, which under
# PyInstaller lands under sys._MEIPASS with the same layout.
#
# Build (PyInstaller cannot cross-compile — each target builds on its own OS):
#   macOS  : ./build_macos.sh             -> dist/Anya Tennis.app  (signed)
#   Windows: .\build_windows.ps1          -> dist/AnyaTennis/ + the setup .exe
#   Linux  : pyinstaller rally_app.spec   -> dist/AnyaTennis/  (one folder)
#
# One folder, never --onefile: the payload is ~2 GB, and a one-file build
# re-extracts all of it to a temp directory on every single launch.
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

_WINDOWS = sys.platform == 'win32'

# UPX compresses each binary and unpacks it in memory at load. That is a bad
# trade here and an outright hazard on Windows: torch ships DLLs (and PyQt6
# ships Qt6*.dll) that UPX is known to corrupt, producing a build that links
# fine and then dies with a bare "DLL load failed while importing _C" the
# first time a stage touches torch. The bundle is ~2 GB of mostly-already-
# compressed weights, so UPX buys little size back even when it works.
# Left enabled on macOS/Linux, where the existing signed builds use it.
_UPX = not _WINDOWS

# The .exe's Details tab (right-click -> Properties). Windows shows "Unknown"
# for every field without this resource, and an unsigned installer already
# gives SmartScreen enough to complain about — a blank publisher makes a
# tester's "is this safe?" question harder to answer than it needs to be.
# Driven off APP_VERSION so it cannot drift from the header the app renders.
def _version_resource():
    """Build a VSVersionInfo from APP_VERSION, or None if it can't be parsed."""
    # The binary FILEVERSION field is four integers and nothing else — it
    # cannot carry "-beta.3". Parse the numeric prefix for that field and keep
    # the full string, suffix and all, in the human-readable StringStruct.
    numeric = APP_VERSION.split('-')[0].split('.')
    try:
        parts = [int(p) for p in numeric][:3]
    except ValueError:
        return None
    while len(parts) < 4:
        parts.append(0)
    filevers = tuple(parts[:4])

    # Imported here rather than at module scope: this module pulls in pefile
    # and pywin32, which PyInstaller only installs on Windows, so a top-level
    # import would break the macOS build outright.
    try:
        from PyInstaller.utils.win32.versioninfo import (
            FixedFileInfo, StringFileInfo, StringStruct, StringTable,
            VarFileInfo, VarStruct, VSVersionInfo,
        )
    except ImportError as ex:
        # Degrade instead of failing. The version resource only populates the
        # .exe's Properties -> Details tab; losing it costs a bit of polish,
        # and aborting a ~25-minute build over cosmetics would be a bad trade.
        print(f'WARNING: no Windows version resource ({ex}) — building without it.')
        return None

    return VSVersionInfo(
        ffi=FixedFileInfo(filevers=filevers, prodvers=filevers),
        kids=[
            StringFileInfo([StringTable('040904B0', [
                StringStruct('CompanyName', 'Anya Tennis'),
                StringStruct('FileDescription', 'Anya Tennis'),
                StringStruct('FileVersion', APP_VERSION),
                StringStruct('InternalName', 'AnyaTennis'),
                StringStruct('OriginalFilename', 'AnyaTennis.exe'),
                StringStruct('ProductName', 'Anya Tennis'),
                StringStruct('ProductVersion', APP_VERSION),
            ])]),
            # 0x0409 = US English, 1200 = UTF-16. Must agree with the '040904B0'
            # key above or Explorer ignores the whole block.
            VarFileInfo([VarStruct('Translation', [0x0409, 1200])]),
        ],
    )


_VERSION_RESOURCE = _version_resource() if _WINDOWS else None

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
    # macOS only. The hook works around a loader failure caused by the .app
    # bundle's Contents/Frameworks -> Contents/Resources symlinks, which no
    # one-folder Windows or Linux build has. Applying it anyway would force
    # cv2's sys.path[0] rewrite on platforms whose import already resolves
    # correctly — an unnecessary change to a working loader path.
    runtime_hooks=['rthook_cv2.py'] if sys.platform == 'darwin' else [],
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
    upx=_UPX,
    console=False,          # no terminal window on Windows
    version=_VERSION_RESOURCE,  # Windows only; None elsewhere
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
    upx=_UPX,
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
