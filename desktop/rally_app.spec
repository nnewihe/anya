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
# ffmpeg IS bundled on macOS (fetch_ffmpeg.sh) and on Windows
# (fetch_ffmpeg.ps1): the static binary lands next to everything else —
# Contents/Frameworks in the .app, _internal\ in the Windows one-folder build —
# and preflight.repair_path() puts that directory first on PATH so the
# pipeline's `subprocess.run(["ffmpeg", ...])` calls find it without any of
# them changing.  Linux builds still expect ffmpeg on PATH.
#
# Windows bundling is not a convenience: relying on the tester's own ffmpeg
# shipped a build where proxy.py's transcode failed, silently returned the
# SOURCE path, and every pass then decoded a 2.7K GoPro file through whatever
# backend OpenCV had — 531 seconds of match came back as 0.9 seconds of
# telemetry.  See the fetch_ffmpeg.ps1 header.
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

# Static ffmpeg, fetched by fetch_ffmpeg.sh (macOS) or fetch_ffmpeg.ps1
# (Windows).  Declared as a *binary* rather than a data file so PyInstaller
# keeps the executable bit; it lands wherever sys._MEIPASS points —
# Contents/Frameworks in the .app, _internal\ in the Windows one-folder build.
# Guarded so a Linux build (which vendors nothing) still works.
#
# The architecture is the one this Python is running as, NOT the host's:
# the Intel build runs PyInstaller under an x86_64 interpreter via Rosetta, so
# platform.machine() is what decides which vendored binary belongs in this
# bundle.  Getting this wrong ships an app that needs Rosetta on a Mac that
# has none — the exact prompt bundling ffmpeg exists to avoid.  The same call
# reports AMD64 on 64-bit Windows, which is the directory fetch_ffmpeg.ps1
# writes to.
import platform
_ARCH = platform.machine()
_FFMPEG = Path('vendor') / _ARCH / ('ffmpeg.exe' if sys.platform == 'win32'
                                    else 'ffmpeg')
_binaries = []
# ffmpeg is GPL, so its licence has to accompany the binary. Taken from
# vendor/<arch>/, which the fetch script populates with the pair matching THIS
# build — the arm64, Intel and Windows binaries come from three different
# upstreams (GPLv2 for arm64, GPLv3 for the other two), so shipping one fixed
# licence would be wrong for two of the three. Also copied to the DMG root
# (make_dmg.sh) where a tester can see it; this copy is so it still travels
# with a bare .app or the installed Windows folder.
#
# Conditional for the same reason _binaries is: these files only exist once
# the fetch script has run, and Linux has no fetch script. Listing them
# unconditionally fails that build outright ("Unable to find
# vendor/.../FFMPEG-LICENSE.txt").
_license_datas = []
_VENDORS_FFMPEG = sys.platform in ('darwin', 'win32')
if _VENDORS_FFMPEG and _FFMPEG.is_file():
    _binaries.append((str(_FFMPEG), '.'))
    _license_datas = [
        (f'vendor/{_ARCH}/FFMPEG-LICENSE.txt', 'licenses'),
        (f'vendor/{_ARCH}/COPYING.txt', 'licenses'),
    ]
elif _VENDORS_FFMPEG:
    _fetch = ('.\\fetch_ffmpeg.ps1' if sys.platform == 'win32'
              else f'./fetch_ffmpeg.sh {_ARCH}')
    raise SystemExit(
        f"{_FFMPEG} is missing — run {_fetch} first.\n"
        "Building without it ships an app that depends on whatever ffmpeg the "
        "tester happens to have: it dies at the final render on a machine with "
        "none, and on a machine with a build that cannot transcode the source "
        "it silently analyses only the first second of the video."
    )

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
    binaries=_binaries,
    datas=[
        # Weights, at the paths the pipeline resolves relative to its own
        # __file__ (pipeline/models, walking/outputs).
        ('../pipeline/models/ball_best.pt',      'pipeline/models'),
        ('../pipeline/models/yolo26n.pt',        'pipeline/models'),
        ('../pipeline/models/yolov8n-pose.pt',   'pipeline/models'),
        # Near-serve trophy-pose detector, lazy-loaded on the first ARMED
        # window by anya_base.trophy_model.  Missing from this list until
        # beta.5, which meant the packaged app raised on the first near-side
        # serve candidate while running from source was fine.
        ('../pipeline/models/trophy_best.pt',     'pipeline/models'),
        ('../walking/outputs/walking_model.joblib', 'walking/outputs'),
        # The 15 Hz walking model.  reel._walk_model_15hz() returns None when
        # this file is absent and the fast point-end path silently falls back
        # to the 30 Hz model — which reel.py's own docstring measures as
        # costing a real point end (Data/21 walk onsets 8/12 -> 7/12).  That
        # fallback has been firing in every packaged build since beta.2, when
        # fast_end became the default, and only in packaged builds.
        ('../walking/outputs/walking_model_15hz.joblib', 'walking/outputs'),
        # Brand logo (shared with the mobile app) — resolved by app._logo_path()
        ('../mobile/assets/images/anya_logo.png', 'assets'),
        # Montserrat, burned into the Scoreboard tab's rendered overlay —
        # resolved by pipeline.scoreboard_reel.render.find_font()
        ('assets/fonts/Montserrat-SemiBold.ttf', 'assets/fonts'),
        ('assets/fonts/Montserrat-Bold.ttf', 'assets/fonts'),
        # ffmpeg licences — macOS and Windows; see _license_datas above.
        *_license_datas,
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
        # The pickled estimator also names sklearn._loss.loss (the fitted
        # HalfBinomialLoss), sklearn._loss.link and sklearn.preprocessing._label
        # (the LabelEncoder for classes_) — none of them reachable from
        # `import sklearn.ensemble` by static analysis.  sklearn._loss._loss is
        # the Cython extension behind walking.predict._alias_sklearn_loss; see
        # that docstring for why the pickle asks for it under the bare name.
        'sklearn._loss',
        'sklearn._loss.loss',
        'sklearn._loss.link',
        'sklearn._loss._loss',
        'sklearn.preprocessing',
        'sklearn.preprocessing._label',
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
        # NOTE matplotlib is NOT excluded. It is a declared hard dependency of
        # ultralytics (install_requires: matplotlib>=3.3.0), and excluding a
        # declared dependency only ever worked by luck. ultralytics 8.4.x added
        # a `semantic` task whose train.py imports matplotlib.pyplot at module
        # scope, and models/yolo/__init__.py imports `semantic` eagerly — so
        # from that release on, `import ultralytics` imports matplotlib
        # unconditionally and the exclusion turns into an immediate
        # "No module named 'matplotlib'" at app startup.
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
    # Pinned to the interpreter's own architecture rather than left None, so
    # PyInstaller fails loudly if a dependency wheel in the environment is the
    # wrong arch — on the Intel build that is the difference between a clear
    # build error and a bundle that dies on a tester's Mac.
    target_arch=_ARCH if sys.platform == 'darwin' else None,
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
    # UPX rewrites a Mach-O in place. On the vendored ffmpeg that both breaks
    # the binary and invalidates the signature build_macos.sh applies to it,
    # and the failure only shows up at the very end of a render. UPX isn't
    # normally installed on macOS so upx is usually a no-op there — this makes
    # sure a build machine that happens to have it doesn't ship a broken app.
    # (On Windows _UPX is False outright; see its definition.)
    upx_exclude=['ffmpeg'],
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
            # Big Sur is the floor for both builds: it's the first release on
            # Apple silicon, and on Intel it's what the oldest wheels in
            # requirements-intel.txt (PyQt6 6.5 / Qt 6.5 LTS) support. Declared
            # so an older Mac refuses to launch it with a clear message rather
            # than dying on a dyld symbol error.
            'LSMinimumSystemVersion': '11.0',
        },
    )
