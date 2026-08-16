"""PyInstaller runtime hook — fix cv2's loader inside a frozen bundle.

Two unrelated problems, one per platform, both of which have to be handled
before cv2 is first imported — hence a runtime hook rather than code in main().

macOS: the bootstrap re-import
------------------------------
cv2/__init__.py runs a bootstrap() that re-imports "cv2" to swap the Python
package for the native extension. To make that second import resolve to the
extension, it inserts its own directory into sys.path — but only at index 0
if it detects that sys.path[0] is already the package's parent; otherwise at
index 1.

In a PyInstaller macOS .app, files under Contents/Frameworks are symlinks to
Contents/Resources, so cv2's realpath(__file__) puts LOADER_DIR (and thus
its parent) under Resources, while sys.path[0] is Frameworks. The paths
never match, cv2 concludes no workaround is needed, and inserts at index 1 —
leaving Contents/Frameworks at index 0, where the re-import finds the cv2
*package* again rather than the extension. __init__ re-executes, bootstrap()
re-enters, and it aborts with "recursion is detected during loading of cv2
binary extensions".

cv2 supports forcing the index-0 insertion via this flag, which makes the
re-import resolve to cv2.abi3.so as intended.

Windows: the FFmpeg backend DLL
-------------------------------
OpenCV's FFmpeg backend is not statically linked on Windows. It lives in a
separate `opencv_videoio_ffmpeg*.dll` that videoio loads BY NAME at runtime,
searching the cv2 module directory and then PATH. In a PyInstaller bundle
neither resembles a pip install, and if the DLL is not found OpenCV does not
raise — it quietly falls back to Media Foundation, which opens a GoPro file,
reports a correct frame count from the container, and then dies partway
through the first sequential read. That shipped: a tester's 531-second match
decoded 53 frames and the app produced an empty reel without an error.

OpenCV >= 4.5.4 reads OPENCV_FFMPEG_DLL_DIR and looks there first, so
pointing it at wherever PyInstaller actually put the DLL removes the guess.
Set only when the file is really there, so that on a bundle where it went
missing the variable's absence is itself a signal — pipeline/videoio.py
prints its value when it has to fall back to another backend.
"""

import os
import sys
from pathlib import Path

# darwin only, deliberately. Forcing the index-0 rewrite on a platform whose
# import already resolves correctly would be an unnecessary change to a
# working loader path — the Windows one-folder build has no symlinks and does
# not have this problem.
if sys.platform == "darwin":
    sys.OpenCV_REPLACE_SYS_PATH_0 = True

if sys.platform == "win32":
    _root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    # cv2/ is where a pip layout puts it and where PyInstaller's cv2 hook
    # normally lands it; the bundle root is where a `binaries` entry with a
    # '.' destination would go. Check both rather than assume either.
    for _d in (_root / "cv2", _root):
        if next(_d.glob("opencv_videoio_ffmpeg*.dll"), None) is not None:
            os.environ["OPENCV_FFMPEG_DLL_DIR"] = str(_d)
            break
