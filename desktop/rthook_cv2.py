"""PyInstaller runtime hook — fix cv2's loader inside a macOS .app bundle.

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
re-import resolve to cv2.abi3.so as intended. Must run before cv2 is first
imported, hence a runtime hook.
"""

import sys

sys.OpenCV_REPLACE_SYS_PATH_0 = True
