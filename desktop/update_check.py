"""
update_check.py — tell the tester when a newer build has been released.

Beta testers install a DMG once and then have no way to learn that a newer
one exists; `version.py` records that beta.3 shipped with a crash-on-
completion bug, and anyone holding that build would have sat on it
indefinitely. This asks GitHub Releases, on launch, whether there is something
newer, and hands the answer to app.py to show as a banner.

Deliberately *not* a real auto-updater. A Sparkle-style download-and-swap
would mean embedding an updater framework in a PyInstaller bundle, a second
signing key and a 320 MB background download; the banner plus a Download
button costs one HTTP GET and gets a tester onto the new build in about the
same wall-clock time.

Privacy: the app's header promises "Runs 100% on this computer — your video is
never uploaded", and this must not undermine that. The request is a plain
unauthenticated GET with no query string, no body, no identifier of any kind —
GitHub learns an IP fetched a public release list, which is what downloading
the DMG in the first place told it. Set ANYA_NO_UPDATE_CHECK=1 to skip it.
"""

import json
import os
import urllib.request
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from applog import logger
from version import APP_VERSION

# Releases API rather than /releases/latest: mobile/ versions independently and
# may get its own releases in this same repo, and /releases/latest would then
# start handing the desktop app a Flutter version number. Filtering on the tag
# prefix keeps the two channels separate no matter what else gets published.
_RELEASES_URL = "https://api.github.com/repos/nnewihe/anya/releases"
_TAG_PREFIX = "desktop-v"

# Long enough for a slow connection, short enough that the check has given up
# long before a tester could have picked a video and pressed Go.
_TIMEOUT_S = 5


def _parse(version: str) -> Optional[Tuple]:
    """Order two Anya version strings, e.g. '0.1.0-beta.4' < '0.1.0-beta.10'.

    `packaging` would do this (and correctly reads our strings as PEP 440
    prereleases), but it is only present here as a transitive dependency of
    torch/ultralytics — relying on it would make the update check breakable by
    an unrelated dependency bump. It's a dotted number triple with an optional
    `-beta.N`, so parse it directly.

    A release with no `-beta` suffix sorts ABOVE every beta of the same triple,
    which is what makes 0.1.0 read as newer than 0.1.0-beta.9.
    """
    try:
        core, _, pre = version.strip().partition("-")
        nums = tuple(int(p) for p in core.split("."))
        if len(nums) != 3:
            # Anything but major.minor.patch isn't ours; comparing tuples of
            # different lengths would give a confidently wrong answer.
            return None
        if not pre:
            return nums + (1, 0)
        label, _, n = pre.partition(".")
        return nums + (0, int(n or 0))
    except (ValueError, AttributeError):
        return None


def _newer_release() -> Optional[Tuple[str, str]]:
    """(version, html_url) of the newest desktop release, or None.

    None covers every uninteresting case identically: we're current, we're
    ahead (a dev build), offline, rate-limited, GitHub is down, the JSON
    changed shape. The caller cannot act differently on any of them.
    """
    req = urllib.request.Request(
        _RELEASES_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"AnyaTennis/{APP_VERSION}",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        releases = json.load(resp)

    mine = _parse(APP_VERSION)
    if mine is None:
        return None

    best = None
    for rel in releases:
        if rel.get("draft"):
            continue
        tag = rel.get("tag_name") or ""
        if not tag.startswith(_TAG_PREFIX):
            continue
        ver = tag[len(_TAG_PREFIX):]
        parsed = _parse(ver)
        if parsed is None or parsed <= mine:
            continue
        if best is None or parsed > best[0]:
            best = (parsed, ver, rel.get("html_url") or "")

    return (best[1], best[2]) if best else None


class UpdateChecker(QThread):
    """Runs `_newer_release` off the GUI thread.

    A QThread for the same reason highlight_tab._Worker is one: five seconds of
    blocking network on the main thread would freeze the window during launch,
    which on macOS is long enough to draw the spinning beachball. Keep a
    reference on the owner until `finished` fires — dropping the last one from
    a slot while the OS thread is still running is what caused the beta.4
    SIGABRT (see HighlightReelTab._release_worker).
    """

    update_available = pyqtSignal(str, str)  # (version, html_url)

    def run(self):
        # One blanket except: a failed update check must never be able to
        # affect a launch, and there is no failure here worth distinguishing
        # to a tester. Logged at debug so it's still in app.log if we ever
        # need to know why nobody is seeing the banner.
        try:
            found = _newer_release()
        except Exception as exc:
            logger().debug("update check failed: %s", exc)
            return
        if found:
            logger().info("update available: %s", found[0])
            self.update_available.emit(*found)


def check_for_updates(parent, on_available) -> Optional[UpdateChecker]:
    """Start a background check. Returns the thread so the caller can hold it.

    Returns None when the check is disabled, so callers get the same "nothing
    to hold onto" shape they'd get from an opt-out.
    """
    if os.environ.get("ANYA_NO_UPDATE_CHECK"):
        logger().info("update check disabled by ANYA_NO_UPDATE_CHECK")
        return None

    checker = UpdateChecker(parent)
    checker.update_available.connect(on_available)
    checker.finished.connect(checker.deleteLater)
    checker.start()
    return checker
