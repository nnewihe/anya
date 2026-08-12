#!/usr/bin/env bash
# setup_intel_env.sh — create the x86_64 Python environment the Intel build
# needs, at desktop/.venv-intel/.
#
# Run once (and again whenever requirements-intel.txt changes):
#   cd desktop && ./setup_intel_env.sh
#
# Why this is not just "make a venv": an Intel build has to run PyInstaller
# under an x86_64 interpreter, and there usually isn't one on an Apple silicon
# Mac. The obvious routes both fail:
#
#   * `arch -x86_64 /usr/bin/python3` is x86_64, but it's Python 3.9 and can't
#     be pip-installed into.
#   * Building one with `pyenv install` under Rosetta fails twice over: it
#     links the arm64 Homebrew openssl/readline (wrong architecture), and with
#     Homebrew skipped it produces a Python with no `_lzma` — which torchvision
#     imports at module load, so the whole stack is unusable. Supplying lzma
#     means building xz from source for x86_64 first.
#
# So this downloads a prebuilt standalone x86_64 CPython, which ships a
# complete stdlib including _lzma and its own OpenSSL. It never touches the
# system or Homebrew Python, and lives entirely inside desktop/.
#
# Rosetta is required on an Apple silicon build machine:
#   softwareupdate --install-rosetta --agree-to-license
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION="3.12.13"
PY_BUILD="20260807"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_BUILD}/cpython-${PY_VERSION}+${PY_BUILD}-x86_64-apple-darwin-install_only.tar.gz"

TOOLCHAIN=".toolchain-intel"
VENV=".venv-intel"

if [ "$(uname -m)" = "arm64" ] && ! arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    echo "error: Rosetta is required to build the Intel version on this Mac." >&2
    echo "       Install it with:" >&2
    echo "         softwareupdate --install-rosetta --agree-to-license" >&2
    exit 1
fi

if [ ! -x "$TOOLCHAIN/python/bin/python3" ]; then
    echo "==> Downloading standalone x86_64 CPython $PY_VERSION"
    rm -rf "$TOOLCHAIN"
    mkdir -p "$TOOLCHAIN"
    curl -fL --retry 3 --max-time 600 -o "$TOOLCHAIN/python.tar.gz" "$PY_URL"
    tar -xzf "$TOOLCHAIN/python.tar.gz" -C "$TOOLCHAIN"
    rm -f "$TOOLCHAIN/python.tar.gz"
fi

PY="$TOOLCHAIN/python/bin/python3"
if [ "$(lipo -archs "$PY")" != "x86_64" ]; then
    echo "error: $PY is not x86_64 — refusing to build an Intel app with it" >&2
    exit 1
fi

echo "==> Creating $VENV"
rm -rf "$VENV"
arch -x86_64 "$PY" -m venv "$VENV"
arch -x86_64 "$VENV/bin/pip" install --upgrade pip

echo "==> Installing requirements-intel.txt (large — torch alone is ~150 MB)"
arch -x86_64 "$VENV/bin/pip" install -r requirements-intel.txt

echo "==> Verifying the stack imports under x86_64"
# torchvision is the canary: it imports lzma at module load, which is exactly
# what a hand-built Python tends to be missing, and ultralytics imports
# torchvision transitively — so a broken stdlib surfaces here rather than
# halfway through a build.
arch -x86_64 "$VENV/bin/python" - <<'PY'
import platform
import numpy, torch, torchvision, cv2, ultralytics, sklearn, scipy, joblib
from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
assert platform.machine() == "x86_64", platform.machine()
print(f"  arch        {platform.machine()}")
print(f"  numpy       {numpy.__version__}")
print(f"  torch       {torch.__version__}")
print(f"  torchvision {torchvision.__version__}")
print(f"  opencv      {cv2.__version__}")
print(f"  ultralytics {ultralytics.__version__}")
print(f"  PyQt6       {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})")
PY

echo "==> Done. Build the Intel app with: ./build_macos.sh x86_64"
