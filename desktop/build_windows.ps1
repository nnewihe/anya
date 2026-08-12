# build_windows.ps1 — build "Anya Tennis" for Windows and wrap it in a
# single setup .exe for beta testers.  The counterpart to build_macos.sh.
#
# Usage (from a PowerShell prompt):
#   cd desktop
#   .\build_windows.ps1
#
# Options:
#   -SkipInstaller   stop after PyInstaller, leaving dist\AnyaTennis\
#   -KeepBuild       don't delete build\ and dist\ first (faster re-runs,
#                    but a stale build\ is the usual cause of "I fixed that
#                    already" bugs — only use it while iterating)
#
# Prerequisites:
#   - Python >= 3.12.1  (see the guard below)
#   - pip install -r requirements.txt
#   - Inno Setup 6.3+   (winget install JRSoftware.InnoSetup)
#
# NOT signed.  Without a code-signing certificate, SmartScreen will warn on
# first run — see the Windows section of desktop/README.md for what to tell
# testers.

[CmdletBinding()]
param(
    [switch]$SkipInstaller,
    [switch]$KeepBuild
)

# Stop on the first error, and make native-command failures (pyinstaller, iscc)
# count as errors too — PowerShell otherwise happily carries on past a failed
# .exe and reports overall success, which would ship a half-built installer.
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

# ── Python version guard ───────────────────────────────────────────────────
# Python 3.12.0 ships a CPython bug (cpython#110543) where code.replace()
# drops the CO_FAST_HIDDEN flag used by PEP 709 inlined comprehensions.
# PyInstaller calls code.replace() on every module, so on 3.12.0 the build
# SUCCEEDS and the app then dies at startup with a NameError out of
# torch/_numpy/_ufuncs.py. Fail loudly here rather than shipping that.
# (Same check as build_macos.sh.)
& python -c "import sys; sys.exit(0 if sys.version_info[:3] != (3, 12, 0) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12.0 cannot build this app (cpython#110543 breaks torch/scipy in the frozen bundle). Use Python >= 3.12.1."
}

$appVersion = (& python -c "import sys; sys.path.insert(0, '.'); from version import APP_VERSION; print(APP_VERSION)").Trim()
if (-not $appVersion) { throw "Could not read APP_VERSION from version.py" }

# The Win32 version resource takes four integers and nothing else, so strip
# any "-beta.N" suffix and pad to four parts. "0.1.0-beta.3" -> "0.1.0.0".
$numericParts = ($appVersion -split '-')[0] -split '\.'
while ($numericParts.Count -lt 4) { $numericParts += '0' }
$versionNumeric = ($numericParts[0..3]) -join '.'

Write-Host "==> Building Anya Tennis $appVersion (version resource $versionNumeric)" -ForegroundColor Cyan

# ── Sanity check: the pipeline package must be importable ──────────────────
# app.py imports pipeline.scoreboard_reel and pipeline.rally_reel. If a
# submodule is missing from the checkout, PyInstaller records the failed
# import as a warning and builds a bundle that dies on launch. Catch it here,
# where the error is one readable line.
Write-Host "==> Checking the app imports cleanly from source"
Invoke-Checked -Exe 'python' -What 'Import check' -Arguments @(
    '-c',
    "import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.'); import pipeline.rally_reel, pipeline.scoreboard_reel; print('imports OK')"
)

# ── Clean ──────────────────────────────────────────────────────────────────
if (-not $KeepBuild) {
    Write-Host "==> Cleaning previous build"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist
}

# ── PyInstaller ────────────────────────────────────────────────────────────
Write-Host "==> Running PyInstaller (expect ~2 GB out and several minutes)"
Invoke-Checked -Exe 'pyinstaller' -Arguments @('--noconfirm', 'rally_app.spec') -What 'PyInstaller'

$exePath = Join-Path 'dist\AnyaTennis' 'AnyaTennis.exe'
if (-not (Test-Path $exePath)) {
    throw "PyInstaller reported success but $exePath is missing"
}

# A one-folder build that is only a few MB means COLLECT silently dropped the
# heavy dependencies, which produces an installer that fails on first launch
# rather than at build time. torch alone is well over 500 MB.
$sizeMb = [math]::Round((Get-ChildItem -Recurse -File 'dist\AnyaTennis' | Measure-Object -Property Length -Sum).Sum / 1MB)
Write-Host "==> Bundle size: $sizeMb MB"
if ($sizeMb -lt 300) {
    throw "dist\AnyaTennis is only $sizeMb MB — torch/ultralytics were not collected. Check the PyInstaller warnings."
}

if ($SkipInstaller) {
    Write-Host "==> Done (installer skipped): dist\AnyaTennis\" -ForegroundColor Green
    exit 0
}

# ── Inno Setup ─────────────────────────────────────────────────────────────
# iscc isn't added to PATH by the installer, so check PATH first and then the
# two standard install roots before giving up.
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $iscc = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $iscc) {
    throw "Inno Setup not found. Install it with: winget install JRSoftware.InnoSetup"
}

Write-Host "==> Building the installer with $iscc"
Invoke-Checked -Exe $iscc -What 'Inno Setup' -Arguments @(
    "/DAppVersion=$appVersion",
    "/DVersionNumeric=$versionNumeric",
    'installer.iss'
)

$setup = "dist\AnyaTennis-Setup-$appVersion.exe"
if (-not (Test-Path $setup)) { throw "Inno Setup reported success but $setup is missing" }
$setupMb = [math]::Round((Get-Item $setup).Length / 1MB)

Write-Host ""
Write-Host "==> Done: $setup ($setupMb MB)" -ForegroundColor Green
Write-Host "    Unsigned — testers will see a SmartScreen warning on first run." -ForegroundColor Yellow
Write-Host "    Tell them: More info -> Run anyway." -ForegroundColor Yellow
