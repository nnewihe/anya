# build_windows.ps1 - build "Anya Tennis" for Windows and wrap it in a
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
#                    already" bugs - only use it while iterating)
#
# Prerequisites:
#   - Python >= 3.12.1  (see the guard below)
#   - pip install -r requirements.txt
#   - Inno Setup 6.3+   (winget install JRSoftware.InnoSetup)
#
# NOT signed.  Without a code-signing certificate, SmartScreen will warn on
# first run - see the Windows section of desktop/README.md for what to tell
# testers.

[CmdletBinding()]
param(
    [switch]$SkipInstaller,
    [switch]$KeepBuild,
    [switch]$SkipSmokeTest,
    # Long enough for a cold ~2 GB bundle to unpack its imports and build both
    # tabs on a slow runner; short enough not to pad every build noticeably.
    [int]$SmokeTestSeconds = 45
)

# Stop on the first error, and make native-command failures (pyinstaller, iscc)
# count as errors too - PowerShell otherwise happily carries on past a failed
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

# -- Python version guard ---------------------------------------------------
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

# -- PowerShell scripts must be pure ASCII ----------------------------------
# See check_ps1_ascii.py. Windows PowerShell 5.1 reads a BOM-less .ps1 as
# cp1252 and pwsh 7 reads it as UTF-8, so one stray em dash parses under one
# interpreter and not the other, with an error naming the wrong line. Checked
# here as well as in CI because a developer running this locally may well be
# on 5.1, which is the interpreter that breaks.
Write-Host "==> Checking the PowerShell scripts are ASCII"
Invoke-Checked -Exe 'python' -Arguments @('check_ps1_ascii.py') -What 'PowerShell ASCII check'

# -- Sanity check: the pipeline package must be importable ------------------
# app.py imports pipeline.scoreboard_reel and pipeline.rally_reel. If a
# submodule is missing from the checkout, PyInstaller records the failed
# import as a warning and builds a bundle that dies on launch. Catch it here,
# where the error is one readable line.
Write-Host "==> Checking the app imports cleanly from source"
Invoke-Checked -Exe 'python' -What 'Import check' -Arguments @(
    '-c',
    "import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.'); import pipeline.rally_reel, pipeline.scoreboard_reel; print('imports OK')"
)

# -- Vendored ffmpeg --------------------------------------------------------
# Before PyInstaller, because rally_app.spec hard-fails if the binary is not
# there. Idempotent, so this is a checksum check on a re-run rather than an
# 88 MB download every build.
#
# Called IN-PROCESS, not as `powershell -File .\fetch_ffmpeg.ps1`. Spawning a
# child hardcodes an interpreter, and `powershell` is Windows PowerShell 5.1,
# which reads a BOM-less .ps1 as cp1252 while the pwsh 7 running this file
# reads it as UTF-8. That mismatch is what broke the first CI run of this
# step. `&` keeps one interpreter for the whole build; the ASCII check above
# means it would no longer matter either way, which is the belt to this
# brace. $ErrorActionPreference = 'Stop' turns any throw in there into a
# build failure, so this needs no Invoke-Checked wrapper.
Write-Host "==> Fetching the ffmpeg to bundle"
& .\fetch_ffmpeg.ps1

# -- Model paths ------------------------------------------------------------
# Handed a bare name like "yolov8n-pose.pt", ultralytics looks in the CWD and
# then DOWNLOADS the weights, ignoring the bundled copy - see
# check_model_paths.py, which exists because that shipped once and was
# invisible to anyone with a working internet connection. build_macos.sh has
# run this since; the Windows build had no equivalent gate.
Write-Host "==> Checking model defaults resolve to bundled files"
Invoke-Checked -Exe 'python' -Arguments @('check_model_paths.py') -What 'Model path check'

# -- Clean ------------------------------------------------------------
if (-not $KeepBuild) {
    Write-Host "==> Cleaning previous build"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist
}

# -- PyInstaller ------------------------------------------------------------
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
    throw "dist\AnyaTennis is only $sizeMb MB - torch/ultralytics were not collected. Check the PyInstaller warnings."
}

# -- OpenCV's FFmpeg backend DLL --------------------------------------------
# videoio loads this by name at runtime. When it is absent OpenCV does not
# fail - it drops to Media Foundation, which opens a GoPro file, reports the
# right frame count, and then stops decoding partway through. That is a silent
# wrong answer, so it is a build gate rather than something to find in the
# field. rthook_cv2.py points OPENCV_FFMPEG_DLL_DIR at whichever of these two
# directories it lands in; both are searched here for the same reason.
$ffmpegDll = Get-ChildItem -Path 'dist\AnyaTennis' -Recurse -File `
    -Filter 'opencv_videoio_ffmpeg*.dll' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $ffmpegDll) {
    throw "opencv_videoio_ffmpeg*.dll is not in dist\AnyaTennis - OpenCV would fall back to Media Foundation and silently truncate long videos. Check the PyInstaller cv2 hook and warn-rally_app.txt."
}
Write-Host "    OpenCV FFmpeg backend: $($ffmpegDll.FullName.Substring((Resolve-Path 'dist\AnyaTennis').Path.Length + 1))"

# -- Bundled ffmpeg.exe -----------------------------------------------------
if (-not (Test-Path 'dist\AnyaTennis\_internal\ffmpeg.exe')) {
    # PyInstaller's layout has moved between majors; look anywhere before
    # failing, so a version bump reports honestly instead of crying wolf.
    $anyFfmpeg = Get-ChildItem -Path 'dist\AnyaTennis' -Recurse -File `
        -Filter 'ffmpeg.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $anyFfmpeg) {
        throw "ffmpeg.exe is not in dist\AnyaTennis - the spec should have bundled it. Run .\fetch_ffmpeg.ps1 and rebuild."
    }
    Write-Host "    Bundled ffmpeg: $($anyFfmpeg.FullName)" -ForegroundColor Yellow
}

# -- Smoke test: does the thing actually start? -----------------------------
# A PyInstaller build that succeeds proves nothing about whether the app runs.
# The first Windows build packaged cleanly and then died instantly on
# "No module named 'matplotlib'" - an exclude in the spec that a newer
# ultralytics had turned into a hard dependency. That failure happens at
# import time, before applog installs its excepthook, so there is no log file
# and no dialog text to go on: the only signal is that the process exits.
#
# QT_QPA_PLATFORM=offscreen so this needs no desktop session and still
# exercises the whole chain - every import, plus building both tabs. If the
# app is alive after the timeout it got through startup; a process that has
# already exited failed, and its exit code and stderr say how.
if (-not $SkipSmokeTest) {
    Write-Host "==> Smoke-testing the built app (offscreen, ${SmokeTestSeconds}s)"
    $stdout = Join-Path $env:TEMP 'anya_smoke_out.txt'
    $stderr = Join-Path $env:TEMP 'anya_smoke_err.txt'
    $env:QT_QPA_PLATFORM = 'offscreen'
    # The launch update check is not what is under test here, and letting it
    # run makes the result depend on GitHub's API being reachable and
    # un-rate-limited from a CI runner. It already fails silently, but a build
    # gate should not have a network dependency it doesn't need.
    $env:ANYA_NO_UPDATE_CHECK = '1'
    try {
        $proc = Start-Process -FilePath (Resolve-Path $exePath) -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        if ($proc.WaitForExit($SmokeTestSeconds * 1000)) {
            # Exited on its own => it crashed. Nothing here ever exits early.
            $err = if (Test-Path $stderr) { Get-Content $stderr -Raw } else { '' }
            $out = if (Test-Path $stdout) { Get-Content $stdout -Raw } else { '' }
            Write-Host "--- stdout ---`n$out" -ForegroundColor DarkGray
            Write-Host "--- stderr ---`n$err" -ForegroundColor Red
            throw "The built app exited during startup (code $($proc.ExitCode)). It does not run - see the output above and build\rally_app\warn-rally_app.txt."
        }
        Write-Host "    still running after ${SmokeTestSeconds}s - startup OK"
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    } finally {
        Remove-Item env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        Remove-Item env:ANYA_NO_UPDATE_CHECK -ErrorAction SilentlyContinue
        Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
    }
}

if ($SkipInstaller) {
    Write-Host "==> Done (installer skipped): dist\AnyaTennis\" -ForegroundColor Green
    exit 0
}

# -- Inno Setup ------------------------------------------------------------
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
Write-Host "    Unsigned - testers will see a SmartScreen warning on first run." -ForegroundColor Yellow
Write-Host "    Tell them: More info -> Run anyway." -ForegroundColor Yellow
