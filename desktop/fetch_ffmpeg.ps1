# fetch_ffmpeg.ps1 - put the static ffmpeg that gets bundled into the Windows
# build at desktop\vendor\<arch>\ffmpeg.exe.  The counterpart to fetch_ffmpeg.sh.
#
#   .\fetch_ffmpeg.ps1
#
# ASCII ONLY, deliberately - no em dashes, no box drawing, nothing above 0x7F.
# Windows PowerShell 5.1 reads a .ps1 with no BOM as ANSI (cp1252), so a UTF-8
# em dash arrives as three characters ending in a smart quote, and 5.1 then
# reports "the string is missing the terminator" on a line that looks fine in
# every editor. That is how the first run of this script failed in CI. pwsh 7
# defaults to UTF-8 and would have been fine, which is exactly what makes the
# bug intermittent and worth designing out rather than remembering.
# desktop/check_ps1_ascii.py enforces this, and the workflow runs it first.
#
# Why bundle at all, when Windows testers can install their own: because
# "their own" is a variable the app cannot see. preflight.ensure_ffmpeg only
# proves *an* ffmpeg is on PATH - it says nothing about which build, which
# version, or which encoders it has. A tester whose ffmpeg cannot transcode
# the source produces a proxy that never gets built, and before the decode
# guard in pipeline/utilities.py existed that failure was silent: proxy.py
# printed a WARN into a console a windowed build does not own, returned the
# SOURCE path, and every pass then decoded a 2.7K GoPro file directly through
# whatever video backend OpenCV had. That is how a 531-second match came back
# as 0.9 seconds of telemetry and an empty reel.
#
# Shipping a known build makes the ffmpeg on the far end of that pipe the one
# the release was tested against, on every machine.
#
# Why gyan.dev rather than the sources fetch_ffmpeg.sh uses: neither of those
# publishes a Windows binary. gyan.dev's "essentials" build is the standard
# redistributable Windows FFmpeg - configured --enable-gpl --enable-version3
# (so, GPLv3) with libx264, which is what proxy.py encodes with. It is NOT
# --enable-nonfree; that is verified below rather than assumed, because a
# nonfree build cannot be redistributed by anyone at any price and the
# fetch_ffmpeg.sh header documents two upstreams that turned out to be
# exactly that trap.
#
# vendor\ is gitignored: an 84 MB binary has no business in git history, and
# this script plus the pinned SHA-256 pair make it reproducible.
#
# Idempotent - a no-op once the binary is in place and verifies.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# platform.machine() is what rally_app.spec uses to pick the vendor directory,
# and it reports the PROCESS architecture. Match it here so a build under an
# x86 Python on an x64 host still finds the binary it bundled.
$arch = if ($env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE } else { 'AMD64' }
if ($arch -ne 'AMD64') {
    throw "Unsupported architecture '$arch': only AMD64 is published by the pinned upstream. Add an ARM64 source here before building on one."
}

# GitHub release assets are immutable once published, which is what makes this
# pinnable. If this 404s, upstream has rotated the release: pick the current
# version, re-record BOTH hashes, and re-check the buildconf below.
$url           = 'https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip'
$archiveSha256 = 'FA7D4D7E795DB0E2503F49F105F46ED5852386F0CFDD819899BE3B65EBDE24FC'
$member        = 'ffmpeg-7.1-essentials_build/bin/ffmpeg.exe'
$binSha256     = '2CE797A0F88D7F067180338FB227F7B1928EA727BD9A4D7A1D022F7C52AF71A3'

$destDir = Join-Path 'vendor' $arch
$dest    = Join-Path $destDir 'ffmpeg.exe'

function Test-Vendored {
    if (-not (Test-Path $dest)) { return $false }
    (Get-FileHash -Path $dest -Algorithm SHA256).Hash -eq $binSha256
}

if (Test-Vendored) {
    Write-Host "==> $dest already present and verified (sha256 ok)" -ForegroundColor Green
    return
}

New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    $archive = Join-Path $tmp 'ffmpeg.zip'
    Write-Host "==> Downloading ffmpeg for $arch (~88 MB)"
    # The progress bar makes Invoke-WebRequest an order of magnitude slower on
    # a large download, and a build log is not watching it anyway.
    $prevProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $url -OutFile $archive -MaximumRedirection 5
    } finally {
        $ProgressPreference = $prevProgress
    }

    Write-Host "==> Verifying archive checksum"
    $got = (Get-FileHash -Path $archive -Algorithm SHA256).Hash
    if ($got -ne $archiveSha256) {
        throw "Archive checksum mismatch.`n  expected $archiveSha256`n  got      $got"
    }

    Write-Host "==> Extracting $member"
    $unpacked = Join-Path $tmp 'unpacked'
    Expand-Archive -Path $archive -DestinationPath $unpacked -Force
    $extracted = Join-Path $unpacked ($member -replace '/', '\')
    if (-not (Test-Path $extracted)) {
        throw "$member is not in the archive: upstream changed its layout."
    }
    Copy-Item -Path $extracted -Destination $dest -Force

    if (-not (Test-Vendored)) {
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        throw "Extracted binary failed its checksum."
    }

    # Belt and braces on the thing that makes this binary shippable at all.
    # ffmpeg embeds its configure line, so this reads the real build rather
    # than trusting the release notes.
    $buildconf = & $dest -hide_banner -buildconf 2>&1 | Out-String
    if ($buildconf -match '--enable-nonfree') {
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        throw "$dest is built --enable-nonfree and CANNOT be redistributed."
    }
    if ($buildconf -notmatch '--enable-libx264') {
        Remove-Item $dest -Force -ErrorAction SilentlyContinue
        throw "$dest has no libx264, which proxy.py encodes with, so this build is unusable."
    }
    Write-Host "==> Licence check: no --enable-nonfree; libx264 present"

    # The GPL obliges us to ship the licence alongside the binary. assets\
    # holds the tracked copies; this puts the matching pair next to the binary
    # so rally_app.spec has one place to collect from, exactly as on macOS.
    Copy-Item 'assets\FFMPEG-LICENSE-AMD64.txt' (Join-Path $destDir 'FFMPEG-LICENSE.txt') -Force
    Copy-Item 'assets\COPYING.GPLv3'            (Join-Path $destDir 'COPYING.txt')        -Force

    Write-Host "==> Done: $dest" -ForegroundColor Green
    & $dest -hide_banner -version | Select-Object -First 1
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
