"""
check_ps1_ascii.py - fail the build if a PowerShell script is not pure ASCII.

Run by build_windows.ps1 and by the Build Windows installer workflow, which
runs it FIRST so this costs seconds rather than the three minutes of pip
install that precede the build step.

The bug this exists to catch
----------------------------
Windows PowerShell 5.1 reads a .ps1 with no byte-order mark as ANSI - cp1252
on an English install - while pwsh 7 reads it as UTF-8. A UTF-8 em dash is the
bytes E2 80 94; decoded as cp1252 that is three characters ending in U+201D
RIGHT DOUBLE QUOTATION MARK, which 5.1 happily treats as a closing quote. So a
line that reads perfectly in every editor becomes

    Unexpected token 'only' in expression or statement.
    The string is missing the terminator: ".

reported against a *different* line than the one containing the character.
Nothing about the message points at encoding.

This has now bitten this project three times: the court-calibration window
title (0.1.0-beta.9), the calibration console prompt (a7c8b97), and the first
CI run of fetch_ffmpeg.ps1. Each time the character was invisible, the error
was misleading, and it only reproduced under one of the two interpreters.

A BOM would also fix it, but BOMs are easy to strip by accident (an editor, a
`Set-Content`, a patch tool) and their absence is silent. Requiring ASCII is
the property that cannot rot.

Comments are checked too, not just string literals. A misdecoded comment is
usually harmless, but "usually" depends on whether the mojibake happens to
contain a quote or a backtick, which is not a thing to reason about per line.
"""

import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    failures = []

    scripts = sorted(HERE.glob("*.ps1"))
    if not scripts:
        print("error: no .ps1 files found next to this script", file=sys.stderr)
        return 1

    for path in scripts:
        raw = path.read_bytes()
        if raw[:3] == b"\xef\xbb\xbf":
            # Not fatal - a BOM makes 5.1 read UTF-8 correctly - but it means
            # the file is relying on the fragile mechanism instead of the
            # robust one, so say so.
            print(f"  note  {path.name} has a UTF-8 BOM; ASCII is still required")

        text = raw.decode("utf-8", errors="replace")
        bad = []
        for lineno, line in enumerate(text.splitlines(), 1):
            for col, ch in enumerate(line, 1):
                if ord(ch) > 127:
                    name = unicodedata.name(ch, "unnamed")
                    bad.append(
                        f"{path.name}:{lineno}:{col}: U+{ord(ch):04X} {name} ({ch!r})"
                    )
        if bad:
            failures.extend(bad)
        else:
            print(f"  ok    {path.name}")

    if failures:
        print("\nerror: non-ASCII characters in PowerShell scripts:",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print("\nWindows PowerShell 5.1 reads these files as cp1252, which "
              "turns a UTF-8 em dash into a stray quote and breaks parsing "
              "with an error pointing at the wrong line. Replace them with "
              "ASCII equivalents (- for dashes, plain text for box drawing).",
              file=sys.stderr)
        return 1

    print(f"  all {len(scripts)} PowerShell script(s) are pure ASCII")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
