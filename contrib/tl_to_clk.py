# /usr/bin/env python

"""
Converts PGN files from fast-chess's `tl` time left notation to the standard `%clk` clock time notation.

May be buggy; other comments may break this script.
Takes stdin and converts to stdout.
"""

import sys
import re

pgn_value = sys.stdin.read()


def convert(m: re.Match[str]) -> str:
    seconds_total = float(m.group(1))
    mins, secs = divmod(seconds_total, 60)
    hrs, mins = divmod(mins, 60)

    secs = round(secs, 4)
    mins = round(mins)
    hrs = round(hrs)

    return f"{{ [%clk {hrs}:{mins:02}:{secs}] }}"

pgn_value = re.sub(
    pattern=r"{book}",
    repl="",
    string=pgn_value,
)

print(
    re.sub(
        pattern=r"{.*?tl=(.*?)s.*?}",
        repl=convert,
        string=pgn_value,
    )
)
