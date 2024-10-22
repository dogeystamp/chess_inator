"""
Print bitboard integer.

Usage:

    python3 contrib/bitboard_print.py 71213177697730560

"""

from sys import argv


def chunks(s: str, n: int = 8):
    for i in range(len(s) // n):
        yield s[i * n : (i + 1) * n]


val = bin(int(argv[1]))[2:].zfill(64)
print("\n".join(''.join(reversed(c)) for c in chunks(val)))
