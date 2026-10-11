"""Fixed windows over a half: no trigger, the whole match is analysed.

Windows of `length` seconds, each starting `length - overlap` after the previous one. An
event on the overlap is proposed by both windows; their candidates overlap in time with the
same type, so the consistency check matches them to one entry.
"""

from __future__ import annotations


def fixed_windows(duration: float, length: float, overlap: float) -> list[tuple[float, float]]:
    """[(start, end)] seconds covering [0, duration); the last window is cut at `duration`."""
    step = length - overlap
    if step <= 0:
        raise ValueError(f"window overlap ({overlap}) must be shorter than the window ({length})")
    out, start = [], 0.0
    while start < duration:
        out.append((start, min(start + length, duration)))
        if start + length >= duration:
            break
        start += step
    return out
