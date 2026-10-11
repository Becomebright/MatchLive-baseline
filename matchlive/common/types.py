"""Per-frame perception records shared across the package.

Boxes are ltwh (left, top, width, height) in pixels of the frame they came from; other box
formats exist only inside the module that needs them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    role: str                                   # "ball" | "goalkeeper" | "player"
    bbox_ltwh: tuple[float, float, float, float]
    score: float
    class_id: int                               # raw detector class id


@dataclass(frozen=True)
class Track:
    # A track is identified by (shot_id, track_id), never by track_id alone:
    # tracking restarts at every shot cut, so the same track_id is re-issued in each shot.
    # Shot ids restart every half, so outside perception (which runs one half at a time)
    # the key is (half, shot_id, track_id).
    # The role is an attribute, not part of the key: it is the track's majority detected
    # class so far and may change.
    shot_id: int
    role: str                                   # "goalkeeper" | "player" (the ball is not tracked)
    track_id: int
    bbox_ltwh: tuple[float, float, float, float]
    score: float
    # Identity, filled in by perception.identity.IdentityResolver.
    team: str | None = None                     # "home" | "away" | None (no team vote yet)
    player_id: str | None = None                # roster hash, None until identified

    @property
    def key(self) -> tuple[int, int]:
        return (self.shot_id, self.track_id)
