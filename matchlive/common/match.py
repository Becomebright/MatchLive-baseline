"""One game of the challenge data: its two half videos and its pre-match facts.

A split's manifest (test.json / challenge.json, OSL JSON) lists one sample per half, keyed by its
`match_key` ("game1", ...) and `half`; each sample's video path is relative to the manifest's
folder (paths.data). The pre-match facts of each split's games come from
paths.prematch/<split>.json, since the manifests carry none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_manifest(config: dict[str, Any], split: str) -> dict[str, Any]:
    """The split's question file as published (OSL JSON)."""
    path = config["paths"]["data"] / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: download the challenge data first (README, Data)")
    return json.loads(path.read_text())


def games(manifest: Mapping[str, Any]) -> list[str]:
    """The games of a manifest, in order of first appearance ("game1", "game2", ...)."""
    return list(dict.fromkeys(s["metadata"]["match_key"] for s in manifest["data"]))


class Match:
    def __init__(self, config: dict[str, Any], split: str, game: str, manifest: Mapping[str, Any]):
        # `manifest` is the split's question file (load_manifest).
        self.split, self.game = split, game
        # Names this game's caches and outputs: "test/game1".
        self.key = f"{split}/{game}"
        # This game's samples (one per half, with its questions), in the manifest's order.
        self.samples = [s for s in manifest["data"] if s["metadata"]["match_key"] == game]
        halves = sorted(s["metadata"]["half"] for s in self.samples)
        if halves != [1, 2]:
            raise KeyError(f"{split}: expected both halves of {game!r}, found halves {halves}")
        self.videos: dict[int, Path] = {
            s["metadata"]["half"]: config["paths"]["data"] / next(i["path"] for i in s["inputs"] if i["type"] == "video")
            for s in self.samples}
        facts = json.loads((config["paths"]["prematch"] / f"{split}.json").read_text())
        if game not in facts:
            raise KeyError(f"no pre-match facts for {split}/{game} in {config['paths']['prematch']}")
        self.prematch: dict[str, Any] = prematch(facts[game])

    def video_path(self, half: int) -> Path:
        return self.videos[half]


def prematch(facts: Mapping[str, Any]) -> dict[str, Any]:
    """What is known before kick-off — the only match information the live system gets: the
    team names, kit colours, which goal each team defends in each half, and the two squads
    (shirt number, starting or substitute, and a starter's position). `facts` is one game of
    paths.prematch/<split>.json:

        {"home" | "away": {"name", "kit_color": {"name", "hex"}, ["goalkeeper_kit_color": {"name", "hex"}],
                           "defends": {"1": "left" | "right", "2": ...},
                           "players": [{"id", "name", "short_name", "long_name", "shirt_number",
                                        "starting", "position"}]}}

    `defends` is the side of the main camera's picture whose goal the team defends in that half.
    Returned as the package reads it: teams, jersey_colors ({kit: [name, hex]}, kit = home, away,
    and home_goalkeeper / away_goalkeeper when given), sides ({half: {team: side}}) and players
    (roster hash, team side, names, jersey number, starting flag, position)."""
    sides = ("home", "away")
    jersey_colors = {}
    for side in sides:
        jersey_colors[side] = [facts[side]["kit_color"]["name"], facts[side]["kit_color"]["hex"]]
        if facts[side].get("goalkeeper_kit_color"):
            kit = facts[side]["goalkeeper_kit_color"]
            jersey_colors[f"{side}_goalkeeper"] = [kit["name"], kit["hex"]]
    return {
        "teams": {side: facts[side]["name"] for side in sides},
        "jersey_colors": jersey_colors,
        "sides": {half: {side: facts[side]["defends"][str(half)] for side in sides} for half in (1, 2)},
        "players": [{"hash": p["id"], "team_side": side, "name": p["name"], "short_name": p["short_name"],
                     "long_name": p["long_name"], "jersey_number": str(p["shirt_number"]),
                     "starting": bool(p["starting"]), "position": p["position"]}
                    for side in sides for p in facts[side]["players"]],
    }
