"""Event analysis: one API call per window proposes 0..n event candidates.

The model sees the window's frames sampled at `sample_fps`, each tracked player marked only
with a window-local ID label ("P3"), plus a text block saying who each ID is (from identity
voting), who is nearest the ball in each frame, and where the camera cuts. It is given no
memory summary: proposers see only pixels and perception, so they cannot echo the memory
and hide a contradiction.

It answers in JSON with the event type, the frame, the team, the participants by P-ID,
the evidence frames, and whether the event was seen in a replay. No confidence is asked
for: whether an event holds is decided by agreeing sources, not by the model's own score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from matchlive.common.types import Track
from matchlive.memory.types import EVENT_TYPES
from matchlive.perception.visualization import draw_ids, frame_jpeg


@dataclass(frozen=True)
class Candidate:
    type: str
    team: str | None                        # "home" | "away" | None
    second: float                           # when it is seen, within the half
    participants: tuple[tuple[str, tuple], ...]   # (role, track key (half, shot_id, track_id))
    evidence: tuple[float, ...]             # seconds of the frames the model cites
    replay: bool                            # seen in a replay, so it happened earlier

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Candidate":
        # The inverse of the __dict__ dump in windows.jsonl (runtime.runner.build_memory): JSON
        # turned the tuples into lists.
        return cls(d["type"], d["team"], d["second"],
                   tuple((role, tuple(key)) for role, key in d["participants"]), tuple(d["evidence"]), d["replay"])


@dataclass(frozen=True)
class Request:
    parts: list[str | bytes]                # text, then "Frame k (clock)" + JPEG for each frame
    pids: dict[str, tuple]                  # "P3" -> track key (half, shot_id, track_id)
    seconds: list[float]                    # second of each shown frame; frame k is seconds[k - 1]


INSTRUCTIONS = """You are analysing a football (soccer) broadcast, {window}.
Home: {home} ({home_kit} kit), defending the goal on the {home_side} of the main camera's picture this half.
Away: {away} ({away_kit} kit), defending the goal on the {away_side}.

Below are {n} frames sampled at {fps} per second, in order, each preceded by its number.
Tracked players carry an ID label (P1, P2, ...) that is only valid inside this request;
the text after these instructions says who each ID is, as far as the tracker knows.

Report every event of these types that you can see in the frames:
shot, goal, yellow_card, red_card, foul, substitution, penalty, free_kick, offside, corner,
injury (play stopped for a player who is down injured or being treated).
- team: "home" or "away", the side performing the action — the shooter's side for shot,
  the side the goal counts for (own goals included) for goal, the booked player's side for
  cards, the offending side for foul, the side taking a free_kick / corner / penalty, the
  side caught offside, the side making a substitution, the injured player's side for injury.
- participants: {{"role", "id"}} with the P-ID. "player" is the main actor (shooter, scorer,
  booked player, fouler, taker, injured player); a foul may add "fouled"; a substitution
  uses "out" and "in".
  Leave out anyone without an ID label; never invent IDs.
- replay: true if the event is shown in a replay rather than live. Live play is shown from
  the main wide broadcast camera. Ball-in-play action shown from any other angle (behind the
  goal, a high tactical camera, pitch level, a tight tracking shot, slow motion) is a replay
  of something that already happened, usually shown during the stoppage after it. Close-ups
  of players' faces and reactions, the coach or the bench, and set-piece preparation (the
  ball being placed, a wall forming) are live. If the same action appears twice in these
  frames, report it once, at its live frame.
- frame: the frame number where the event happens (or is best seen, for a replay);
  evidence_frames: the frame numbers that show it.
Report nothing that is not visible; an empty list is a valid answer.

Answer with JSON only:
{{"events": [{{"type": "...", "team": "home", "frame": 7, "participants": [{{"role": "player", "id": "P3"}}], "evidence_frames": [6, 7, 8], "replay": false}}]}}"""


def build_request(half: int, window: tuple[float, float], frames: Sequence[tuple[float, np.ndarray]],
                  perceived: Mapping[float, tuple[Sequence[Track], tuple | None, Sequence[str | None]]],
                  prematch: Mapping[str, Any], sample_fps: float) -> Request:
    """`frames`: (second, BGR image) of the sampled frames; `perceived`: second ->
    (tracks with identity, key of the track nearest the ball, number read clearly off each
    track) at that frame (see runtime.runner.frame_view); `prematch`: Match.prematch."""
    teams, kits, sides = prematch["teams"], prematch["jersey_colors"], prematch["sides"][half]
    block, images, pid_of = labelled_frames(frames, perceived, prematch)
    t0, t1 = window
    text = INSTRUCTIONS.format(
        window=f"half {half}, {clock(t0)}-{clock(t1)} of the half video",
        home=teams["home"], away=teams["away"], home_kit=kits["home"][0], away_kit=kits["away"][0],
        home_side=sides["home"], away_side=sides["away"], n=len(frames), fps=sample_fps)
    return Request([text + "\n\n" + block, *images],
                   {pid: (half, *key) for key, pid in pid_of.items()},
                   [second for second, _ in frames])


def labelled_frames(frames: Sequence[tuple[float, np.ndarray]],
                    perceived: Mapping[float, tuple[Sequence[Track], tuple | None, Sequence[str | None]]],
                    prematch: Mapping[str, Any]) -> tuple[str, list[str | bytes], dict[tuple, str]]:
    """Frames as a model is shown them (the event analyzer, the QA's current view): each
    tracked player marked with a local ID label ("P3"), and a text block saying who each ID is
    as far as identity voting knows, who is nearest the ball in each frame, and where the
    camera cuts. Returns (text block, ["Frame k (clock)", JPEG, ...], track key -> ID)."""
    players = {p["hash"]: p for p in prematch["players"]}
    teams = prematch["teams"]

    # Local IDs, numbered by first appearance.
    pid_of: dict[tuple, str] = {}
    last: dict[tuple, Track] = {}                 # each track as last seen (latest votes)
    seen_clear: dict[tuple, list[int]] = {}       # frames where its number was read clearly
    carriers, cuts = [], []
    prev_shot = None
    for k, (second, _) in enumerate(frames, start=1):
        tracks, carrier, numbers = perceived.get(second, ((), None, ()))
        for t, number in zip(tracks, numbers):
            pid_of.setdefault(t.key, f"P{len(pid_of) + 1}")
            last[t.key] = t
            if number is not None:
                seen_clear.setdefault(t.key, []).append(k)
        carriers.append(f"{k}:{pid_of[carrier]}" if carrier in pid_of else f"{k}:-")
        shot = tracks[0].shot_id if tracks else prev_shot
        if prev_shot is not None and shot != prev_shot:
            cuts.append(str(k))
        prev_shot = shot

    lines = []
    for key, pid in pid_of.items():
        t = last[key]
        who = f"{teams[t.team]} ({t.team})" if t.team else "team unknown"
        if t.player_id:
            p = players[t.player_id]
            who += f", #{p['jersey_number']} {p['short_name']}"
        if key in seen_clear:
            who += f"; number clearly visible in frames {_ranges(seen_clear[key])}"
        lines.append(f"{pid}: {who}")
    block = "Player IDs:\n" + "\n".join(lines or ["(none tracked)"])
    block += "\nNearest to the ball (frame:ID, - = ball not found): " + " ".join(carriers)
    block += "\nCamera cuts just before frames: " + (", ".join(cuts) or "none")

    parts: list[str | bytes] = []
    for k, (second, image) in enumerate(frames, start=1):
        tracks = perceived.get(second, ((), None, ()))[0]
        image = draw_ids(image, [(pid_of[t.key], t.bbox_ltwh) for t in tracks])
        parts += [f"Frame {k} ({clock(second)})", frame_jpeg(image)]
    return block, parts, pid_of


def parse_candidates(text: str, request: Request) -> list[Candidate]:
    """The model's JSON answer -> candidates. Anything malformed is dropped rather than
    guessed: an unknown type or frame drops the event, an unknown ID drops the participant."""
    events = parse_json(text).get("events", [])
    n = len(request.seconds)
    out = []
    for e in events if isinstance(events, list) else []:
        if not isinstance(e, dict) or e.get("type") not in EVENT_TYPES:
            continue
        frame = e.get("frame")
        if not isinstance(frame, int) or not 1 <= frame <= n:
            continue
        participants = tuple((str(p.get("role")), request.pids[p.get("id")])
                             for p in e.get("participants") or []
                             if isinstance(p, dict) and p.get("id") in request.pids)
        evidence = tuple(request.seconds[k - 1] for k in e.get("evidence_frames") or []
                         if isinstance(k, int) and 1 <= k <= n)
        team = e.get("team") if e.get("team") in ("home", "away") else None
        out.append(Candidate(e["type"], team, request.seconds[frame - 1], participants, evidence,
                             bool(e.get("replay"))))
    return out


def parse_json(text: str) -> dict:
    """A model's JSON answer as a dict ({} if it is not one); a ```json fence is tolerated."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        # Stray text after the object (a lone closing fence; some endpoints go on with a second JSON
        # block, or with tool results the model made up): the first complete object is the answer.
        start = text.find("{")
        try:
            answer = json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else {}
        except json.JSONDecodeError:
            return {}
    return answer if isinstance(answer, dict) else {}


def clock(second: float) -> str:
    return f"{int(second // 60):02d}:{second % 60:04.1f}"


def minute_phrase(half: int, second: float) -> str:
    """15:04 of half 2 -> "16th minute of the 2nd half"; 45:45 -> "1st minute of 2nd-half stoppage time"."""
    minute = int(second // 60)
    stoppage = minute >= 45
    n = minute - 45 + 1 if stoppage else minute + 1
    nth = f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    h = "1st" if half == 1 else "2nd"
    return f"{nth} minute of {h}-half stoppage time" if stoppage else f"{nth} minute of the {h} half"


def _ranges(ks: Sequence[int]) -> str:
    # [4, 5, 6, 9] -> "4-6, 9"
    out, start = [], ks[0]
    for a, b in zip(ks, list(ks[1:]) + [None]):
        if b != a + 1:
            out.append(str(start) if start == a else f"{start}-{a}")
            start = b
    return ", ".join(out)
