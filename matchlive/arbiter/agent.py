"""Arbiter: settles one conflict case by gathering evidence and ruling.

The exchange is plain JSON over single-turn calls, so any model behind
matchlive.common.api works and every round is cached like the event analyzer's calls:
  round 1  the case packet — members, the rule they break, their justifications and what
           the event analyzer said in each cited observation (no pictures: cheap first);
  answer   {"requests": [{"tool", "args"}]} for more evidence, or {"verdict": {...}};
  round n  the packet again plus every piece of evidence gathered so far, each with an id
           ("ev3") that a verdict must cite.
After `max_rounds` without a verdict the case stays open. Tools come from the runtime (they
read video, perception and memory, which this package does not hold: runtime.evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matchlive.events.analyzer import clock, parse_json
from matchlive.memory.store import MatchMemory, resolve_players
from matchlive.memory.types import Case, Identity, Time, track_name
from matchlive.memory.view import EventRecord


@dataclass
class Evidence:
    text: str
    images: list[bytes] = field(default_factory=list)     # JPEG bytes
    id: str = ""                                          # set by the loop: "ev1", "ev2", ...


@dataclass
class Decision:
    verdict: dict | None            # None: no verdict within the budget (the case stays open)
    evidence_ids: list[str]
    calls: int
    latency: float                  # seconds, summed over the calls
    note: str = ""                  # why there is no verdict


# What each constraint means, in the words the arbiter reads.
RULES = {
    "kickoff_after_goal": "After a goal play restarts with a kick-off, so one team cannot score twice this close "
                          "together: one of these goals is most likely a replay taken for a new goal.",
    "duplicate_card": "The same player is carded twice within a minute: most likely one card reported twice "
                      "(two yellows would send the player off).",
    "restart_after_foul": "A foul is normally followed by a free kick or a penalty for the other team, but none was "
                          "found in the next minute: either the restart was missed, the referee played advantage, "
                          "or it was not a foul.",
    "infringement_before_free_kick": "A free kick is always preceded by an infringement (a foul or an offside by the "
                                     "other team), but none was found in the minute before: either it was missed, "
                                     "or this was not a free kick.",
    "claims_disagree": "Two sources name different players for the same thing (the same event, or the same track).",
    "ambiguous_match": "A new sighting fits more than one existing event; say which one it is (or that it is new).",
    "substitution_invalid": "This substitution does not fit the lineups: the player going off is not on the pitch, "
                            "the player coming on is not on the bench, or the team is past its substitution limit.",
    "player_off_pitch": "A player who was already sent off or substituted takes part in play again.",
    "off_roster_read": "The shirt number of a player memory has off the pitch (on the bench, substituted or sent off) "
                       "was read clearly on several tracked players, and nobody on the pitch wears that number: either "
                       "a substitution was missed (add it), or the reads are wrong. A substitution puts the substitute "
                       "on the pitch only when both of its players are known: name its tracks with identity writes, or "
                       "reject its observations and add it again naming the two players.",
}

INSTRUCTIONS = """You are the arbiter of a live football-match memory. A consistency rule flagged a conflict between
beliefs the memory holds; the beliefs are proposals from an automatic event analyzer and a player tracker, and some
of them are wrong. Investigate with the tools, then rule. Round {round} of at most {max_rounds}.

Tools (at most {max_requests} requests per round; nothing after the time the case opened can be seen):
- get_frames {{"half": 1, "start": 1830, "end": 1860}}: up to {max_frames} broadcast frames spread over [start, end]
  (seconds of the half), each labelled with its clock, tracked players marked with their track id "shot/track".
- get_track_crops {{"track": [half, shot, track]}}: pictures of one tracked player across the frames they were seen.
- jersey_reads {{"track": [half, shot, track]}}: which shirt numbers the number reader saw clearly on that track.
- list_events {{"half": 1, "start": 1700, "end": 1900}}: the events memory holds in that span, with their status.
- why {{"entry": "E12"}}: what an entry rests on.

Rule with writes, each citing the evidence ids ("ev1", ...) that show it:
- {{"op": "support", "entry": "E12", "evidence": [...]}}: the entry is right (it happened / the identity is right).
- {{"op": "reject", "observation": "obs:...#1", "evidence": [...]}}: an observation is wrong, e.g. a replay taken for a
  new event, a misread number, an event that did not happen.
- {{"op": "add_event", "type": "free_kick", "team": "home", "half": 1, "second": 1234, "participants":
  [{{"role": "player", "track": [1, 40, 7]}}], "evidence": [...]}}: an event that happened but nobody reported. A
  participant with no track may name a roster player instead, {{"role": "in", "player": "<roster hash>"}} (e.g. a
  substitution nobody saw, inferred from the substitute's number being read on the pitch).
- {{"op": "identity", "track": [1, 40, 7], "player": "<roster hash>", "evidence": [...]}}: who a track is.
Support the entries you find to be right as well as rejecting the wrong ones. An empty action list dismisses the case.

How to tell a replay: live play is shown from the main wide broadcast camera. Ball-in-play action shown from any
other angle — behind the goal, a high tactical camera, pitch level, a tight tracking shot, slow motion — is a replay
of something that already happened, usually shown during the stoppage after it. Close-ups of players' faces and
reactions, the coach or the bench, and set-piece preparation (the ball being placed, a wall forming) are live. A
replay is never a new event: find the live moment it repeats.
Only rule on what the evidence shows; if it cannot be told, ask for more evidence or leave the actions empty.

Answer with JSON only, either
{{"requests": [{{"tool": "get_frames", "args": {{...}}}}]}}
or
{{"verdict": {{"summary": "one sentence", "actions": [...]}}}}{last}"""


def instructions(config: Mapping[str, Any], round: int) -> str:
    """INSTRUCTIONS for round `round` under the `arbiter` config section."""
    last = "\nThis is the last round: give your verdict now." if round == config["max_rounds"] else ""
    return INSTRUCTIONS.format(round=round, max_rounds=config["max_rounds"], last=last,
                               max_requests=config["max_requests"], max_frames=config["max_frames"])


def build_packet(mem: MatchMemory, case: Case, t_bound: Time, observations: Mapping[str, Mapping[str, Any]],
                 prematch: Mapping[str, Any], context: float) -> str:
    """The case as text: `observations` maps an observation id to what produced it (the
    analyzer window and candidate); `context` s around the members lists neighbouring events."""
    teams, kits, sides = prematch["teams"], prematch["jersey_colors"], prematch["sides"]
    view = mem.at(t_bound, known_by=t_bound)
    track_players = mem.track_players()
    lines = [f"Case {case.case_id} — {case.constraint}: {RULES.get(case.constraint, '')}",
             f"The case opened at half {t_bound[0]}, {clock(t_bound[1])}.",
             *(f"{side.capitalize()}: {teams[side]} ({kits[side][0]} kit), defending the goal on the {sides[1][side]} "
               f"of the main camera's picture in half 1 and on the {sides[2][side]} in half 2." for side in ("home", "away")),
             "", "Members:"]
    spans = []
    for m in case.members:
        if m not in mem.entries:
            lines.append(f"- observation {m}: {_observation(m, observations, prematch)}")
            continue
        e = mem.entries[m]
        if isinstance(e, Identity):
            lines.append(f"- {m}: identity — track {track_name(e.track_key)} is {_player(e.player_id, prematch)}"
                         f" [{mem.status(m)}]")
        else:
            (h, lo), (_, hi) = e.t_interval
            spans.append((h, lo, hi))
            who = ", ".join(f"{p.role} = track {track_name(p.track_key) if p.track_key else '-'}"
                            + "".join(f" / {track_name(k)}" for k in p.alternates)
                            + f" ({_player(pid, prematch)})"
                            for p, pid in zip(e.participants, resolve_players(e, track_players))) or "no participants"
            lines.append(f"- {m}: {e.type}, team {e.team}, half {h} {clock(lo)}–{clock(hi)} [{mem.status(m)}]; {who}")
        for t, j in mem.justifications[m]:
            state = "rejected" if not mem.holds(j) else ("sufficient" if j.sufficient else "insufficient")
            lines.append(f"    · {j.obs_id} ({j.kind}, {state}): {_observation(j.obs_id, observations, prematch)}")
    # Independent sources merged into the case after it opened and not shown above (e.g. more
    # tracks reading the same off-pitch player's number).
    shown = {j.obs_id for m in case.members if m in mem.entries for _, j in mem.justifications[m]}
    merged = [e for e in dict.fromkeys(case.evidence) if e not in case.members and e not in shown]
    if merged:
        lines += ["", "Further observations merged into this case:"]
        lines += [f"- observation {e}: {_observation(e, observations, prematch)}" for e in merged]
    if case.constraint == "off_roster_read":
        # The lineup comes from the substitutions memory holds; show them with their players.
        lines += ["", "Substitutions memory holds so far:"]
        subs = view.events(type="substitution")
        for r in subs:
            (h, lo), (_, hi) = r.event.t_interval
            who = ", ".join(f"{p.role} = track {track_name(p.track_key) if p.track_key else '-'} ({_player(pid, prematch)})"
                            for p, pid in zip(r.event.participants, r.players)) or "no participants"
            lines.append(f"- {r.entry_id}: team {r.event.team}, half {h} {clock(lo)}–{clock(hi)} [{r.status}]; {who}")
        if not subs:
            lines.append("(none)")
    if spans:
        lines += ["", "Other events memory holds nearby:"]
        for r in view.events():
            (h, lo), (_, hi) = r.event.t_interval
            if r.entry_id not in case.members and any(h == sh and lo - context <= shi and slo <= hi + context
                                                      for sh, slo, shi in spans):
                lines.append(event_line(r))
    lines += ["", "Roster (hash: side, number, name):"]
    lines += [f"- {p['hash']}: {p['team_side']}, #{p['jersey_number']}, {p['short_name']}"
              + ("" if p["starting"] else " (substitute)") for p in prematch["players"]]
    return "\n".join(lines)


class Arbiter:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `arbiter` section; `generate(parts)` is one JSON-answer model call
        # returning an object with .text and .latency (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def decide(self, packet: str, call_tool: Callable[[str, Mapping[str, Any]], Evidence]) -> Decision:
        evidence: list[Evidence] = []
        calls, latency = 0, 0.0
        for r in range(1, self.cfg["max_rounds"] + 1):
            parts: list[str | bytes] = [instructions(self.cfg, r), packet]
            for ev in evidence:
                parts += [f"[{ev.id}] {ev.text}", *ev.images]
            result = self.generate(parts)
            calls, latency = calls + 1, latency + result.latency
            answer = parse_json(result.text)
            if isinstance(answer.get("verdict"), dict):
                return Decision(answer["verdict"], [ev.id for ev in evidence], calls, latency)
            requests = answer.get("requests")
            if not isinstance(requests, list) or not requests:
                return Decision(None, [ev.id for ev in evidence], calls, latency, "answered neither requests nor verdict")
            for req in requests[: self.cfg["max_requests"]]:
                ev = call_tool(str(req.get("tool")), req.get("args") or {}) if isinstance(req, dict) \
                    else Evidence("error: a request must be an object")
                ev.id = f"ev{len(evidence) + 1}"
                evidence.append(ev)
        return Decision(None, [ev.id for ev in evidence], calls, latency, "out of rounds")


def _observation(obs_id: str, observations: Mapping[str, Mapping[str, Any]], prematch: Mapping[str, Any]) -> str:
    if obs_id.startswith("verdict:"):
        return "an earlier arbiter verdict"
    if obs_id.startswith("vote:"):
        return "the number reader's vote on that track"
    if obs_id.startswith("read:"):
        # "read:<half>/<shot>/<track>:<player hash>" (runtime.runner.off_roster_obs)
        key, pid = obs_id.removeprefix("read:").split(":")
        return (f"the number reader read {_player(pid, prematch)}'s shirt number clearly and repeatedly on track {key}, "
                f"though memory has that player off the pitch")
    o = observations.get(obs_id)
    if o is None:
        return "(no record)"
    c, (w0, w1) = o["candidate"], o["window"]
    who = "; ".join(f"{role} = track {track_name(key)} ({_player(pid, prematch)} by the number vote)"
                    for (role, key), pid in zip(c["participants"], c.get("players", [None] * len(c["participants"]))))
    return (f"the analyzer, watching {clock(w0)}–{clock(w1)}, reported {c['type']} by {c['team']} at "
            f"{clock(c['second'])}{' in a replay' if c['replay'] else ' live'}" + (f"; {who}" if who else ""))


def event_line(r: EventRecord) -> str:
    """One event in a list the arbiter reads: "- E12: shot, team home, half 1 10:05.0–10:25.0 [pending]"."""
    (h, lo), (_, hi) = r.event.t_interval
    return f"- {r.entry_id}: {r.event.type}, team {r.event.team}, half {h} {clock(lo)}–{clock(hi)} [{r.status}]"


def _player(pid: str | None, prematch: Mapping[str, Any]) -> str:
    if not pid:
        return "unidentified"
    p = next((p for p in prematch["players"] if p["hash"] == pid), None)
    return f"{p['team_side']} #{p['jersey_number']} {p['short_name']}" if p else pid
