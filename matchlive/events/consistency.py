"""Consistency check: pure rules that turn event candidates into memory ops.

Every candidate goes through two steps:
  1. Match it to existing entries — same type, same half, compatible team, overlapping event
     time. One match: the candidate becomes another justification of that entry. None: a new
     entry. Several: a new entry plus a case for the arbiter to say which one it was.
  2. Check the constraints. Mutually exclusive beliefs (two goals by one team with no time
     for a kick-off, the same player carded twice within a minute, an invalid substitution,
     a player back on after leaving, two sources naming different players) open a case with
     those beliefs as members. A missing partner (a foul with no restart after it, a free
     kick with no infringement before it) is only judged once enough match time has passed
     for the partner to have been analysed — see `tick()`.
Perception proposes beliefs too: a number read on the pitch that
belongs to a player memory has off it — see `off_roster_read()`.

A foul and the free kick that restarts play after it confirm each other: each one's first
observation is added as a justification of the other, since two independent sources agree.

A justification is sufficient when a second, independent source agrees with the entry, or
when a re-check of a lone live sighting sees it again (`recheck()`: the same analyzer on the
same footage — not independent, a precision-for-recall trade); a
single sighting otherwise leaves it pending. Who did it is not part of this: the players
behind an event come from the Identity entries, which carry their own status.

Status is never written here, and cases are never closed here (that is the arbiter's call):
memory derives status from the justifications and open cases. No LLM is called. The memory
instance is passed in on every call by the runtime; this module keeps none.
"""

from __future__ import annotations

from typing import Any, Sequence

from matchlive.events.analyzer import Candidate
from matchlive.memory.store import MatchMemory, resolve_players
from matchlive.memory.types import Event, Identity, Justification, Participant, Time, track_name

# The two mutual exclusions that compare an entry with its neighbours: (constraint, event types, config key of the gap, must be the same player).
# After a goal play restarts with a kick-off, so one team cannot score twice within the gap
# (one of the two is almost always a replay seen as new). The same player carded twice
# within the gap is far likelier one card reported twice — and two yellows would send them off.
EXCLUSIONS = (("kickoff_after_goal", ("goal",), "kickoff_gap", False),
              ("duplicate_card", ("yellow_card", "red_card"), "card_gap", True))

# Infringement and restart pairing: type -> (partner types, window as multiples of
# pair_window relative to the event, constraint opened when the partner is missing or None).
# Offsides and penalties pair too, but only a foul and a free kick have a missing-partner rule.
PARTNERS = {"foul": (("free_kick", "penalty"), (0, 1), "restart_after_foul"),
            "offside": (("free_kick",), (0, 1), None),
            "free_kick": (("foul", "offside"), (-1, 0), "infringement_before_free_kick"),
            "penalty": (("foul",), (-1, 0), None)}


class ConsistencyChecker:
    def __init__(self, config: dict[str, Any]):
        # `config` is the whole config: the `consistency` and `events` sections.
        self.cfg = config["consistency"]
        self.live_margin = config["events"]["live_margin"]
        self.replay_lookback = config["events"]["replay_lookback"]

    # ---- step 1 + 2 for one candidate ----

    def propose(self, mem: MatchMemory, t_known: Time, half: int, window: tuple[float, float],
                cand: Candidate, obs_id: str, players: Sequence[str | None]) -> str:
        """Write one candidate (its participants' voted players in `players`) into memory at
        knowledge time `t_known`; return the id of the entry it now supports."""
        # Participants' identities first: a track an event references gets an Identity entry,
        # justified by the perception vote (which needs two clear reads, so it suffices).
        for (_, key), player in zip(cand.participants, players):
            if player:
                self._identity(mem, t_known, key, player)

        event = self._event(half, window, cand)
        events = live_events(mem)
        matches = matching(events, event)
        if len(matches) == 1:
            self._join(mem, t_known, matches[0], event, obs_id, "observation", None)
            return matches[0]

        # A single sighting never suffices on its own.
        eid = mem.add_entry(t_known, event, Justification(obs_id, "observation", False))
        if len(matches) > 1:
            mem.open_case(t_known, [*matches, eid], "ambiguous_match", [obs_id])
        self._check_new(mem, t_known, eid, events)
        return eid

    def recheck(self, mem: MatchMemory, t_known: Time, eid: str, half: int, window: tuple[float, float],
                cand: Candidate, obs_id: str, players: Sequence[str | None]) -> bool:
        """One candidate of the re-check of entry `eid`: if it is a sighting of
        that entry by step 1's rule, it joins the entry as a "recheck" justification, sufficient
        unless it names other players. Returns whether it joined. Whatever else the re-check saw
        is not proposed: the regular windows have analysed that footage already."""
        event = self._event(half, window, cand)
        if not matching([(eid, mem.entries[eid])], event):
            return False
        for (_, key), player in zip(cand.participants, players):
            if player:
                self._identity(mem, t_known, key, player)
        # It re-checks the entry's lone sighting, and holds only while that does.
        self._join(mem, t_known, eid, event, obs_id, "recheck", mem.justifications[eid][0][1].obs_id)
        return True

    def _event(self, half: int, window: tuple[float, float], cand: Candidate) -> Event:
        """The event a candidate claims, with its event-time interval: a replay shows an event that
        happened in the `replay_lookback` s before the window; a live one, its second ± a margin."""
        if cand.replay:
            interval = ((half, max(0.0, window[0] - self.replay_lookback)), (half, window[1]))
        else:
            margin = self.live_margin.get(cand.type, self.live_margin["default"])
            interval = ((half, max(0.0, cand.second - margin)), (half, cand.second + margin))
        return Event(cand.type, cand.team, interval,
                     tuple(Participant(role, track_key=key) for role, key in cand.participants))

    def _join(self, mem: MatchMemory, t_known: Time, eid: str, event: Event, obs_id: str, kind: str,
              of: str | None) -> None:
        """A sighting of entry `eid` becomes one more justification of it (step 1)."""
        # Two claims about one event disagree when they name different identified players
        # in the same role.
        track_players = mem.track_players()
        old, new = _by_role(mem.entries[eid], track_players), _by_role(event, track_players)
        clash = any(old[r] and new[r] and old[r] != new[r] for r in old.keys() & new.keys())
        # Claims that agree pool their participants: who this sighting saw joins the entry
        # so a player only the second window identified is not lost.
        mem.add_justification(t_known, eid, Justification(obs_id, kind, not clash,
                                                          () if clash else event.participants, of))
        if clash:
            mem.open_case(t_known, [eid, obs_id], "claims_disagree", [obs_id])

    def _identity(self, mem: MatchMemory, t_known: Time, key: tuple, player: str) -> None:
        claims = [(eid, e) for eid, e in mem.entries.items()
                  if isinstance(e, Identity) and e.track_key == key and mem.status(eid) != "retracted"]
        if any(e.player_id == player for _, e in claims):
            return
        eid = mem.add_entry(t_known, Identity(key, player),
                            Justification(f"vote:{track_name(key)}:{player}", "observation", True))
        if claims:
            mem.open_case(t_known, [c for c, _ in claims] + [eid], "claims_disagree")

    # ---- constraints on a new entry ----

    def _check_new(self, mem: MatchMemory, t_known: Time, eid: str, events: list[tuple[str, Event]]) -> None:
        # `events` is the entry list from before `eid` was added.
        e = mem.entries[eid]
        half, t = e.t_interval[1]
        track_players = mem.track_players()
        players = _by_role(e, track_players)

        for constraint, types, gap, same_player in EXCLUSIONS:
            if e.type not in types or (same_player and players.get("player") is None):
                # Without the player identified, nothing says two cards went to the same one.
                continue
            clash = [oid for oid, o in events if o.type in types and o.t_interval[1][0] == half and _same_team(o, e)
                     and abs(o.t_interval[1][1] - t) < self.cfg[gap]
                     and (not same_player or _by_role(o, track_players).get("player") == players.get("player"))]
            if clash:
                mem.open_case(t_known, [*clash, eid], constraint)

        if e.type == "substitution" or (e.type not in ("yellow_card", "red_card") and any(players.values())):
            # The lineups as known now, just before this event.
            before = mem.at((half, t - 0.01), known_by=t_known)
            on_field, bench = sum(before.on_field.values(), []), sum(before.bench.values(), [])
            if e.type == "substitution":
                out, sub = players.get("out"), players.get("in")
                made = len(before.events(type="substitution", team=e.team, status="confirmed")) if e.team else 0
                if (out and out not in on_field) or (sub and sub not in bench) \
                        or made >= self.cfg["substitution_limit"]:
                    mem.open_case(t_known, [eid], "substitution_invalid")
            elif any(p in mem.roster and p not in on_field and p not in bench for p in players.values() if p):
                # A player who was sent off or substituted is neither on the pitch nor on the
                # bench; they cannot take part in play again (cards can go to anyone).
                mem.open_case(t_known, [eid], "player_off_pitch")

        # An infringement and the restart after it confirm each other.
        partner = self._partner(events, e)
        if partner is not None:
            for a, b in ((eid, partner), (partner, eid)):
                first = mem.justifications[b][0][1].obs_id
                if all(j.obs_id != first for _, j in mem.justifications[a]):
                    mem.add_justification(t_known, a, Justification(first, "observation", True))
            # A partner that turns up after its absence was already judged is new evidence
            # for that case; closing it is the arbiter's call.
            for c in (mem.cases[cid] for x in (eid, partner) for cid in mem.open_cases(x)):
                if c.constraint in ("restart_after_foul", "infringement_before_free_kick"):
                    mem.add_evidence(t_known, c.case_id, [first])

    def _partner(self, events: list[tuple[str, Event]], e: Event) -> str | None:
        """The nearest entry among `events` that pairs with `e` (PARTNERS), taken by the other
        team — the infringing side never takes the free kick."""
        if e.type not in PARTNERS:
            return None
        types, (lo, hi), _ = PARTNERS[e.type]
        w = self.cfg["pair_window"]
        half, t = e.t_interval[1]
        found = [(abs(o.t_interval[1][1] - t), oid) for oid, o in events
                 if o.type in types and o.t_interval[1][0] == half and lo * w <= o.t_interval[1][1] - t <= hi * w
                 and (o.team is None or e.team is None or o.team != e.team)]
        return min(found)[1] if found else None

    # ---- evidence from perception ----

    def off_roster_read(self, mem: MatchMemory, t_known: Time, obs_id: str, player: str) -> str | None:
        """A track keeps reading as `player`, whom memory has off the pitch (the observation
        `obs_id`, see runtime.runner.off_roster_obs). It contradicts the lineup — a missed
        substitution, or misreads — so it joins the open off_roster_read case of that player
        (opening one if there is none) as one more independent source. Returns the case id
        when the case has just reached `off_roster_k` sources: per-frame perception conflicts
        go to the arbiter only then, so one misread never does."""
        case = next((c for c in mem.cases.values() if c.status == "open" and c.constraint == "off_roster_read"
                     and c.members[0].endswith(":" + player)), None)
        if case is None:
            case_id = mem.open_case(t_known, [obs_id], "off_roster_read", [obs_id])
        else:
            case_id = case.case_id
            mem.add_evidence(t_known, case_id, [obs_id])
        return case_id if len(mem.cases[case_id].evidence) == self.cfg["off_roster_k"] else None

    # ---- missing-partner constraints, judged once time has passed ----

    def tick(self, mem: MatchMemory, t_known: Time) -> None:
        """Open a case for every foul with no restart after it, and every free kick with no
        infringement before it, once `missing_wait` s of match time have passed since the
        event (so the partner's window has been analysed and its result has arrived)."""
        events = live_events(mem)
        cased = {(m, c.constraint) for c in mem.cases.values() for m in c.members}
        for eid, e in events:
            constraint = PARTNERS.get(e.type, (None, None, None))[2]
            half, t = e.t_interval[1]
            if constraint and (half, t + self.cfg["missing_wait"]) <= t_known \
                    and (eid, constraint) not in cased and self._partner(events, e) is None:
                mem.open_case(t_known, [eid], constraint)


def matching(events: list[tuple[str, Event]], event: Event) -> list[str]:
    """Step 1: the entries among `events` that a sighting of `event` belongs
    to — same type and half, overlapping time, compatible team."""
    return [eid for eid, e in events
            if e.type == event.type and _overlaps(e.t_interval, event.t_interval) and _same_team(e, event)]


def live_events(mem: MatchMemory) -> list[tuple[str, Event]]:
    """The event entries that still stand (not retracted)."""
    return [(eid, e) for eid, e in mem.entries.items() if isinstance(e, Event) and mem.status(eid) != "retracted"]


def _by_role(event: Event, track_players: dict[tuple, str]) -> dict[str, str | None]:
    return dict(zip((p.role for p in event.participants), resolve_players(event, track_players)))


def _overlaps(a: tuple[Time, Time], b: tuple[Time, Time]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _same_team(a: Event, b: Event) -> bool:
    return a.team is None or b.team is None or a.team == b.team
