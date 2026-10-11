"""Match memory: an append-only op log of beliefs and their justifications.

Nothing derived is stored, except each entry's participants pooled from its justifications
(kept current on every op, see _rebuild). An entry's status comes from its justifications on every read,
and so do the rule inferences (score, lineups, sendings-off — see
matchlive.memory.view) and the player behind every event participant. A correction —
rejecting an observation, confirming a different identity — therefore reaches everything
derived from it without any dependency tracking: the next read simply sees the new state.

Reading the past replays a prefix of the log: t_known never decreases, so the ops known by
time t are exactly the first ones up to t. The log is JSONL, one op per line, flushed as it is
written; MatchMemory.from_log() reads a run's memory back from it.
"""

from __future__ import annotations

import json
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from matchlive.memory.types import Case, Event, Identity, Justification, Op, Participant, Time
from matchlive.memory.view import View

# How many past memories (MatchMemory.known_by) are kept.
PAST_CACHE_SIZE = 16


class MatchMemory:
    def __init__(self, roster: Iterable[Mapping[str, Any]], log_path: str | Path | None = None):
        # `roster` is Match.prematch `players`: pre-match facts only (side, starting flag, hash),
        # so nothing after kick-off leaks into the lineups.
        self.roster: dict[str, Mapping[str, Any]] = {str(p["hash"]): p for p in roster}
        self.ops: list[Op] = []
        # Each entry as it reads now: as written, plus the participants its holding justifications
        # add (_rebuild). Insertion order = write order.
        self.entries: dict[str, Event | Identity] = {}
        self.written: dict[str, Event | Identity] = {}            # each entry as it was written
        self.justifications: dict[str, list[tuple[Time, Justification]]] = {}
        self.rejected: dict[str, Time] = {}                        # obs_id -> when it was rejected
        self.cases: dict[str, Case] = {}
        # Memories rebuilt from the first n ops, by n. The log only grows, so a prefix never changes.
        # Each is a full copy, so only the most recently used are kept (QA asks at many times).
        self._past: OrderedDict[int, MatchMemory] = OrderedDict()
        self._past_lock = threading.Lock()

        # Every op is written to `log_path` (a new log) as it is recorded.
        self._log = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "w")

    @classmethod
    def from_log(cls, roster: Iterable[Mapping[str, Any]], log_path: str | Path) -> "MatchMemory":
        """A finished run's memory read back from its op log (read-only)."""
        mem = cls(roster)
        with open(log_path) as f:
            for line in f:
                d = json.loads(line)
                mem._apply(Op(d["op_id"], tuple(d["t_known"]), d["kind"], d["payload"]))
        return mem

    # ---- writes (each one is an op) ----

    def add_entry(self, t_known: Time, entry: Event | Identity, justification: Justification) -> str:
        """Write a new belief with its first justification; return its entry id."""
        # An entry without a justification would be retracted from the start, so the first
        # one comes with it.
        kind = "event" if isinstance(entry, Event) else "identity"
        entry_id = f"{kind[0].upper()}{len(self.entries) + 1}"
        self._record(t_known, "add_entry", {"entry_id": entry_id, "entry": {"kind": kind, **asdict(entry)},
                                            "justification": asdict(justification)})
        return entry_id

    def add_justification(self, t_known: Time, entry_id: str, justification: Justification) -> None:
        self._record(t_known, "add_justification", {"entry_id": entry_id,
                                                    "justification": asdict(justification)})

    def reject_observation(self, t_known: Time, obs_id: str) -> None:
        """Rule an observation out (a misread number, an overturned verdict "verdict:<case id>");
        every justification citing it stops holding."""
        self._record(t_known, "reject_observation", {"obs_id": obs_id})

    def open_case(self, t_known: Time, members: Iterable[str], constraint: str,
                  evidence: Iterable[str] = ()) -> str:
        case_id = f"C{len(self.cases) + 1}"
        self._record(t_known, "open_case", {"case_id": case_id, "members": list(members),
                                            "constraint": constraint, "evidence": list(evidence)})
        return case_id

    def add_evidence(self, t_known: Time, case_id: str, evidence: Iterable[str]) -> None:
        self._record(t_known, "add_evidence", {"case_id": case_id, "evidence": list(evidence)})

    def close_case(self, t_known: Time, case_id: str, verdict: Any = None) -> None:
        # The verdict's effects are written as ops of their own (justifications with obs_id
        # "verdict:<case id>", rejections, new entries); `verdict` is kept for audit only.
        self._record(t_known, "close_case", {"case_id": case_id, "verdict": verdict})

    def _record(self, t_known: Time, kind: str, payload: dict[str, Any]) -> None:
        t_known = (int(t_known[0]), float(t_known[1]))
        if self.ops and t_known < self.ops[-1].t_known:
            raise ValueError(f"t_known {t_known} is earlier than the last op's {self.ops[-1].t_known}")
        op = Op(len(self.ops), t_known, kind, payload)
        # Applied before it is logged, so an op that fails (e.g. an unknown entry id) never
        # reaches the log. Live writes and log replay go through the same _apply().
        self._apply(op)
        if self._log is not None:
            self._log.write(json.dumps(asdict(op)) + "\n")
            self._log.flush()

    def _apply(self, op: Op) -> None:
        p = op.payload
        if op.kind == "add_entry":
            self.entries[p["entry_id"]] = self.written[p["entry_id"]] = _decode_entry(p["entry"])
            self.justifications[p["entry_id"]] = [(op.t_known, _decode_justification(p["justification"]))]
        elif op.kind == "add_justification":
            j = _decode_justification(p["justification"])
            self.justifications[p["entry_id"]].append((op.t_known, j))
            if j.participants:
                self._rebuild(p["entry_id"])
        elif op.kind == "reject_observation":
            self.rejected.setdefault(p["obs_id"], op.t_known)
            # The participants the rejected observation brought (or a re-check of it) leave the
            # entries it supported.
            for entry_id, js in self.justifications.items():
                if any(p["obs_id"] in (j.obs_id, j.of) and j.participants for _, j in js):
                    self._rebuild(entry_id)
        elif op.kind == "open_case":
            self.cases[p["case_id"]] = Case(p["case_id"], tuple(p["members"]), p["constraint"],
                                            list(p["evidence"]))
        elif op.kind == "add_evidence":
            self.cases[p["case_id"]].evidence.extend(p["evidence"])
        elif op.kind == "close_case":
            case = self.cases[p["case_id"]]
            case.status, case.verdict = "closed", p["verdict"]
        else:
            raise ValueError(f"unknown op kind {op.kind!r}")
        self.ops.append(op)

    def _rebuild(self, entry_id: str) -> None:
        """The entry as written, plus the participants of its justifications that still hold: a
        role it lacks is added, a role it has gets the track as an
        alternate. Kept up to date on every op that can change it, so reads see it directly."""
        e = self.written[entry_id]
        parts = list(e.participants)
        for _, j in self.justifications[entry_id]:
            if not self.holds(j):
                continue
            for q in j.participants:
                i = next((k for k, part in enumerate(parts) if part.role == q.role), None)
                if i is None:
                    parts.append(q)
                elif q.track_key not in (parts[i].track_key, *parts[i].alternates):
                    parts[i] = replace(parts[i], alternates=parts[i].alternates + (q.track_key,))
        self.entries[entry_id] = replace(e, participants=tuple(parts))

    # ---- derived on read ----

    def status(self, entry_id: str) -> str:
        """"confirmed" | "pending" | "retracted", from the entry's justifications: confirmed when a
        sufficient one holds and no open case disputes the entry, retracted when none holds."""
        holding = [j for _, j in self.justifications[entry_id] if self.holds(j)]
        if not holding:
            return "retracted"
        if any(j.sufficient for j in holding) and not self.open_cases(entry_id):
            return "confirmed"
        return "pending"

    def holds(self, j: Justification) -> bool:
        """A justification holds while its observation is not rejected — and, for a re-check,
        while the observation it re-checks is not rejected either."""
        return j.obs_id not in self.rejected and j.of not in self.rejected

    def open_cases(self, entry_id: str) -> list[str]:
        """The open cases this entry is a member of; while there are any it is pending."""
        return [c.case_id for c in self.cases.values() if c.status == "open" and entry_id in c.members]

    def track_players(self) -> dict[tuple, str]:
        """{track_key: player_id} from the confirmed Identity entries."""
        return {e.track_key: e.player_id for entry_id, e in self.entries.items()
                if isinstance(e, Identity) and self.status(entry_id) == "confirmed"}

    def event_players(self, entry_id: str, track_players: Mapping[tuple, str]) -> tuple[str | None, ...]:
        """The player behind each participant of an event (None if not yet identified), given
        track_players()."""
        return resolve_players(self.entries[entry_id], track_players)

    def counted_events(self) -> dict[str, tuple[Event, tuple[str | None, ...]]]:
        """The events the rule inferences count — the confirmed ones — each with its
        participants' players."""
        track_players = self.track_players()
        return {entry_id: (e, resolve_players(e, track_players)) for entry_id, e in self.entries.items()
                if isinstance(e, Event) and self.status(entry_id) == "confirmed"}

    # ---- queries ----

    def known_by(self, t: Time) -> MatchMemory:
        """Memory as it stood at match time t: t_known never decreases, so what was known by
        then is a prefix of the log."""
        n = bisect_right(self.ops, t, key=lambda op: op.t_known)
        if n == len(self.ops):
            return self
        with self._past_lock:
            mem = self._past.get(n)
            if mem is not None:
                self._past.move_to_end(n)
                return mem
        mem = MatchMemory(self.roster.values())
        for op in self.ops[:n]:
            mem._apply(op)
        with self._past_lock:
            self._past[n] = mem
            while len(self._past) > PAST_CACHE_SIZE:
                self._past.popitem(last=False)
        return mem

    def at(self, event_time: Time, known_by: Time | None = None) -> View:
        """The match as of `event_time`, using only what was known by `known_by` (default:
        `event_time` itself — no peeking into the future)."""
        return View(self.known_by(event_time if known_by is None else known_by), event_time)


def resolve_players(event: Event, track_players: Mapping[tuple, str]) -> tuple[str | None, ...]:
    # A participant named directly wins; otherwise its track's confirmed identity, else the
    # first of its alternate tracks (other sightings of the event) with one.
    players = [p.player_id or track_players.get(p.track_key)
               or next((track_players[k] for k in p.alternates if k in track_players), None)
               for p in event.participants]
    # One player takes one role in an event. A player found in two roles (e.g. a substitution
    # whose "in" and "out" tracks were both identified as Oscar) is a misidentification, and
    # neither role can say which: both stay unidentified.
    return tuple(None if pid and players.count(pid) > 1 else pid for pid in players)


def _decode_entry(d: Mapping[str, Any]) -> Event | Identity:
    # Ops carry entries as plain dicts (as written to the JSONL log), so JSON arrays come
    # back as lists; keys and times are tuples again here. Every write passes through, so
    # this is also where the track-key contract is checked.
    if d["kind"] == "identity":
        return Identity(_track_key(d["track_key"]), d["player_id"])
    return Event(d["type"], d["team"], (tuple(d["t_interval"][0]), tuple(d["t_interval"][1])),
                 tuple(map(_decode_participant, d["participants"])))


def _decode_justification(d: Mapping[str, Any]) -> Justification:
    return Justification(d["obs_id"], d["kind"], d["sufficient"],
                         tuple(map(_decode_participant, d["participants"])), d["of"])


def _decode_participant(p: Mapping[str, Any]) -> Participant:
    return Participant(p["role"], None if p["track_key"] is None else _track_key(p["track_key"]), p["player_id"],
                       tuple(map(_track_key, p["alternates"])))


def _track_key(key) -> tuple:
    # Outside perception a track is (half, shot_id, track_id); perception's
    # own (shot_id, track_id) would collide across halves and never match an identity.
    if len(key) != 3:
        raise ValueError(f"track key {key!r} is not (half, shot_id, track_id)")
    return tuple(key)
