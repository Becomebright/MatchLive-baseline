"""Match runner.

MatchRunner makes one pass over a half video and writes the raw perception cache
(matchlive.common.cache). build_memory() then plays the whole match forward on one match clock:
identity voting frame by frame (fed by memory), the event analyzer on every fixed window, the
consistency check and the arbiter — each API answer taking effect at its issue time + measured
latency, as it would live.
"""

from __future__ import annotations

import heapq
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from matchlive.arbiter.agent import Arbiter, Decision, build_packet
from matchlive.arbiter.verdict import apply_verdict
from matchlive.common.api import NoAPIKey
from matchlive.common.cache import ROLE_CODE, cache_path, half_cache, iter_frames, load_perception, save_perception
from matchlive.common.match import Match
from matchlive.common.types import Track
from matchlive.events.analyzer import Candidate, build_request, parse_candidates
from matchlive.events.consistency import ConsistencyChecker
from matchlive.events.windows import fixed_windows
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, track_name
from matchlive.perception.detection import RFDETRDetector, to_detections
from matchlive.perception.identity import IdentityResolver, kit_color, kit_lab, roster_index
from matchlive.perception.jersey import NUM_LABELS, JerseyReader, clear_number, crop_upper_body
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import PlayerTracker, select_ball
from matchlive.perception.visualization import nearest_to_ball
from matchlive.runtime.evidence import EvidenceTools


class MatchRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        paths, perception = config["paths"], config["perception"]
        # Model sessions are expensive to create (TensorRT engines load from disk), so
        # they live as long as the runner; per-sequence state is created in perceive().
        self.detector = RFDETRDetector(perception["detection"], paths["checkpoints"], paths["cache"])
        self.jersey = JerseyReader(perception["jersey"], paths["checkpoints"], paths["cache"])

    def run_perception(self, match: Match, half: int, start_frame: int = 0,
                       n_frames: int | None = None) -> Path:
        """Write the raw perception cache for frames [start_frame, start_frame + n_frames)
        of one half (the whole half when n_frames is None); return its path. An existing
        cache for the same inputs is reused."""
        video = match.video_path(half)
        path = cache_path(self.config, video, match.key, half, start_frame, n_frames)
        if path.exists():
            return path

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        def frames():
            frame_idx = start_frame
            while n_frames is None or frame_idx < start_frame + n_frames:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame_idx, frame
                frame_idx += 1

        try:
            arrays, timing = self.perceive(frames(), fps)
        finally:
            cap.release()
        meta = {"match": match.key, "half": half, "fps": fps, "start_frame": start_frame,
                "n_frames": len(arrays["shot_id"]), **timing}
        save_perception(path, arrays, meta)
        return path

    def perceive(self, frames: Iterable[tuple[int, np.ndarray]], fps: float) -> tuple[dict, dict]:
        """Run the per-frame chain over (frame_idx, BGR frame) pairs and return the raw cache
        arrays (layout in matchlive.common.cache) and the mean ms/frame of each stage
        ({"ms_per_frame": ...})."""
        perception = self.config["perception"]
        shots = ShotDetector(perception["shots"]["cut_threshold"])
        tracker = PlayerTracker(perception["tracking"], frame_rate=fps)

        shot_ids, det_rows, track_rows, jersey_rows, color_rows = [], [], [], [], []
        timing = {"decode": 0.0, "detect": 0.0, "shots": 0.0, "track": 0.0, "jersey": 0.0, "color": 0.0}
        t_prev = time.perf_counter()
        for frame_idx, frame in frames:
            t0 = time.perf_counter()
            dets = self.detector.predict(frame)
            t1 = time.perf_counter()
            shot_id = shots.update(frame)
            t2 = time.perf_counter()
            tracks = tracker.update(dets, frame, shot_id)
            t3 = time.perf_counter()
            # Read every track on every frame; boxes too small to crop get NaN rows so the
            # jersey and colour arrays stay aligned with the track rows.
            crops = [crop_upper_body(frame, t.bbox_ltwh) for t in tracks]
            probs = np.full((len(tracks), NUM_LABELS), np.nan, dtype=np.float32)
            readable = [i for i, c in enumerate(crops) if c is not None]
            if readable:
                probs[readable] = self.jersey.predict_batch([crops[i] for i in readable])
            t4 = time.perf_counter()
            colors = np.full((len(tracks), 3), np.nan, dtype=np.float32)
            for i in readable:
                colors[i] = kit_color(crops[i])
            t5 = time.perf_counter()

            shot_ids.append(shot_id)
            det_rows += [(frame_idx, *d.bbox_ltwh, d.score, d.class_id) for d in dets]
            track_rows += [(frame_idx, shot_id, ROLE_CODE[t.role], t.track_id, *t.bbox_ltwh) for t in tracks]
            jersey_rows.append(probs)
            color_rows.append(colors)
            # "decode" is the time spent getting this frame from the iterator.
            for key, dt in zip(timing, (t0 - t_prev, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                timing[key] += dt
            t_prev = time.perf_counter()

        n = max(len(shot_ids), 1)
        arrays = {
            "shot_id": np.asarray(shot_ids, dtype=np.int32),
            "det": np.asarray(det_rows, dtype=np.float32).reshape(-1, 7),
            "track": np.asarray(track_rows, dtype=np.float32).reshape(-1, 8),
            "jersey": np.concatenate(jersey_rows or [np.zeros((0, NUM_LABELS))]).astype(np.float16),
            "color": np.concatenate(color_rows or [np.zeros((0, 3))]).astype(np.float32),
        }
        return arrays, {"ms_per_frame": {k: 1000 * v / n for k, v in timing.items()}}


def frame_view(config: dict[str, Any], tracks: list[Track], jersey: np.ndarray,
               det: np.ndarray) -> tuple[list[Track], tuple | None, list[str | None]]:
    """One frame as the event analyzer (and the QA's frame tools) are told about it: (tracks
    with identity, the key of the track nearest the ball or None, the number read clearly off
    each track or None)."""
    ball = select_ball(to_detections(det[:, :4], det[:, 4], det[:, 5], 0.0), config["perception"]["ball"]["min_score"])
    ball_point = None if ball is None else (ball.bbox_ltwh[0] + ball.bbox_ltwh[2] / 2,
                                            ball.bbox_ltwh[1] + ball.bbox_ltwh[3])
    carrier = nearest_to_ball(tracks, ball_point)
    # "Clearly read" uses the same bar as a jersey vote (identity.jersey_min_prob).
    min_prob = config["perception"]["identity"]["jersey_min_prob"]
    return tracks, None if carrier is None else carrier.key, [clear_number(probs, min_prob) for probs in jersey]


def off_roster_obs(half: int, key: tuple, player: str) -> str:
    """Observation id of a track reading as a player memory has off the pitch:
    "read:<half>/<shot>/<track>:<player hash>"."""
    return f"read:{track_name((half, *key))}:{player}"


def obs_id(call_id: str, i: int) -> str:
    """Observation id of the i-th event candidate of an analyzer call: "obs:<call id>#<i>"."""
    return f"obs:{call_id}#{i}"


def row_sources(row: dict) -> dict[str, dict]:
    """Observation id -> the analyzer window and candidate behind it, for one windows.jsonl row."""
    return {obs_id(row["call_id"], i): {"window": row["window"], "candidate": c}
            for i, c in enumerate(row.get("candidates", []))}


def memory_dir(config: dict[str, Any], match: Match, run: str) -> Path:
    """outputs/memory/<split>/<game>/<run>/, where build_memory() writes a run."""
    return config["paths"]["outputs"] / "memory" / match.key / run


def load_run(config: dict[str, Any], match: Match, run: str) -> tuple[MatchMemory, dict, dict]:
    """A finished build_memory() run read back, as the QA tools read it: (memory, perceived
    half -> second -> frame view, observation id -> its analyzer window and candidate)."""
    run_dir = memory_dir(config, match, run)
    if not (run_dir / "perceived_half2.pkl").exists():
        raise FileNotFoundError(f"no finished match memory in {run_dir}: run scripts/build_memory.py "
                                f"--split {match.split} --game {match.game} first")
    mem = MatchMemory.from_log(match.prematch["players"], run_dir / "ops.jsonl")
    perceived = {h: pickle.load(open(run_dir / f"perceived_half{h}.pkl", "rb")) for h in (1, 2)}
    sources: dict[str, dict] = {}
    for row in map(json.loads, open(run_dir / "windows.jsonl")):
        sources.update(row_sources(row))
    return mem, perceived, sources


def build_memory(config: dict[str, Any], match: Match, out_dir: Path, client) -> MatchMemory:
    """The whole match, live, from the raw perception cache.

    One match clock drives everything, half by half and frame by frame:
      - identity voting (perception.identity) on every frame: it votes only among the players
        memory has on the pitch at that moment, keeps memory's confirmed identity entries, and
        reports a number worn by nobody on the pitch that names an off-pitch player as evidence
        (consistency.off_roster_read);
      - at the end of every fixed window, the event analyzer's request is built from the
        frames and identities seen so far and sent through `client` (APIClient); its answer
        takes effect at the window end + the measured latency;
      - an entry still pending on one live sighting once the next window's answer has arrived
        is re-checked once: the analyzer, unchanged, on a window centred on that sighting,
        unless that window is one of the fixed ones; the last window of a half has no next
        one, so its lone sightings are not re-checked;
      - answers go through the consistency check; every event-level case goes to the arbiter
        at once, a perception case once it has `off_roster_k` sources; the verdict takes effect
        at the case's opening + the arbiter's summed latency, however late.
    The model calls are synchronous, so the wall time is the sum of their latencies; what memory
    holds at each match time is what a live deployment with the same latencies would have held.

    Writes to `out_dir`: ops.jsonl (the memory log — everything the QA reads), windows.jsonl
    (each window's call and candidates; a re-check's row has "recheck": <entry id> and "of":
    the sighting it re-checks), cases.json (each arbitrated case), and
    perceived_half<h>.pkl (second -> the frame as the analyzer was told about it, at the
    sampled seconds, for the QA's frame tools). Returns memory.
    """
    ev, prematch = config["events"], match.prematch
    out_dir.mkdir(parents=True, exist_ok=True)
    mem = MatchMemory(prematch["players"], log_path=out_dir / "ops.jsonl")
    checker = ConsistencyChecker(config)
    tools = EvidenceTools(config, match)
    arbiter = Arbiter(config["arbiter"], client.generate)
    squad = roster_index(prematch["players"])
    sources: dict[str, dict] = {}      # observation id -> the analyzer window and candidate behind it
    records: list[dict] = []           # one per arbitrated case
    windows_log = open(out_dir / "windows.jsonl", "w")
    # One timeline of things taking effect later: analyzer answers arriving, verdicts landing.
    queue: list = []
    order = itertools.count()
    n_seen = 0                         # cases already handled (case ids are handed out in order)
    # Re-checks: a lone live sighting is re-checked once the window after its own has had its say.
    after: dict[tuple, tuple] = {}     # (half, window) -> (half, the window after it)
    done: set[tuple] = set()           # (half, window) whose answer has arrived, or that failed
    rechecked: set[str] = set()        # entries re-checked already (once each)
    halves: dict[int, dict] = {}       # half -> the sampled frames a re-check is built from

    def arbitrate(case, t, opened=None, void=None):
        """Send `case` to the arbiter at match time t. `void` is why its previous verdict was void:
        the arbiter is told and rules once more. `opened` is when the case opened. There is no
        deadline: a valid verdict takes effect when it arrives, however late."""
        opened = opened or t
        packet = build_packet(mem, case, t, sources, prematch, config["arbiter"]["context"])
        if void:
            packet += (f"\n\nYour previous verdict on this case was void: {void}. Rule again; every write must cite "
                       "evidence ids returned by the tools in this investigation.")
        try:
            decision = arbiter.decide(packet, lambda tool, args: tools.call(mem, t, tool, args))
        except NoAPIKey:
            raise
        except Exception as e:  # noqa: BLE001 — out of API retries: the case stays open, nothing blocks
            decision = Decision(None, [], 0, 0.0, f"API failure: {type(e).__name__}: {str(e)[:200]}")
        record = {"case": case.case_id, "constraint": case.constraint, "opened": opened, "calls": decision.calls,
                  "latency": decision.latency, "verdict": decision.verdict, "retry": bool(void),
                  "evidence": len(decision.evidence_ids), "void": decision.note or None}
        records.append(record)
        print(f"  case {case.case_id} {case.constraint} at {t[0]}H {t[1]:.0f}s{' (again)' if void else ''}: "
              f"{'verdict' if decision.verdict else decision.note} ({decision.calls} calls, {decision.latency:.0f}s)",
              flush=True)
        if decision.verdict is None:
            # No usable answer (malformed JSON, out of rounds or API retries) is a void verdict
            # too: one more try, told why, once this one's time has passed.
            if not void:
                heapq.heappush(queue, ((t[0], t[1] + decision.latency), next(order), "retry",
                                       (case, opened, decision.note)))
            return
        heapq.heappush(queue, ((t[0], t[1] + decision.latency), next(order), "verdict", (record, decision, t)))

    def settle(t):
        # Event-level conflicts go to the arbiter at once; perception ones (off_roster_read)
        # only when off_roster_read() says they have enough sources.
        nonlocal n_seen
        new, n_seen = list(mem.cases.values())[n_seen:], len(mem.cases)
        for case in new:
            if case.constraint != "off_roster_read":
                arbitrate(case, t)

    def advance(t):
        """Apply everything that has taken effect by match time t, in time order."""
        while queue and queue[0][0] <= t:
            when, _, kind, payload = heapq.heappop(queue)
            if kind == "verdict":
                record, decision, t_bound = payload
                if mem.cases[record["case"]].status == "open":
                    record["void"] = apply_verdict(
                        mem, when, record["case"], decision.verdict, decision.evidence_ids, t_bound,
                        ev["live_margin"], lambda key: len(tools.track_frames(key, t_bound)) > 0,
                        lambda a, b: tools.cooccur(a, b, t_bound))
                    if record["void"] and not record["retry"]:
                        # A void verdict leaves the case open and its members pending for good;
                        # the arbiter gets one more try, told why.
                        arbitrate(mem.cases[record["case"]], when, record["opened"], record["void"])
                continue
            if kind == "retry":
                case, opened, note = payload
                if case.status == "open":
                    arbitrate(case, when, opened, note)
                continue
            checker.tick(mem, when)
            if kind == "recheck":
                half, eid, r = payload
                # The first candidate that is a sighting of the entry joins it (any() stops there);
                # the rest of the footage was analysed by the regular windows already.
                seen = any(checker.recheck(mem, when, eid, half, tuple(r["window"]), Candidate.from_dict(c),
                                           obs_id(r["call_id"], i), c["players"])
                           for i, c in enumerate(r["candidates"]))
                print(f"  re-check {eid}: {'seen again' if seen else 'not seen'}", flush=True)
                settle(when)
                continue
            half, r = payload
            for i, c in enumerate(r["candidates"]):
                checker.propose(mem, when, half, tuple(r["window"]), Candidate.from_dict(c),
                                obs_id(r["call_id"], i), c["players"])
            settle(when)
            done.add((half, tuple(r["window"])))
            for eid in due():
                recheck(eid, when)

    def due():
        """The entries to re-check now: pending on a single live sighting (not a replay), whose
        next window's answer has arrived (or failed) without seeing it again."""
        out = []
        for eid, e in mem.entries.items():
            if not isinstance(e, Event) or eid in rechecked:
                continue
            js = [j for _, j in mem.justifications[eid]]
            if (len(js) != 1 or js[0].kind != "observation" or js[0].sufficient or js[0].obs_id not in sources
                    or mem.status(eid) != "pending"):
                continue
            src = sources[js[0].obs_id]
            if src["candidate"]["replay"]:
                continue
            half = e.t_interval[0][0]
            if after.get((half, tuple(src["window"]))) in done:
                out.append(eid)
        return out

    def analyze(half, window, **extra):
        """One event-analysis call on `window` of `half`, its request built from the frames and
        identities seen so far. The candidates carry the player behind each participant track as
        the model was told. The call is logged to windows.jsonl with `extra` (a re-check's entry)
        and its observations are registered. Returns the logged row; out of retries, the row has
        "error" instead."""
        h = halves[half]
        shown = [(s, h["images"][s]) for s in h["grid"] if window[0] <= s < window[1] and s in h["images"]]
        request = build_request(half, window, shown, h["perceived"], prematch, ev["sample_fps"])
        # The player behind each track as last seen in the window — what the model was told.
        players_of = {(half, *tr.key): tr.player_id
                      for s, _ in shown for tr in h["perceived"].get(s, ((), None, ()))[0] if tr.player_id}
        try:
            result = client.generate(request.parts)
        except NoAPIKey:
            raise
        except Exception as e:  # noqa: BLE001 — out of retries: live, this call is simply lost
            r = {"half": half, "window": window, **extra, "error": f"{type(e).__name__}: {e}"}
        else:
            cands = [{**c.__dict__, "players": [players_of.get(key) for _, key in c.participants]}
                     for c in parse_candidates(result.text, request)]
            r = {"half": half, "window": window, "call_id": result.call_id, "model": result.model,
                 "latency": result.latency, "cached": result.cached, **extra, "candidates": cands}
            sources.update(row_sources(r))
        windows_log.write(json.dumps(r) + "\n")
        windows_log.flush()
        return r

    def recheck(eid, t):
        """Re-check entry `eid` at match time t: the analyzer, unchanged, on a window of the same
        length centred on its sighting; the answer takes effect at t + the
        measured latency. A window that is one of the fixed ones was analysed already: asking
        again would be the same request, so the entry is not re-checked."""
        rechecked.add(eid)
        e = mem.entries[eid]
        half = e.t_interval[0][0]
        obs_id = mem.justifications[eid][0][1].obs_id
        second, h = sources[obs_id]["candidate"]["second"], halves[half]
        length = ev["window_seconds"]
        w0 = max(0.0, min(second - length / 2, h["grid"][-1] - length))
        window = (w0, w0 + length)
        if window in h["windows"]:
            print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s: skipped, a fixed window", flush=True)
            return
        # Logged whole: only a candidate that is a sighting of this entry will join it.
        r = analyze(half, window, recheck=eid, of=obs_id)
        if "error" in r:
            print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s failed: {r['error']}", flush=True)
            return
        print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s: {len(r['candidates'])} candidates, "
              f"{r['latency']:.1f}s{' cached' if r['cached'] else ''}", flush=True)
        heapq.heappush(queue, ((t[0], t[1] + r["latency"]), next(order), "recheck", (half, eid, r)))

    for half in (1, 2):
        arrays, meta = load_perception(half_cache(config, match, half))
        fps = meta["fps"]
        duration = meta["n_frames"] / fps
        windows = fixed_windows(duration, ev["window_seconds"], ev["window_overlap"])
        # Sample times on one grid for the whole half, so overlapping windows share frames.
        step = 1 / ev["sample_fps"]
        grid = [k * step for k in range(int(duration / step) + 1)]
        wanted = {int(round(s * fps)): s for s in grid}
        resolver = IdentityResolver(config["perception"]["identity"], kit_lab(prematch["jersey_colors"]))
        cap = cv2.VideoCapture(str(match.video_path(half)))
        video = {"pos": 0}
        images: dict[float, np.ndarray] = {}      # sampled frames a current or later window still needs
        perceived: dict[float, tuple] = {}        # sampled second -> frame_view()
        reads_seen: set[str] = set()
        roster, confirmed, version = squad, {}, None
        halves[half] = {"images": images, "perceived": perceived, "grid": grid, "windows": set(windows)}
        after.update({(half, a): (half, b) for a, b in zip(windows, windows[1:])})

        def fire(window):
            """Window over: build the analyzer's request from what has been seen, call the model,
            and queue the answer for when it arrives."""
            t = (half, window[1])
            advance(t)
            checker.tick(mem, t)
            settle(t)
            while video["pos"] <= int(round(window[1] * fps)):
                if not cap.grab():
                    break
                if video["pos"] in wanted:
                    ok, image = cap.retrieve()
                    if ok:
                        images[wanted[video["pos"]]] = image
                video["pos"] += 1
            r = analyze(half, window)
            if "error" in r:
                print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s failed: {r['error']}", flush=True)
                done.add((half, tuple(window)))
                return
            print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s {len(r['candidates'])} candidates, {r['latency']:.1f}s"
                  f"{' cached' if r['cached'] else ''}", flush=True)
            heapq.heappush(queue, ((half, window[1] + r["latency"]), next(order), "answer", (half, r)))
            # A re-check is sent once a sighting's window and the one after it have answered, on
            # frames from half a window before the sighting's window: keep them back to the
            # earliest window still waiting for either answer.
            waiting = next(a for a in windows if not {(half, a), after.get((half, a))} <= done)
            horizon = waiting[0] - ev["window_seconds"] / 2
            for s in [s for s in images if s < horizon]:
                del images[s]

        w = 0
        for frame_idx, tracks, jersey, color, det in iter_frames(arrays):
            second = frame_idx / fps
            while w < len(windows) and windows[w][1] <= second:
                fire(windows[w])
                w += 1
            advance((half, second))
            if len(mem.ops) != version:
                # Memory changed: the players on the pitch now, and the confirmed identities of
                # this half's tracks.
                version = len(mem.ops)
                view = mem.at((half, second))
                on = set(view.on_field["home"] + view.on_field["away"])
                roster = roster_index([p for p in prematch["players"] if p["hash"] in on])
                confirmed = {key[1:]: pid for key, pid in mem.track_players().items() if key[0] == half}
            tracks = resolver.update(frame_idx, tracks, jersey, color, roster, confirmed)
            for key, player in resolver.off_roster_reads({tr.key: tr.team for tr in tracks}, roster, squad):
                obs = off_roster_obs(half, key, player)
                if obs in reads_seen:
                    continue
                reads_seen.add(obs)
                case_id = checker.off_roster_read(mem, (half, second), obs, player)
                settle((half, second))
                if case_id:
                    arbitrate(mem.cases[case_id], (half, second))
            if frame_idx in wanted:
                perceived[wanted[frame_idx]] = frame_view(config, tracks, jersey, det)
        while w < len(windows):
            fire(windows[w])
            w += 1
        cap.release()
        # Everything of this half lands before the next half starts ((half, s) < (half + 1, 0)),
        # re-checks included; then its frames are no longer needed.
        advance((half, float("inf")))
        images.clear()
        del halves[half]
        with open(out_dir / f"perceived_half{half}.pkl", "wb") as f:
            pickle.dump(perceived, f)

    windows_log.close()
    (out_dir / "cases.json").write_text(json.dumps(records, indent=1, default=str))
    return mem
