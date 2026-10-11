"""Streaming question answering: a question asked during the match, answered from what the
system knows at that moment.

The agent reads memory by status through tools (plus the video up to now) in rounds of plain
JSON, like the arbiter: {"requests": [...]} for more information, or {"answer": "B"}. Every
round is one cached model call. The tools are lent by the runtime (runtime.qa_tools), bounded
by the question time; this module holds no memory, perception or video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from matchlive.arbiter.agent import Evidence
from matchlive.events.analyzer import clock, minute_phrase, parse_json

# The instructions of every round. The last sentence before {last}: every option set is forced
# choice, so an answer that picks none counts as wrong, and a model that declines to guess when
# unsure (claude-opus-5-5 often does) would be scored on its caution.
INSTRUCTIONS = """You answer questions about a football (soccer) match while it is being broadcast live. You see
the match through tools: the match memory (the events an automatic system has recognised so far) and the video up
to now. Nothing after now can be seen. Round {round} of at most {max_rounds}.

Tools (at most {max_requests} requests per round):
- get_state {{}}: the score, who is on the pitch and on the bench, substitutions and cards so far, each team's
  event counts.
- list_events {{"type": "shot", "team": "home", "player": "Costa", "half": 1, "start": 600, "end": 900}}: the events
  memory holds, oldest first; every filter is optional ("start"/"end" are seconds of that half; use
  {{"last_minutes": 10}} instead of half/start/end for the last N minutes up to now, across half-time).
- get_player {{"name": "Costa"}}: a player's number, team, position, where they are now, and their events.
- why {{"entry": "E12"}}: what an event rests on (which analyses saw it, arbiter verdicts).
- get_frames {{"half": 1, "start": 600, "end": 620}}: up to {max_frames} broadcast frames spread over a span before
  now; tracked players are labelled P1, P2, ... and the text says who each is, as far as shirt-number reading
  knows by now, and who is nearest the ball.
- get_event_evidence {{"entry": "E12"}}: the same, for the seconds around an event.
- get_current_view {{}}: the same, for the last {current_view} seconds up to now.

About the match memory:
- An event reaches memory 30-90 s after it happens (the video is analysed in 30-s windows, then checked), so
  what is happening now or happened in the last minute or two is not in it yet.
- Each event is confirmed (two independent sightings agree, or an arbiter checked it) or pending (seen once so
  far, or disputed). Pending events are often real; do not ignore them, but a pending event right next to one
  of the same type and team may be the same thing reported twice (e.g. from a replay).
- The players on an event come from reading shirt numbers and are often unidentified.
- A goal also counts as a shot. Cards are yellow and red cards. A corner or free kick counts when it is taken.
- Times are the clock of each half: "half 2, 15:04" is 15 min 4 s into the second half. The Nth minute of a half
  runs from (N-1):00 to (N-1):59, so 15:04 of the second half is "4 seconds into the 16th minute of the second
  half". Stoppage time starts at 45:00 of a half: 45:45 of the second half is "45 seconds into the 1st minute of
  second-half stoppage time".
- For what memory does not hold yet, and for who did something when memory does not say, look at the video:
  get_current_view for now, get_event_evidence or get_frames for the past.

Answer with JSON only: {{"requests": [{{"tool": "get_state", "args": {{}}}}]}} to gather information, or
{{"answer": "B"}} with the letter of the option you choose. Always choose one option: if you cannot be sure, give your best guess.{last}"""

@dataclass
class Answer:
    choice: str | None              # the option letter, None if the model gave none in budget
    calls: int
    latency: float                  # seconds, summed over the calls
    tools: list[str] = field(default_factory=list)


def instructions(config: Mapping[str, Any], round: int) -> str:
    """INSTRUCTIONS for round `round` under the `qa` config section."""
    last = "\nThis is the last round: answer now." if round == config["max_rounds"] else ""
    return INSTRUCTIONS.format(round=round, max_rounds=config["max_rounds"], max_requests=config["max_requests"],
                               max_frames=config["max_frames"], current_view=config["current_view"], last=last)


def question_prompt(prematch: Mapping[str, Any], question: str, options: Sequence[str],
                    now: tuple[int, float]) -> tuple[str, str]:
    """The question as the agent reads it — the match, the time, the question and its options
    lettered A, B, ... — and those letters."""
    letters = "ABCDEFGH"[: len(options)]
    teams, kits = prematch["teams"], prematch["jersey_colors"]
    return (f"The match: {teams['home']} (home, {kits['home'][0]} kit) vs {teams['away']} (away, {kits['away'][0]} kit).\n"
            f"It is now half {now[0]}, {clock(now[1])} ({minute_phrase(*now)}).\n\nQuestion: {question}\nOptions:\n"
            + "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))), letters


class QAAgent:
    def __init__(self, config: Mapping[str, Any], generate: Callable[[Sequence[str | bytes]], Any]):
        # `config` is the `qa` section; `generate(parts)` is one JSON-answer model call returning
        # an object with .text and .latency (matchlive.common.api.APIClient).
        self.cfg = config
        self.generate = generate

    def answer(self, question: str, letters: Sequence[str],
               call_tool: Callable[[str, Mapping[str, Any]], Evidence]) -> Answer:
        """Rounds of tool requests until an answer, at most max_rounds calls."""
        evidence: list[tuple[str, Evidence]] = []
        out = Answer(None, 0, 0.0)
        for r in range(1, self.cfg["max_rounds"] + 1):
            parts: list[str | bytes] = [instructions(self.cfg, r), question]
            for request, ev in evidence:
                parts += [f"[{request}] {ev.text}", *ev.images]
            result = self.generate(parts)
            out.calls, out.latency = out.calls + 1, out.latency + result.latency
            reply = parse_json(result.text)
            out.choice = _letter(reply.get("answer"), letters)
            if out.choice:
                return out
            requests = reply.get("requests")
            if r == self.cfg["max_rounds"]:
                return out
            if not isinstance(requests, list) or not requests:
                # Neither an answer nor a request: often a plan in prose ("let me check the video")
                # with the JSON left out. The next round sees it and is reminded of the format.
                evidence.append(("your previous reply, which had no JSON", Evidence(
                    result.text.strip()[:1000] + '\n(Reply with JSON only: {"requests": [...]} or {"answer": "B"}.)')))
                continue
            for req in requests[: self.cfg["max_requests"]]:
                if not isinstance(req, dict):
                    continue
                tool, args = str(req.get("tool")).strip(), req.get("args") or {}
                out.tools.append(tool)
                evidence.append((f"{tool} {args}", call_tool(tool, args)))
        return out

def _letter(answer: Any, letters: Sequence[str]) -> str | None:
    # "B", or the option as written ("B. 1", "B)"): its leading letter.
    text = str(answer or "").strip()
    text = text[:1].upper() + text[1:]
    return text[0] if text and text[0] in letters and (len(text) == 1 or not text[1].isalpha()) else None
