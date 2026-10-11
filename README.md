# MatchLive baseline — SoccerNet Challenge 2027 Streaming VQA

MatchLive is the video-based baseline of the
[SoccerNet Challenge 2027 – Soccer Streaming Visual Question Answering](https://github.com/jyrao/soccernet-streaming-vqa-challenge).
It watches a match broadcast as a live stream, keeps a revisable memory of what has happened,
and answers each question at the moment it is asked, from the video seen so far.

**How this repository relates to the challenge repository.** The
[challenge repository](https://github.com/jyrao/soccernet-streaming-vqa-challenge) defines the
task, the rules, the submission format and the evaluation script. This repository is one way of
solving the task: it reads the challenge's question files and videos
([SoccerNet/SN-VQA-2027](https://huggingface.co/datasets/SoccerNet/SN-VQA-2027)) and writes a
submission file in the challenge's format (OSL JSON), which you score with the challenge's
`codes/eval_predictions.py` (test split) or submit to Codabench (challenge split). Use it as a
starting point, or as a reference to compare against.

## Results on the test split

Accuracy (%) on the 700 test questions, scored with the challenge's `codes/eval_predictions.py`:

| | counting | recall_event_time | recall_event_player | event_team | event_player | causal reasoning | **Average** |
|---|---|---|---|---|---|---|---|
| MatchLive (this repository, `gemini-3.8-flash`) | 58.3 | 82.0 | 96.1 | 96.6 | 96.9 | 76.9 | **84.5** |
| MatchLive (this repository, `Qwen3.8-27B` served locally) | 42.9 | 70.5 | 79.1 | 84.7 | 72.3 | 53.8 | 67.2 |
| Text-only baseline (GPT-5.5, no video), not run by this repository | 36.9 | 26.0 | 45.0 | 48.3 | 43.1 | 51.3 | 41.8 |

The predictions of this repository's rows are in `results/` of the
[Hugging Face repository](https://huggingface.co/Becomebright/MatchLive-baseline) that holds the weights.

## How it works

Three steps, one script each:

| Step | Script | Needs | What it does |
|---|---|---|---|
| 1. Perception | `scripts/run_perception.py` | NVIDIA GPU | One pass over each half video: player and ball detection (RF-DETR), tracking (BoT-SORT), camera cuts, jersey numbers (DINOv3-based classifier) and kit colours. Written to a cache. |
| 2. Match memory | `scripts/build_memory.py` | LLM API | Plays each match as a live stream, on one clock. Every 30 s of video, a vision-language model reports the events it sees in frames where each tracked player is labelled and identified as far as known. Reports become a match memory of confirmed and pending events; an arbiter model settles reports that contradict each other. Every model answer takes effect only after its measured latency, as it would live. |
| 3. Question answering | `scripts/run_qa.py` | LLM API | Each question is answered at its timestamp by an agent whose tools read the memory as known at that moment (score, line-ups, events and why they are believed) and show frames up to that moment. |

How the baseline follows the [challenge rules](https://github.com/jyrao/soccernet-streaming-vqa-challenge#-core-rules):

- **Visual input only.** Only video frames are read; there is no audio.
- **Streaming.** A question at time *t* is answered from the memory as written by *t* and from
  frames up to *t*; building the memory itself only ever looks at frames up to the current moment.
- **Permitted metadata only.** Beyond the video, the system knows team names, kit colours, the
  squads (shirt numbers, starting XI, positions) and which goal each team defends
  (`data/prematch/`, see [Pre-match metadata](#pre-match-metadata)).
- **No web search.** Model calls are plain chat-completion requests with no tools, search or
  browsing enabled. If you use another endpoint, make sure it adds no search grounding.

## Setup

**Requirements.** Linux x86_64 with conda. Perception needs an NVIDIA GPU whose driver supports
CUDA 13 (`nvidia-smi` shows the highest supported version); the other two steps need no GPU.
About 13 GB of disk for the challenge data, plus 0.25 GB for the weights and about 0.3 GB of
perception cache per match.

### 1. Code and environment

```bash
git clone https://github.com/Becomebright/MatchLive-baseline.git
cd MatchLive-baseline
conda create -n matchlive -c conda-forge --override-channels python=3.11 pip
conda activate matchlive
pip install -r requirements.txt
pip install -e . --no-deps        # makes `matchlive` importable from scripts/
```

The perception models run on ONNX Runtime with TensorRT (then CUDA, then CPU, whichever is
available); the CUDA, cuDNN and TensorRT libraries come as pip wheels, so no system CUDA toolkit
is needed. Keep the `tensorrt-cu13==10.16.*` pin: ONNX Runtime 1.30 cannot load TensorRT 11 and
would silently fall back to CUDA.

### 2. Model weights

Two ONNX models (233 MB) go to `checkpoints/`:

```bash
hf download Becomebright/MatchLive-baseline --include "checkpoints/*" --local-dir .
```

| File | Model |
|---|---|
| `checkpoints/rfdetr-l.onnx` | RF-DETR-L detector for ball, goalkeepers, players and referees |
| `checkpoints/jersey_dinov3_convnext_tiny-20_matches.onnx` | jersey-number classifier (DINOv3 ConvNeXt-tiny backbone + linear head, numbers 0–99 and "unreadable") |

### 3. Challenge data

The data is gated: accept the SoccerNet terms on
[its Hugging Face page](https://huggingface.co/datasets/SoccerNet/SN-VQA-2027), log in with
`hf auth login`, then:

```bash
hf download SoccerNet/SN-VQA-2027 --repo-type dataset --local-dir data/SN-VQA-2027
```

This gives `test.json`, `challenge.json`, `video-test/` and `video-challenge/` under
`data/SN-VQA-2027/`. If you already have the data elsewhere (for example in the challenge
repository's `data/`), link it instead: `ln -s /path/to/that/folder data/SN-VQA-2027`.

### 4. LLM API

Building the match memory and answering the questions call a vision-language model through any
OpenAI-compatible endpoint:

```bash
cp configs/api.example.yaml configs/api.local.yaml   # then fill in base_url and api_key
```

`configs/api.local.yaml` is gitignored. The model is `api.model` in `configs/default.yaml`
(`gemini-3.8-flash`, used for the results above); any model that accepts images and returns
JSON can be used. Every answer is cached under `cache/api/`, keyed by model and request, so an
interrupted or repeated run replays what it already has.

## Run

```bash
python scripts/run_perception.py --split test     # GPU: about 18 min per half on an H100
python scripts/build_memory.py --split test       # API: 1-2 h per match
python scripts/run_qa.py --split test             # API: about 30 min per match (8 questions at once)

# score it with the challenge repository's evaluator
python /path/to/soccernet-streaming-vqa-challenge/codes/eval_predictions.py \
    outputs/submission_test.json data/SN-VQA-2027/test.json --average-by subcategory
```

For the challenge split, use `--split challenge` and submit `outputs/submission_challenge.json`.

- Each script runs every game of the split; `--game game1` (repeatable) runs some of them, so
  games can run in parallel processes (for perception, one per GPU with `CUDA_VISIBLE_DEVICES`).
- Perception skips halves that are already cached. On first use TensorRT builds its engines
  (a few minutes) and caches them under `cache/trt/`. To check the setup on a short segment:
  `python scripts/run_perception.py --split test --game game1 --half 1 --start 600 --seconds 60`.
- `run_qa.py` writes the submission from every game answered so far; questions of games not yet
  answered are `null` (unanswered).

| Output | Written by |
|---|---|
| `cache/perception/<split>/<game>/half{1,2}_all_<key>.npz` | step 1: detections, tracks, jersey-number probabilities and kit colours, frame by frame |
| `outputs/memory/<split>/<game>/full/` | step 2: `ops.jsonl` (the match memory, op by op), `windows.jsonl` (every event report), `cases.json` (every arbitration), `perceived_half{1,2}.pkl` |
| `outputs/qa/<split>/full/<game>.jsonl` | step 3: one line per question: the answer, model calls, tools used |
| `outputs/submission_<split>.json` | step 3: the submission (OSL JSON, `prediction` = option id or `null`) |

## Local model: Qwen3.8-27B (optional)

Steps 2 and 3 can use [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) served on your own
GPUs by vLLM instead of an API. vLLM goes into its own environment (it brings its own PyTorch and
CUDA libraries); the steps themselves still run in `matchlive`.

```bash
conda create -n matchlive-vllm -c conda-forge --override-channels python=3.12 pip
conda activate matchlive-vllm
pip install vllm==0.30.0
bash scripts/serve_qwen.sh        # downloads the weights (55 GB) on first start; keep it running
```

`scripts/serve_qwen.sh` serves on `http://localhost:8000/v1`, one replica per 4 GPUs (the
reported run: 8 H100 80 GB); set `TP` for another number of GPUs per replica. Then, in the
`matchlive` environment, with the perception cache of step 1:

```bash
python scripts/build_memory.py --split test --name qwen --config configs/qwen.yaml   # 1.3-1.7 h per match
python scripts/run_qa.py --split test --run qwen --config configs/qwen.yaml          # -> outputs/submission_test_qwen.json
```

`configs/qwen.yaml` is `configs/default.yaml` with the model, its endpoint and
`reasoning_effort: medium` (Qwen's default effort makes each analyzer call take minutes).

## Pre-match metadata

The challenge question files carry no information about the matches, while the rules permit
pre-match metadata. `data/prematch/<split>.json` provides it for every game of both splits:

```json
{"game1": {"home": {"name": "Chelsea",
                    "kit_color": {"name": "blue", "hex": "#0000FF"},
                    "goalkeeper_kit_color": {"name": "cyan", "hex": "#79B1BC"},
                    "defends": {"1": "right", "2": "left"},
                    "players": [{"id": "CMGUADgt", "name": "Azpilicueta C.", "short_name": "Azpilicueta",
                                 "long_name": "Cesar Azpilicueta", "shirt_number": "28",
                                 "starting": true, "position": "defender"}, ...]},
           "away": {...}}}
```

- `name`: the team as the questions spell it.
- `kit_color`: the kit the team wears in this match (a name for the prompts, a rough colour for
  naming the two kit-colour clusters found in the video); `goalkeeper_kit_color` is optional.
- `defends`: the side of the main camera's picture whose goal the team defends in each half.
- `players`: the 18-man squad with shirt numbers, the starting XI and each starter's position
  (goalkeeper, defender, midfielder or forward, from the announced formation).

Squads come from the SoccerNet match annotations (`Labels-caption.json`); kit colours and playing
directions were annotated from the broadcast. Nothing about what happens during the match
(goals, cards, substitutions, final score) is included.

## Configuration

Settings are in `configs/default.yaml`; every script takes `--config <file>` to use another
copy. The ones you are most likely to change:

| Setting | Default | Meaning |
|---|---|---|
| `paths.data` | `data/SN-VQA-2027` | the challenge data |
| `api.model` | `gemini-3.8-flash` | the vision-language model of steps 2 and 3 |
| `qa.max_rounds` | 4 | model calls per question (the last one must answer) |
| `qa.max_frames` | 16 | frames per frame-tool call |
| `events.window_seconds` | 30 | length of the video windows the event analyzer reports on |
| `arbiter.max_rounds` | 4 | model calls per conflict |

`run_qa.py --workers N` sets how many questions are answered at once (default 8).

## Repository layout

```
configs/            default.yaml (all settings), api.example.yaml (endpoint template),
                    qwen.yaml + api.qwen.yaml (the local Qwen model)
data/prematch/      pre-match metadata of the test and challenge games
matchlive/
  perception/       detection, tracking, camera cuts, jersey numbers, kit colours, identity voting
  events/           event analyzer prompts, fixed windows, consistency check
  memory/           the match memory: an append-only log of beliefs and their evidence
  arbiter/          arbiter prompts and the checks a verdict must pass
  runtime/          building the match memory, and the tools lent to the arbiter and to the QA agent
  app/qa.py         the question-answering agent
  common/           config, data loading, API client with cache, perception cache
scripts/            run_perception.py, build_memory.py, run_qa.py, serve_qwen.sh
```

## License

The code is released under the [Apache License 2.0](LICENSE). The model weights are distributed
under their own terms; the jersey-number classifier is built on a DINOv3 backbone and is subject
to Meta's DINOv3 License. The challenge videos and questions are provided by SoccerNet under its
data agreement, and the squads in `data/prematch/` come from SoccerNet annotations.

## Citation

The reference paper is coming soon; its citation will be added here.
