#!/usr/bin/env bash
# Serve Qwen/Qwen3.8-27B locally with vLLM, as the OpenAI-compatible endpoint of configs/qwen.yaml
# (http://localhost:8000/v1). Optional: run it in the separate `matchlive-vllm` environment
# (README, "Local model"), in its own terminal, and leave it running while the scripts call it.
#
#   conda activate matchlive-vllm
#   bash scripts/serve_qwen.sh                     # all GPUs: TP=4 per replica, as many replicas as fit
#   TP=1 DP=1 CUDA_VISIBLE_DEVICES=0 bash scripts/serve_qwen.sh   # one 80 GB GPU
#
# The weights (55 GB, bf16) are downloaded from Hugging Face on first start. One replica per TP
# GPUs; vLLM spreads the requests over the DP replicas behind the one port. Building the match
# memory calls the model one request at a time per game, so its speed is single-request decoding
# speed: hence a wide TP rather than many replicas.
set -euo pipefail

# One machine: GPUs talk over NVLink, no network plugin is needed. A host-wide NCCL_NET (GCP A3
# images set NCCL_NET=gIB) names a plugin vLLM's own NCCL cannot load, and the TP group fails
# to start with "NCCL error: invalid usage".
unset NCCL_NET

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
TP=${TP:-4}
DP=${DP:-$((GPUS / TP))}
PORT=${PORT:-8000}

# --reasoning-parser: the model thinks first (on by default); the thinking goes to a separate
#   field, so the answer text holds only the answer.
# --limit-mm-per-prompt: a QA or arbiter round can carry the frames of several tool calls
#   (up to 16 each), and they add up over the rounds.
# --speculative-config: the model's built-in multi-token-prediction head drafts 3 tokens per
#   step; it speeds up decoding without changing what is sampled.
# Sampling is the model's own (generation_config.json: temperature 1.0, top_p 0.95, top_k 20),
#   except where a request sets it.
exec vllm serve Qwen/Qwen3.8-27B \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --port "$PORT" \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 256, "video": 0}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --gpu-memory-utilization 0.9
