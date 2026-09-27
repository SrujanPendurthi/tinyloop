# tinyloop — design spec

Date: 2026-09-26
Status: draft for review

## 1. Intent

**Goal.** A small research language model, written entirely in Rust and trained and run on an Apple M4 (24 GB unified memory), that combines four ideas into a "tiny powerhouse":

- Gated DeltaNet linear attention (Qwen3-Next style)
- NSA-style block-sparse attention
- Recurrent depth (Huginn style: a shared core block looped `r` times)
- Mixture-of-experts FFNs

**Context target.** 1M tokens in v1. Unbounded (streaming, tiered/evicted KV) is phase 2.

**Success criteria.**
1. The model trains stably end to end on the M4 inside a memory cap.
2. Multi-hop retrieval accuracy rises with loop count `r` for the main config and stays flat for the attention-outside-the-loop baseline (the headline result).
3. Needle retrieval works at lengths well beyond the training length, up to 1M tokens.
4. Everything is reproducible: seeded data, resumable training, CSV results.

**Explicit non-goals for v1.** Competitive general text quality, comparisons against external models, serving, KV quantization, adaptive per-token loop count, RL, unbounded context.

**Constraints.**
- Rust only. `mlx-rs` is the framework (user choice).
- Training runs only on this Mac, in scheduled windows.
- The machine is shared with other Claude sessions that run 4–20 GB Laguna jobs. The rule is one heavy job at a time: announce before any build, training run, or 1M-token eval, and wait if a Laguna job is running. Never write to `lagunaLab/`.

## 2. Architecture

### 2.1 Shape

- ~90M total parameters; ~27M active non-embedding parameters per token (~52M counting the tied embedding), `d_model = 512`.
- Tokenizer: SmolLM2 (49,152 vocab), loaded with the `tokenizers` crate. Tied input/output embeddings (~25M params). The vocab fits in `u16`.

```
tokens → embed → PRELUDE (2 layers) → e
s₀ ~ N(0, σ²)
for i in 1..=r:  s = CORE(adapter(concat[s, e]))    # 4 layers, shared weights
CODA (2 layers) → RMSNorm → tied LM head
```

- Sandwich norm (RMSNorm before and after each sub-layer) in every layer, for loop stability.
- `adapter` is a linear map `2·d → d`.

### 2.2 Layer types

Each layer is a mixer (G or S) followed by an MoE FFN. Layer order per stage is a pattern string in the config, because sweeping the ratio is a planned experiment.

**G — Gated DeltaNet**
- 4 heads, `d_k = d_v = 128`.
- Short causal conv (kernel 4) on q, k, v; L2-normalised q and k.
- β = sigmoid(·) write strength; α = exp(−A·softplus(·)) decay.
- Output gate, then RMSNorm.
- Training form: chunked (chunk 64), WY / matmul formulation. Decode form: recurrent step.

**S — NSA-style sparse attention**
- GQA, 2 KV heads, `d_head = 64`.
- Three branches:
  - (a) compressed: one pooled key/value per 64-token block, visible to a token only once the block is complete. A zero-logit sink slot keeps early tokens (no complete block yet) finite;
  - (b) selected: full attention inside the top-16 blocks, scored by (a). In training and prefill, the selection is shared across the query heads of a GQA group **and across the 64 queries of a query block** (MoBA-style). Per-token selection would need a `[T, topk·64, d]` gather per head, about 4 GB at 16k tokens, which is not feasible without custom kernels. Decode selects per token (its query block is itself). A query block always selects its own block and never a future block;
  - (c) sliding window of 512 tokens.
- A learned sigmoid gate per branch mixes the three.
- NoPE (no positional encoding); position comes from DeltaNet's conv and recurrence.
- Log-length logit scaling (scalable softmax: logits × `s·log n`, `s` learned per head) in branches (a) and (b), to counter attention dilution at 1M tokens.

**MoE FFN**
- 8 routed experts (SwiGLU, hidden 512), top-2, plus 1 shared expert.
- Auxiliary-loss-free balancing: a per-expert bias is added to the router scores for selection only, and is nudged every step toward uniform load.

### 2.3 Configs

| Config | Prelude | Core (looped) | Coda | Role |
|---|---|---|---|---|
| main | `GG` | `GGGS` | `GG` | retrieval inside the loop; hop capacity grows with `r` |
| baseline | `GS` | `GGGG` | `GS` | same parameter count, retrieval only outside the loop |
| ratio sweep | `GG` | `GS` / `GGS` / `GGGS` | `GG` | G:S ratio inside the core |

### 2.4 KV cache across loop iterations

- **2a (default): per-iteration cache.** Iteration `i` has its own KV cache and DeltaNet states. This matches training exactly. `r` is fixed per sequence.
- **2b (`--shared-kv`): shared cache.** A single cache, written by the final iteration and read by every iteration. Memory stays flat in `r`, but there is a train/inference mismatch; the eval measures the gap against 2a.

### 2.5 Memory at 1M tokens

- One S layer's KV is ~0.5 KB/token (2 heads × 64 × K,V × bf16).
- Main config under 2a: ~4 GB at `r = 8`, ~8 GB at `r = 16`, plus ~0.4 GB for weights and buffers.
- G layers are constant: ~128 KB per layer per iteration.

## 3. Training

### 3.1 Budget

- Estimated ~0.5 GFLOP per training token, giving ~1.5–3k tokens/s on the M4. The spike measures the real figure.
- Total plan ≈ 650M tokens ≈ 3–5 machine-days.

### 3.2 Stages

| Stage | Data | Seq len | Tokens |
|---|---|---|---|
| 0 Spike | random tensors | up to 16k | — |
| 1 Pretrain | FineWeb-Edu | 4k | ~450M |
| 2 Long-context CPT | PG-19 + long FineWeb-Edu docs | 16k | ~120M |
| 3 Synthetic SFT | 70% synthetic tasks, 30% text replay | 16k; last 10% at 64k | ~80M |

- Ablation runs (baseline and ratio sweep) use 100M tokens each, about 12 h per run. They are compared against the main run's checkpoint at the same token count.
- Known limitation: ~450M pretraining tokens is below compute-optimal (~20 tokens per active param ≈ 550M–1B tokens). It can be extended later because of the WSD schedule.

### 3.3 Loop sampling

- `r` ~ log-normal Poisson, mean 6, clamped to [1, 16], sampled once per step for the whole batch.
- Truncated backprop: gradients flow through the last 4 iterations only; earlier iterations sit behind `stop_gradient`.

### 3.4 Optimisation

- AdamW (β = 0.9, 0.95; weight decay 0.1).
- WSD schedule, peak LR 1e-3, grad clip 1.0.
- Effective batch ~0.5M tokens via gradient accumulation.
- fp32 master weights, bf16 matmuls. The spike confirms this is stable and fast in `mlx-rs`.

### 3.5 Data pipeline

- Download with `hf-hub`; read with `parquet`.
- Tokenize once into flat `u16` shards, memory-mapped at training time (~1.3 GB total).
- Synthetic tasks are generated on the fly from a seed, not stored.

### 3.6 Coexisting with Laguna

- MLX memory limit defaults to 8 GB (flag-configurable). Micro-batch is tuned to fit.
- A checkpoint is written every 15 min and on SIGINT/SIGTERM. It holds weights, optimiser state, data cursor, and RNG state, so `--resume` reproduces the run exactly.
- Before every run, message the other sessions and wait if a Laguna job is active. The user picks the run windows.
- Per-step CSV log: loss, LR, tokens/s, per-expert load, peak memory.

## 4. Inference

- **Chunked prefill.** 16k-token chunks. Per chunk: prelude → iterations `1..r` → coda. Each iteration carries its own DeltaNet/conv state and KV cache (2a) across chunks.
- **Cache storage.** Block-addressed, 64-token blocks plus compressed per-block K/V. Buffers are preallocated in 64k-token steps and updated in place (no `concat`). A partial last block is not compressed; the window covers it.
- **Decode.** Per token, run `r` iterations; the S layer scores all compressed blocks, gathers the top 16, adds the window, and gates.
- **Rough performance.** 10–30 tok/s single stream at `r = 8`. 1M-token prefill at `r = 8` takes ~3–10 min.
- **Shared-prefix batched sampling.** The prompt cache and states are computed once and treated as read-only. Each of B samples owns a small suffix cache and copies of the DeltaNet states. Needed for eval speed and for phase-2 GRPO.
- **Sampling.** Greedy, temperature, top-p.
- **Formats.** `safetensors` weights plus `config.json`.
- **CLI.** `tinyloop generate --ckpt DIR --prompt-file F --r 8 [--shared-kv] [--samples 8 --temp 0.7]`.

## 5. Evaluation

`tinyloop eval` writes one CSV per suite. Charts are out of scope for v1.

- **Perplexity**
  - FineWeb-Edu held-out at `r ∈ {1, 2, 4, 8, 16}`.
  - Loss by token position on concatenated PG-19 out to 1M tokens.
- **Synthetic suite** (held-out seeds)
  - Single needle: lengths {4k, 16k, 64k, 256k, 1M} × 5 depths.
  - Multi-key and multi-value retrieval.
  - Variable tracking.
  - Multi-hop chains with 1–6 hops, both **in-order and out-of-order**, with distractors. The out-of-order chains are essential: in-order chains can be solved by forward-chaining at read time, which would let the baseline pass for the wrong reason.
  - 1M-length cells use 20 samples each and count as heavy jobs.
- **Headline comparisons**
  1. Multi-hop accuracy vs `r`, main vs baseline.
  2. Needle heatmap (length × depth).
  3. Accuracy vs FLOPs.
  4. 2a vs 2b.
  5. Ratio sweep.

## 6. Project layout

`Projects/Current/tinyloop/` is its own git repo: one crate, lib plus bin.

```
Cargo.toml                 mlx-rs, tokenizers, hf-hub, parquet, safetensors, serde, clap, anyhow
src/main.rs                CLI: prepare-data | train | generate | eval
src/config.rs              model + train config (JSON)
src/model/mod.rs           embed, prelude/core/coda, adapter, loop, stop_gradient
src/model/deltanet.rs      Gated DeltaNet: chunked + recurrent step
src/model/sparse_attn.rs   NSA-style S layer
src/model/moe.rs           MoE FFN + bias balancing
src/cache.rs               per-iteration block KV + states, shared-prefix batching
src/data.rs                download, tokenize, u16 shards, mmap loader
src/tasks.rs               synthetic generator + scorer
src/train.rs               optimiser, schedule, accumulation, checkpoint/resume, CSV log
src/generate.rs            prefill, decode, sampling
src/eval.rs                suites, r sweeps, CSV
tests/reference.rs         correctness tests
```

## 7. Testing

All tests run at tiny dimensions in fp32. None of them is a heavy job.

1. Chunked DeltaNet vs a naive token-by-token delta-rule loop: forward output and gradients match.
2. Gather-based sparse attention vs dense masked attention (mask built from the same top-k selection): forward output and gradients match.
3. MoE dispatch vs dense evaluation of all experts, weighted: outputs match.
4. Finite-difference gradient check on the full model (`r = 3`, backprop through 2 iterations).
5. Chunked prefill, step-wise decode, and batched greedy sampling each match the full forward pass (bf16 tolerance; ~8k tokens, 2k-token chunks).
6. A tiny model overfits one batch (loss → ~0).
7. Generator: deterministic per seed; every answer is verifiable; out-of-order chains really are out of order.
8. Stop-and-resume produces weights identical to an uninterrupted run.

Tests 1–4 plus a 16k-token throughput benchmark form the **stage-0 spike**. If `mlx-rs` lacks a required op (autodiff through gather/scatter, `topk`, in-place slice update, memory limit) or throughput is far below estimate, stop and revisit the framework choice (`burn` is the fallback).

## 8. Phase 2 (not v1)

- **Unbounded context.** Tier or evict cold KV blocks to SSD. Block selection frequency per iteration is the retention signal; evaluate with the `kvAdmission` oracle and screening methodology.
- **Stage 4 RL.** GRPO on synthetic tasks with verifiable rewards, using shared-prefix batched rollouts and a difficulty curriculum from the generator. Start only if stage-3 accuracy on the hard tasks lands between 20% and 80%. Optional compute-aware reward, which needs per-token adaptive `r` and therefore 2b.
- **Cache quantization.** Needed for `r > 16` or for unbounded context.
