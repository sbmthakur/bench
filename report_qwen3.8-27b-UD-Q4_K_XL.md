# Tool-Calling Benchmark — Qwen3.8-27B (UD-Q4_K_XL)

Single-model run. No cross-model comparison is drawn here.

---

## Three headline numbers

| Metric | Qwen3.8-27B UD-Q4_K_XL |
|---|---|
| **Strict tool-call accuracy** | **92.3%** (36/39) |
| **Correct tasks per minute** | **4.69** |
| **Overcall rate** | **0.0%** (0/9 trap runs) |

Supporting: `hallucinated_arg_rate` **0.0%**, `chain_complete` 36/39, median generation
**27.97 tok/s**, **0** harness errors, **0** runs needing review, **0** runs truncated by the
turn cap.

## Header

| Setting | Value |
|---|---|
| Model | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` |
| Weights | Q4_K_XL (Unsloth dynamic), 16.35 GiB |
| Architecture | `qwen35` hybrid — 65 blocks: 16 attention, 48 SSM, 1 `nextn` (MTP) |
| GPU | AMD RX 7900 XT, 20 GiB (19.98 GiB usable), gfx1100 |
| CPU / RAM | Ryzen 5 7500X3D, 30 GiB |
| llama.cpp | `b9733-8dba0e9`, ROCm/HIP |
| Context | 32,768, q8_0 K and V |
| VRAM idle / peak | 18.71 / **19.06 GiB** (~95%) |
| Offload | none — fully GPU-resident (`--fit off -ngl 999`) |
| Reasoning effort | `medium`; no reasoning budget cap |
| Sampling | temperature 0, 3 reps, seeded order (20260822) |
| Turn cap | 10 (never reached; observed maximum 7) |
| Prompt cache | on; one warm-up run discarded before scoring |
| Total wall clock | 460.8 s (7.7 min) for 39 runs |

Quant rationale: Q4_K_XL is the highest tier that fits fully on the GPU. Only 16 of 65 layers
carry a KV cache, so q8_0 KV at 32K costs roughly 0.5 GiB, leaving room for the weights.

## Per-tier results

| Tier | Exercises | Strict | Notes |
|---|---|---|---|
| T1 — schema fidelity | 1–3 | **9/9** | enum, relative date, float expression all exact |
| T2 — tool selection | 4–8 | **15/15** | no overcalls; `send_email` trap never fired |
| T3 — multi-step | 9–11 | **9/9** | result-threading correct; 3/3 parallel in one turn |
| T4 — error recovery | 12–13 | **3/6** | ex 12 clean (`APPL` → `AAPL`); ex 13 failed all 3 reps |

Determinism: **zero disagreement across 3 reps at temperature 0** on all 13 exercises,
compared on the tool-call `function` payload (llama.cpp assigns a random `id` per call, which
will falsely flag nondeterminism if included).

Exercise 2 chose 2026-08-28 (the coming Friday) in all 3 reps; 2026-09-04 is also accepted.

## Schema recovery without being asked

Exercise 9 ("city of the customer who placed order #1042, then its weather") took 7 turns in
all 3 reps:

```
SELECT c.city ... o.customer_id = c.customer_id WHERE o.order_id = 1042   wrong on both joins
SELECT * FROM orders LIMIT 1                                              samples orders
SELECT c.city ... WHERE o.id = 1042                                       order key fixed
SELECT * FROM customers LIMIT 1                                           samples customers
SELECT c.city ... ON o.customer_id = c.id WHERE o.id = 1042               correct
get_weather(city="New York", unit="celsius")                              threaded result
```

Final answer names New York at 19°C, matching ground truth. Given no schema, the model guessed
wrong column names twice and recovered by sampling both tables rather than fabricating a city.

**This exercise was previously mis-scored.** An earlier run of this suite used a turn cap of 6.
Exercise 9 needs 7, so all three reps were cut off after issuing `get_weather` but before the
result could be returned — no final answer was ever produced. They still scored as passes,
because the exercise-9 scorer checked call ordering and the threaded city but not completion.
The cap was raised to 10 and `chain_complete` now returns false for any run that exhausts it.
The effect on the headline number was material: correct tasks/min was **4.97** under the
truncating cap and is **4.69** now, because the truncated runs were skipping work they should
have done.

## The one failure, and why the criterion is arguable

All three exercise-13 reps did the same thing:

```
get_stock_price(ticker="TSLA", date="2026-08-22")
→ "Tesla (TSLA) closed at $338.90 today (August 22, 2026)."
```

The exercise withholds a required date and passes only if the model **asks** for one. The model
instead used the date supplied in the system prompt and returned the correct value. That is
inference from available context, not fabrication — `hallucinated_arg_rate` is 0%.

Scored the guide's way this is 92.3%. Accepting inference as valid makes it **39/39 = 100%**.
The exercise should withhold the date from the system prompt if asking behaviour is what it
means to test.

## Config findings

**`--ctx-checkpoints 0` costs an order of magnitude.** Carried over from a widely-shared
config, it disables prefix reuse entirely. Measured on this model:

| | tokens prefilled | prefill ms |
|---|---|---|
| `--ctx-checkpoints 0` | 843 | 1264.0 |
| default (32) | **15** | **118.2** |

**10.7× on a short prompt, and it widens with context** — at ~13K tokens the default
reprocessed ~1,030 tokens per turn instead of ~13,000. Context checkpoints matter more on
hybrid SSM models than on plain transformers: recurrent state cannot be truncated and resumed
the way a KV cache can, so checkpoints are the only route to mid-sequence restore. Cost at
~13K context: 419 MiB.

**Over-thinking did not materialise.** `reasoning_effort: medium` produced a median of 61
reasoning words per run. A `--reasoning-budget` cap was tested and dropped — it never fired,
and capping would suppress the exact tradeoff `tasks/min` exists to measure.

## Reproducing

```bash
llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --no-mmproj --fit off -ngl 999 --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 -fa on --no-context-shift --parallel 1 --no-cont-batching --batch-size 1024 --ubatch-size 512 --jinja --chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "medium"}' --temp 0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1.0 --presence-penalty 0.0 --host 0.0.0.0 --port 8080 --log-file ~/llama-q38-git.log -lv 1
```

```bash
python3 phase0.py qwen3.8-27b-UD-Q4_K_XL        # gate: must pass before scoring
python3 runner.py --label qwen3.8-27b-UD-Q4_K_XL
```

Notes on the flags that matter:

- `--fit off -ngl 999` — `-ngl` defaults to `auto`, which silently leaves layers on the CPU
  rather than failing. Explicit values make an over-large config OOM loudly.
- **`--ctx-checkpoints` is deliberately absent**, leaving the default of 32. Setting it to 0
  disables prefix reuse on this architecture and costs ~10x on prefill.
- `--no-mmproj` skips the 0.87 GiB vision projector; vision is out of scope.
- `--parallel 1` — the default is auto, which allocates 4 slots and divides the context.
- Server sampling values are overridden per request by the runner (temperature 0, `max_tokens`
  1024, `cache_prompt` true), so they only matter for manual testing.
- `--host 0.0.0.0` binds every interface. Use `127.0.0.1` unless LAN access is wanted.
- `--log-file` did not capture the `load_tensors` block even at `-lv 1`; VRAM figures come from
  `rocm-smi`, sampled by the runner.

Phase 0 passed 10/10 with a working tool-result round trip. Requires Python 3 (standard library
only) and `rocm-smi` on PATH for VRAM sampling.

## Not measured

Vision (present in this model), MTP speculative decoding (a draft file ships with the repo and
`blk.64.nextn.*` is in the GGUF — upside not quantified), long-context tiers, BFCL.
