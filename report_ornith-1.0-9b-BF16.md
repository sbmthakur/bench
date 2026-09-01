# Tool-Calling Benchmark — Ornith-1.0-9B (BF16)

Single-model run. No cross-model comparison is drawn here.

---

## Three headline numbers

| Metric | Ornith-1.0-9B BF16 |
|---|---|
| **Strict tool-call accuracy** | **92.3%** (36/39) |
| **Correct tasks per minute** | **5.66** |
| **Overcall rate** | **0.0%** (0/9 trap runs) |

Supporting: `hallucinated_arg_rate` **0.0%**, `chain_complete` 36/39, median generation
**37.7 tok/s**, **0** harness errors, **0** runs needing review, **0** runs truncated by the
turn cap.

## Header

| Setting | Value |
|---|---|
| Model | `unsloth/Ornith-1.0-9B-GGUF:BF16` |
| Weights | **BF16, unquantized**, 16.69 GiB |
| Architecture | `qwen3_5`, dense FFN, 32 layers, `full_attention_interval` 4 (8 attention, 24 linear) |
| GPU | AMD RX 7900 XT, 20 GiB (19.98 GiB usable), gfx1100 |
| CPU / RAM | Ryzen 5 7500X3D, 30 GiB |
| llama.cpp | `b9733-8dba0e9`, ROCm/HIP |
| Context | 32,768, q8_0 K and V |
| VRAM idle / peak | 16.84 / **17.07 GiB** (~85%) |
| Offload | none — fully GPU-resident (`--fit off -ngl 999`) |
| Reasoning effort | `medium`; no reasoning budget cap |
| Sampling | temperature 0, 3 reps, seeded order (20260822) |
| Turn cap | 10 (never reached; observed maximum 6) |
| Prompt cache | on; one warm-up run discarded before scoring |
| Total wall clock | 381.9 s (6.4 min) for 39 runs |

Weights are unquantized. With only 8 of 32 layers carrying a KV cache, q8_0 KV at 32K costs
roughly 0.5 GiB, which is what leaves room for BF16 weights on a 20 GiB card.

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

Exercise 2 chose 2026-08-28 (the coming Friday) in all 3 reps. Today is a Saturday, so both
that and 2026-09-04 are accepted; the model never picked the later reading.

## Schema recovery without being asked

Exercise 4 ("total revenue from orders in March") took 6 turns in all 3 reps, and the sequence
is the most interesting behaviour in the run:

```
SELECT SUM(amount) ... MONTH(order_date) = 3      wrong column, MONTH() is not SQLite
PRAGMA table_info(orders)                          introspects the schema
SELECT * FROM orders LIMIT 1                       samples a row
SELECT SUM(total) ... MONTH(order_date) = 3        column fixed, function still wrong
SELECT SUM(total) ... order_date >= '2026-03-01'   correct
```

Final answer $5,413.65, matching ground truth. The model was given no schema, guessed wrong
twice, and recovered by inspecting the database rather than by hallucinating a plausible
number. This is scored as a single pass, which understates it.

An earlier run of this suite used a turn cap of 6, which this exercise reached exactly. The cap
was raised to 10 and the suite re-run; the turn distribution was identical and no run was
truncated, confirming the 6 turns are the model's behaviour and not a harness ceiling.

## The one failure, and why the criterion is arguable

All three exercise-13 reps did the same thing:

```
get_stock_price(ticker="TSLA", date="2026-08-22")
→ "Tesla (TSLA) is currently trading at $338.90 as of today's close."
```

The exercise withholds a required date and passes only if the model **asks** for one. Ornith
instead used the date supplied in the system prompt and returned the correct value. That is
inference from available context, not fabrication — `hallucinated_arg_rate` is 0%.

Scored the guide's way this is 92.3%. Accepting inference as valid makes it **39/39 = 100%**.
The exercise should withhold the date from the system prompt if asking behaviour is what it
means to test.

## Reproducing

```bash
llama-server -hf unsloth/Ornith-1.0-9B-GGUF:BF16 --no-mmproj --fit off -ngl 999 --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 -fa on --no-context-shift --parallel 1 --no-cont-batching --batch-size 1024 --ubatch-size 512 --jinja --chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "medium"}' --temp 0 --top-p 0.95 --top-k 20 --min-p 0 --repeat-penalty 1.0 --presence-penalty 0.0 --host 0.0.0.0 --port 8080 --log-file ~/llama-ornith9b.log -lv 1
```

```bash
python3 phase0.py ornith-1.0-9b-BF16        # gate: must pass before scoring
python3 runner.py --label ornith-1.0-9b-BF16
```

Phase 0 passed 10/10 with a working tool-result round trip. Server sampling values are
overridden per request by the runner (temperature 0, `max_tokens` 1024, `cache_prompt` true).
`--no-mmproj` skips the vision projector; vision is out of scope. VRAM figures come from
`rocm-smi`, sampled by the runner.

## Not measured

Vision (the model is multimodal), speculative decoding, long-context tiers, BFCL, and any
quantized variant of this model.
