# Tool-Calling Benchmark

A tight tool-calling eval for **open-weight models** run on personal AMD hardware
(RX 7900 XT, 20 GB, ROCm / `llama.cpp-hip`). Thirteen exercises drive real agentic
conversations against an OpenAI-compatible endpoint; tool calls are dispatched through a
deterministic mock toolset. Headline metric is **correct tasks per minute**, not accuracy alone.

## Layout

| File | Role |
|---|---|
| `mock_tools.py` | 6 deterministic tools (seeded SQLite, no network/clock/randomness) |
| `exercises.py` | 13 prompts (verbatim from the spec) + auto-scorers |
| `runner.py` | Agentic loop, scoring, JSONL output |
| `phase0.py` | Mandatory sanity gate — must pass before scoring |
| `runs_*.jsonl` | Per-run calls + timings (39 runs per model) |
| `phase0_*.jsonl` | Gate record |
| `report_*.md` | Per-model results, config findings, and reproduction steps |
| `tool-calling-benchmark.md` | The spec: 13 verbatim prompts + ground-truth values |

## Requirements

- Python 3 (standard library only)
- `rocm-smi` on `PATH` (for VRAM sampling)
- A `llama-server` instance serving the target model at `http://localhost:8080`

## Run

1. Start the server (exact command and flag rationale in each `report_*.md` → Reproducing):

   ```bash
   llama-server -hf <your-model> --jinja --temp 0 --parallel 1 --host 127.0.0.1 --port 8080 ...
   ```

2. Pass the sanity gate:

   ```bash
   python3 phase0.py <label>
   ```

3. Run the suite:

   ```bash
   python3 runner.py --label <label>
   ```

Output: `runs_<label>.jsonl` (one record per run) and a scored summary.

## Results so far

All runs: 13 exercises x 3 reps = 39 runs, temperature 0, seeded order, turn cap 10,
32K context, fully GPU-resident on a single RX 7900 XT (20 GB).

| Model | Quant | Strict | Correct tasks/min | Median tok/s | VRAM peak |
|---|---|---|---|---|---|
| Ornith-1.0-9B | UD-Q8_K_XL | 36/39 (92.3%) | **7.88** | 51.6 | 12.97 GiB |
| Ornith-1.0-9B | BF16 | 36/39 (92.3%) | 5.66 | 37.7 | 17.07 GiB |
| Qwen3.8-27B | UD-Q4_K_XL | 36/39 (92.3%) | 4.69 | 28.0 | 19.06 GiB |

Overcall rate is 0.0% and no run was truncated by the turn cap in any of the three.

**Read the accuracy column with care.** Every model and every quant scores 92.3%, failing the
same single exercise the same way (exercise 13, where the model uses the date supplied in the
system prompt instead of asking for one). That is the suite reaching its ceiling, not evidence
the models are equally capable. The accuracy axis does not currently discriminate; `correct
tasks/min` does. See the per-model reports for the full breakdown and the argument that
exercise 13's pass criterion is itself questionable.

## Known limitations

- **Exercises 6 and 7 are weak overcall traps.** They are general-knowledge questions with no
  plausibly applicable tool, so passing them shows little. Exercise 8 (draft an email with
  `send_email` offered) is the only strong trap of the three.
- **Exercise 8's scorer is heuristic.** It checks for draft markers ("subject", "dear",
  "regards") in a reply over 120 characters; a reply that merely asks about a subject line
  could match. Both models happened to write genuine drafts.
- **`hallucinated_arg` is evaluated on exercise 13 only**, and only for the `date` argument of
  `get_stock_price`. It is not a suite-wide measure.
- An earlier revision used a turn cap of 6, which truncated three Qwen3.8-27B runs mid-chain
  while still scoring them as passes. The cap is now 10 and `chain_complete` returns false for
  any run that exhausts it. All results here are post-fix.
