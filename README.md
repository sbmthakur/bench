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
| `runs_*.jsonl` | Per-run calls + timings (39 runs for Qwen3.8-27B) |
| `phase0_*.jsonl` | Gate record |
| `report_qwen3.8-27b-UD-Q4_K_XL.md` | Results, config findings, and reproduction steps |
| `tool-calling-benchmark.md` | The spec: 13 verbatim prompts + ground-truth values |

## Requirements

- Python 3 (standard library only)
- `rocm-smi` on `PATH` (for VRAM sampling)
- A `llama-server` instance serving the target model at `http://localhost:8080`

## Run

1. Start the server (full flag rationale in `report_qwen3.8-27b-UD-Q4_K_XL.md` → Reproducing):

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

Only **Qwen3.8-27B-UD-Q4_K_XL** has a scored suite (the planned 2×2 dense-vs-MoE was not
completed). Strict tool-call accuracy **92.3%** (36/39), **4.97** correct tasks/min,
**0.0%** overcall rate. See `report_qwen3.8-27b-UD-Q4_K_XL.md` for the full breakdown and the one arguable failure.
