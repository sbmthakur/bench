# Tool-Calling Benchmark

Instructions for harness. Build and run a tight tool-calling eval against three models served
through Hermes Agent on an RX 7900 XT (20GB, RDNA3, ROCm, llama.cpp-hip):

1. **Muse Glimmer 30B** (dense)
2. **Nemotron 3.5 Lightning 30B-A3B** (MoE)
3. **Qwen3.6-35B-A3B** (MoE, incumbent anchor — already installed and benchmarked on this box;
   reuse the known-good quant and MoE offload config from prior runs, don't re-derive them)
4. **Qwen3.8-27B** (dense, released Aug 14 — successor to the Qwen3.6-27B daily driver on this box)

This is a clean 2×2: two dense vs two MoE, all ~17GB-class, all released or refreshed within
the same month. Results go into a LinkedIn post: small exercise count, high rigor per exercise,
3 headline numbers.

**Qwen3.8-27B specifics:**
- Its default reasoning effort is `xhigh` and is known to dramatically over-think simple
  prompts. For this benchmark, set reasoning effort per the model card's recommendation for
  agentic/tool use (likely `medium` or `low`); log the chosen setting in the report header.
  Do NOT leave it at default — that would unfairly tank its tasks/min.
- It supports MTP speculative decoding in llama.cpp. Keep MTP OFF in v1: no other model gets
  a draft accelerator, and the throughput comparison must be like-for-like. Note it as
  upside-not-measured in the report.

**Framing:** same-VRAM-footprint comparison. Headline metric = correct-tool-calls-per-minute of
wall clock, not accuracy alone. The dense-vs-MoE tradeoff curve is the finding. The Qwen anchor
also lets us independently check NVIDIA's claim of ~35% faster task completion than Qwen3.6 at
similar accuracy — note in the report whether our numbers directionally agree.

---

## Phase -1 — Quant selection (harness decides)

For Glimmer, Lightning, and Qwen3.8-27B, pick the quant yourself (Qwen3.6-35B-A3B uses its
existing config). Procedure:

1. List available GGUFs (official + Unsloth dynamic) for each of the three on Hugging Face.
2. **Bias toward accuracy: pick the highest-bpw quant that fits.** Preference order
   Q8 > Q6_K > Q5_K_M > Q4 dynamic > plain Q4. Never spill to system RAM or CPU layers —
   this is a GPU throughput benchmark.
3. Budget: weights + KV/state + compute buffer must fit in 20GB with ~1GB headroom.
   The suite needs little context (longest exercise ≈ 3–4K tokens with tool defs and
   multi-turn history), so **trade context ceiling down to afford a higher quant**: budget
   for an 8K cap by default, and accept 6K if that's what unlocks the next quant tier.
   Note Lightning's Mamba-2 layers carry recurrent state, not KV — its context cost is
   near-flat, so it can likely afford a higher quant than the dense model; that's fine.
4. Verify empirically: load each model, run a prompt at the chosen ctx cap, confirm no OOM,
   log actual VRAM high-water mark (`rocm-smi`).
5. Record chosen quant, bpw, file size, ctx cap, VRAM peak, and rationale in the report header.
   The models will likely land at different bpw (the ~27–28B dense pair can afford more than a
   30B dense would; MoEs differ again) — say so plainly in the report; it's part of the
   same-footprint story, not a confound to hide.

If Lightning fails to load or throws Mamba-2 kernel errors on ROCm: stop, report versions and
errors. That is itself a publishable finding.

## Phase 0 — Harness sanity (mandatory)

Prove a parse failure means the model failed, not the harness.

1. One trivial tool: `get_time(timezone: str)`. Prompt: "What time is it in Tokyo? Use the tool."
2. 10 reps per model (all four). Require 100% structurally-valid `tool_calls` output.
   - Glimmer emits XML-style ATEM tool calls, not JSON — confirm Hermes/llama.cpp translates
     them to standard tool_calls. If not, halt and diagnose (template/parser flags) before
     proceeding. Do not score anything through a broken parser.
3. Verify round-trip: feed a `tool` role result back, confirm a final answer referencing it.

---

## Mock toolset

`mock_tools.py`, deterministic, no network. All return JSON.

1. `get_weather(city: str, unit: enum["celsius","fahrenheit"])` — fixed 20-city table
2. `calculator(expression: str)` — safe arithmetic eval
3. `sql_query(query: str)` — in-memory SQLite, 3 seeded tables (orders, customers, products, ~200 rows)
4. `create_task(title: str, due: str, priority: enum["low","med","high"], tags: list[str])` — echoes payload
5. `get_stock_price(ticker: str, date: str)` — fixed table; unknown ticker → `{"error": "unknown ticker"}`
6. `send_email(to: str, subject: str, body: str)` — echoes payload; exists only as a trap

---

## Exercises (13 total, 3 reps each, temperature 0 only)

Every run uses this system prompt verbatim (the date matters for exercise 2):

> You are a helpful assistant with access to tools. Today's date is 2026-08-22.

All 6 mock tools are offered on every exercise, including the no-tool traps — otherwise the
traps measure nothing.

### T1 — Schema fidelity (3)

**1. Weather with enum unit**
Prompt: *"What's the weather in Tokyo? Use fahrenheit."*
Pass: `get_weather`, `city="Tokyo"`, `unit="fahrenheit"` — a valid enum member, not `"F"`,
`"Fahrenheit"`, or `"imperial"`.

**2. `create_task` with relative date**
Prompt: *"Create a task to review the Q3 budget, due next Friday, high priority, tagged
finance, q3 and review."*
Pass: exact field match after normalization — `due` as YYYY-MM-DD, `priority="high"`,
`tags` a 3-element array.
**Accept both 2026-08-28 and 2026-09-04.** Today is a Saturday, so "next Friday" is genuinely
ambiguous and an exact-match criterion would fail a correct model on a coin flip. Log which
reading each model picked; the split is itself a result.

**3. Float-heavy calculator**
Prompt: *"What's 1284.75 × 3.6 + 91.4 divided by 7?"*
Pass: `calculator` with the operands 1284.75, 3.6, 91.4 and 7 all preserved in the expression
argument — not pre-computed to a constant, not rounded.

### T2 — Tool selection (5) — all 6 tools offered

**4. SQL aggregate**
Prompt: *"What was the total revenue from orders in March?"*
Pass: `sql_query` with valid SQL. Ground truth: **5413.65**.

**5. Near-miss (financial phrasing, arithmetic task)**
Prompt: *"If our March revenue was 5413.65 and we take a 12% platform cut, what do we keep?"*
Pass: `calculator`, **not** `sql_query`. The embedded figure is the bait — it looks like it
came from the database.

**6. No-tool trap A**
Prompt: *"What's the capital of Australia?"*
Pass: direct answer, zero tool calls. Scores `overcall_rate`.

**7. No-tool trap B**
Prompt: *"Why does iron rust?"*
Pass: direct answer, zero tool calls. Scores `overcall_rate`.

**8. Draft trap**
Prompt: *"Draft an email to the vendor about the delayed shipment."*
Pass: writes the draft in its reply; does **not** call `send_email`.

### T3 — Multi-step (3)

**9. 2-hop, result-threading**
Prompt: *"Find the city of the customer who placed order #1042, then get me its weather."*
Pass: `sql_query` → `get_weather`, where the second call uses the **actual** first result.
Ground truth: customer Chen Costa, city **New York**, 19°C cloudy.

**10. 2-hop with math**
Prompt: *"Get the total for order #1042 and calculate 8% tax on it."*
Pass: `sql_query` → `calculator`. Ground truth: total 86.50 → tax **6.92**.

**11. Parallel**
Prompt: *"What's the weather in Tokyo, London, and Singapore?"*
Pass: all 3 correct. Log whether issued in one turn (parallel) or sequentially — turn count is
reported, not a pass condition.

### T4 — Error recovery (2)

**12. Misspelled ticker, recoverable**
Prompt: *"What did APPL close at on 2026-08-21?"*
The tool returns `{"error": "unknown ticker"}`; the correct symbol is inferable from context.
Pass: retry with `ticker="AAPL"` within 2 extra turns. Ground truth: **243.02**.

**13. Missing required argument**
Prompt: *"What's Tesla trading at?"*
`get_stock_price` requires a date the user never supplied. Pass: the model **asks** for it.
Any call carrying an invented date scores `hallucinated_arg_rate`.

Total: 13 exercises × 3 reps × 4 models = 156 runs. Roughly 1.5–2 hours; the two dense models
dominate wall clock, the MoEs are cheap.

---

## Metrics & report

Per-run JSONL: `parse_ok`, `tool_correct`, `args_correct`, `chain_complete`, `turns`,
`overcall`, `hallucinated_arg`, token counts, `wall_clock_s`, `gen_tok_per_s`, `raw_response`.

Aggregate per-model report (`report_<label>.md`):

1. **One table:** per-tier strict accuracy (parse ∧ tool ∧ args), overcall rate,
   hallucinated-arg rate, median gen tok/s, per model.
2. **One chart:** strict accuracy (y) vs correct-tasks-per-minute (x), four labeled points —
   encode dense vs MoE by marker shape or color so the 2×2 reads at a glance. This is the
   LinkedIn image — render it clean at 1200×675, dark background, no chartjunk.
3. **Three headline numbers**, stated first, no warm-up prose:
   - strict tool-call accuracy per model
   - correct tasks/min per model
   - overcall rate per model
   Plus one line on the NVIDIA-vs-Qwen speed claim: confirmed, partial, or not reproduced.
4. Header: quants chosen + VRAM peaks, llama.cpp/Hermes/ROCm versions, ctx cap, temp 0, 3 reps.
5. A `linkedin_draft.md`: ≤120 words, leads with the numbers, notes the same-footprint framing
   and quant choices in one line each, no hashtag spam, no em-dash chains.

## Run protocol

1. Phase -1, then Phase 0. Halt on failure at either.
2. Run each model fully before loading the next (no reload thrash on 20GB). Seeded exercise
   order. Suggested sequence: Qwen3.6-35B-A3B first (known-good, validates the harness
   end-to-end), then Qwen3.8-27B, then Lightning, then Glimmer.
3. Any temp-0 disagreement across reps → flag nondeterminism, note batch-effect suspicion.

## Out of scope

Vision (Glimmer and Qwen3.8 both have it; note as one-line asymmetry vs the MoEs). Long-context
tiers, speculative decoding (MTP/D-Flash — Qwen3.8's MTP boost is real but stays off in v1),
BFCL — all follow-up material if v1 lands well.
