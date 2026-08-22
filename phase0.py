#!/usr/bin/env python3
"""Phase 0 - harness sanity gate. Proves a parse failure is the model's fault, not ours."""
import json, time, sys, urllib.request

URL = "http://localhost:8080/v1/chat/completions"
MODEL_LABEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b-UD-Q4_K_XL"
REPS = 10

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time in a given timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Tokyo"}
            },
            "required": ["timezone"],
        },
    },
}]
PROMPT = "What time is it in Tokyo? Use the tool."


def post(payload, timeout=300):
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def check_structure(msg):
    """Return (ok, reason, call)."""
    tcs = msg.get("tool_calls")
    if not tcs:
        return False, "no tool_calls emitted", None
    if len(tcs) != 1:
        return False, f"expected 1 tool_call, got {len(tcs)}", None
    tc = tcs[0]
    fn = tc.get("function", {})
    if fn.get("name") != "get_time":
        return False, f"wrong tool name: {fn.get('name')!r}", tc
    raw = fn.get("arguments")
    if not isinstance(raw, str):
        return False, f"arguments not a string: {type(raw).__name__}", tc
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"arguments not valid JSON: {e}", tc
    if "timezone" not in args:
        return False, f"missing required arg 'timezone': {args}", tc
    return True, f"timezone={args['timezone']!r}", tc


rows, passes = [], 0
print(f"Phase 0 — {MODEL_LABEL} — {REPS} reps, temp 0\n")
print(f"{'rep':>3}  {'parse':>5}  {'finish':>10}  {'think':>6}  {'gen':>5}  {'tok/s':>7}  detail")
print("-" * 78)

for i in range(1, REPS + 1):
    payload = {"messages": [{"role": "user", "content": PROMPT}],
               "tools": TOOLS, "tool_choice": "auto",
               "temperature": 0, "max_tokens": 512}
    try:
        d, wall = post(payload)
    except Exception as e:
        rows.append({"rep": i, "parse_ok": False, "error": str(e)})
        print(f"{i:>3}  {'FAIL':>5}  {'-':>10}  {'-':>6}  {'-':>5}  {'-':>7}  request error: {e}")
        continue

    ch = d["choices"][0]
    msg = ch["message"]
    ok, detail, tc = check_structure(msg)
    fin = ch.get("finish_reason")
    tim = d.get("timings", {})
    reasoning = msg.get("reasoning_content") or ""
    think_tok = len(reasoning.split())
    gen = tim.get("predicted_n", 0)
    tps = tim.get("predicted_per_second", 0.0)

    # a truncated generation is a harness/config problem, not a model failure
    if fin == "length":
        ok, detail = False, "TRUNCATED (finish_reason=length) — reasoning-budget or max_tokens too low"

    passes += ok
    print(f"{i:>3}  {'ok' if ok else 'FAIL':>5}  {fin:>10}  {think_tok:>6}  {gen:>5}  {tps:>7.1f}  {detail}")
    rows.append({"rep": i, "parse_ok": ok, "detail": detail, "finish_reason": fin,
                 "reasoning_words": think_tok, "gen_tokens": gen, "gen_tok_per_s": tps,
                 "wall_clock_s": round(wall, 3),
                 "tool_call": tc, "raw_content": msg.get("content"),
                 "reasoning_content": reasoning})

print("-" * 78)
print(f"structural pass rate: {passes}/{REPS}")

# determinism check (guide: any temp-0 disagreement across reps => flag)
# compare the function payload only - llama.cpp assigns a random id per tool_call,
# so including it would flag every run as nondeterministic
sigs = {json.dumps((r["tool_call"] or {}).get("function"), sort_keys=True)
        for r in rows if r.get("parse_ok")}
print(f"distinct tool_calls across passing reps: {len(sigs)}"
      + ("  [OK - deterministic]" if len(sigs) <= 1 else "  [!] NONDETERMINISM at temp 0"))

# round-trip: feed a tool result back, confirm the final answer uses it
rt = {"ok": False}
if passes:
    good = next(r for r in rows if r["parse_ok"])
    tc = good["tool_call"]
    tool_result = json.dumps({"timezone": "Asia/Tokyo", "time": "23:47", "date": "2026-08-22"})
    payload = {"messages": [
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "tool_calls": [tc], "content": None},
        {"role": "tool", "tool_call_id": tc.get("id", "call_0"),
         "name": "get_time", "content": tool_result}],
        "tools": TOOLS, "temperature": 0, "max_tokens": 512}
    try:
        d2, _ = post(payload)
        final = (d2["choices"][0]["message"].get("content") or "").strip()
        rt = {"ok": "23:47" in final, "final": final,
              "finish_reason": d2["choices"][0].get("finish_reason")}
    except Exception as e:
        rt = {"ok": False, "error": str(e)}

print("\nround-trip (tool result -> final answer):")
print(f"  references returned value: {rt['ok']}")
if rt.get("final"):
    print(f"  final answer: {rt['final'][:200]}")
if rt.get("error"):
    print(f"  error: {rt['error']}")

out = f"/home/shubham/test/bench/phase0_{MODEL_LABEL}.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.write(json.dumps({"summary": {"pass_rate": f"{passes}/{REPS}",
                                    "distinct_tool_calls": len(sigs),
                                    "round_trip": rt}}) + "\n")
print(f"\nwrote {out}")

gate = passes == REPS and rt["ok"] and len(sigs) <= 1
print("\nPHASE 0 GATE: " + ("PASS" if gate else "FAIL — halt, do not score through this"))
sys.exit(0 if gate else 1)
