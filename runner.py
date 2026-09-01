#!/usr/bin/env python3
"""Tool-calling benchmark runner.

Executes 13 exercises x N reps as real agentic conversations against an
OpenAI-compatible endpoint, dispatching tool calls through mock_tools.

  python3 runner.py --label qwen3.8-27b-UD-Q4_K_XL
"""
import argparse, json, random, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mock_tools
from exercises import EXERCISES, SYSTEM

MAX_TURNS = 10


def vram_gib():
    try:
        o = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                           capture_output=True, text=True, timeout=15).stdout
        for l in o.splitlines():
            if "GPU[0]" in l and "Used" in l:
                return round(int(l.split(":")[-1].strip()) / 2**30, 2)
    except Exception:
        pass
    return None


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_one(url, ex, cache_prompt=True):
    """Drive one exercise to completion. Returns a flat record for scoring."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ex["prompt"]}]
    calls, results, calls_per_turn = [], [], []
    prompt_tok = gen_tok = reasoning_tok = 0
    turns = 0
    final_text = ""
    harness_error = None
    t0 = time.time()

    while turns < MAX_TURNS:
        payload = {"messages": messages, "tools": mock_tools.TOOLS,
                   "temperature": 0, "max_tokens": 1024, "cache_prompt": cache_prompt}
        try:
            d = post(url, payload)
        except urllib.error.HTTPError as e:
            harness_error = f"HTTP {e.code}: {e.read().decode()[:300]}"
            break
        except Exception as e:
            harness_error = f"{type(e).__name__}: {e}"
            break

        turns += 1
        ch = d["choices"][0]
        msg = ch["message"]
        t = d.get("timings", {})
        prompt_tok += t.get("prompt_n", 0)
        gen_tok += t.get("predicted_n", 0)
        reasoning_tok += len((msg.get("reasoning_content") or "").split())

        if ch.get("finish_reason") == "length":
            harness_error = "generation truncated (finish_reason=length)"

        tcs = msg.get("tool_calls") or []
        this_turn = []
        if not tcs:
            final_text = (msg.get("content") or "").strip()
            break

        # assistant turn carrying tool calls
        messages.append({"role": "assistant", "content": msg.get("content"),
                         "tool_calls": tcs})
        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name")
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"__unparsable__": raw}
            rec = {"name": name, "args": args}
            calls.append(rec); this_turn.append(rec)

            out = mock_tools.dispatch(name, args if "__unparsable__" not in args else {})
            try:
                parsed = json.loads(out)
            except json.JSONDecodeError:
                parsed = {}
            results.append({"name": name, "content": out, "parsed": parsed})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", f"call_{len(calls)}"),
                             "name": name, "content": out})
        calls_per_turn.append(this_turn)

    wall = time.time() - t0
    # the loop exits either on a tool-call-free reply (final answer), on error, or by
    # exhausting MAX_TURNS. the third case is a harness limit, not a model result.
    cap_exhausted = (turns >= MAX_TURNS and not final_text and harness_error is None)
    # parse_ok: every emitted tool call had valid JSON arguments
    parse_ok = harness_error is None and all(
        "__unparsable__" not in c["args"] for c in calls)
    return {"turns": turns, "cap_exhausted": cap_exhausted,
            "calls": calls, "results": results,
            "calls_per_turn": calls_per_turn, "final_text": final_text,
            "parse_ok": parse_ok, "harness_error": harness_error,
            "prompt_tokens": prompt_tok, "gen_tokens": gen_tok,
            "reasoning_words": reasoning_tok, "wall_clock_s": round(wall, 3),
            "gen_tok_per_s": round(gen_tok / wall, 2) if wall > 0 else 0,
            "messages": messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="model label for the output file")
    ap.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--only", type=int, nargs="*", help="run only these exercise ids")
    ap.add_argument("--no-cache", action="store_true", help="send cache_prompt=false")
    args = ap.parse_args()

    exercises = [e for e in EXERCISES if not args.only or e["id"] in args.only]
    plan = [(e, rep) for e in exercises for rep in range(1, args.reps + 1)]
    random.Random(args.seed).shuffle(plan)

    props = json.load(urllib.request.urlopen(args.url.replace("/v1/chat/completions", "/props")))
    header = {"label": args.label,
              "model_alias": props.get("model_alias"),
              "n_ctx": props["default_generation_settings"]["n_ctx"],
              "total_slots": props.get("total_slots"),
              "build": props.get("build_info"),
              "vram_gib_idle": vram_gib(),
              "reps": args.reps, "seed": args.seed, "temperature": 0,
              "cache_prompt": not args.no_cache, "max_turns": MAX_TURNS,
              "exercise_count": len(exercises), "run_count": len(plan)}
    print(json.dumps(header, indent=2))

    # warm-up: pay the cold prefill once so no scored run carries it
    print("\nwarm-up (discarded)...", end=" ", flush=True)
    w = run_one(args.url, EXERCISES[0], cache_prompt=not args.no_cache)
    print(f"{w['wall_clock_s']}s\n")

    out_path = Path(__file__).parent / f"runs_{args.label}.jsonl"
    # stream: header first, then one row per completed run, so a crash keeps partials
    # and `wc -l` works as a live progress counter
    stream = open(out_path, "w", buffering=1)
    stream.write(json.dumps({"header": header}) + "\n")
    rows, peak = [], header["vram_gib_idle"] or 0
    print(f"{'#':>3} {'ex':>3} {'tier':>4} {'rep':>3} {'turns':>5} {'strict':>6} "
          f"{'wall_s':>7} {'tok/s':>6}  notes", flush=True)
    print("-" * 84)

    for i, (ex, rep) in enumerate(plan, 1):
        r = run_one(args.url, ex, cache_prompt=not args.no_cache)
        if r["harness_error"] is None:
            try:
                m = ex["score"](r)
            except Exception as e:
                # a scorer bug must degrade one run, never kill the suite
                m = {"scorer_error": f"{type(e).__name__}: {e}", "needs_review": True}
        else:
            m = {}
        if r["cap_exhausted"]:
            m["chain_complete"] = False
            m["needs_review"] = True
        strict = bool(r["parse_ok"] and m.get("tool_correct") and m.get("args_correct"))
        v = vram_gib()
        peak = max(peak, v or 0)
        row = {"n": i, "exercise": ex["id"], "tier": ex["tier"], "rep": rep,
               "prompt": ex["prompt"], "strict_correct": strict,
               **{k: r[k] for k in ("parse_ok", "harness_error", "turns", "cap_exhausted",
                                    "prompt_tokens",
                                    "gen_tokens", "reasoning_words", "wall_clock_s",
                                    "gen_tok_per_s")},
               **m,
               "calls": r["calls"], "final_text": r["final_text"], "vram_gib": v}
        rows.append(row)
        stream.write(json.dumps(row) + "\n")

        notes = []
        if r["harness_error"]:
            notes.append(f"HARNESS: {r['harness_error'][:40]}")
        if m.get("overcall"):
            notes.append("overcall")
        if m.get("hallucinated_arg"):
            notes.append("hallucinated_arg")
        if m.get("used_today_date"):
            notes.append("used_today_date")
        if r["cap_exhausted"]:
            notes.append("CAP_EXHAUSTED")
        if m.get("needs_review"):
            notes.append("NEEDS_REVIEW")
        if m.get("issued_parallel"):
            notes.append("parallel")
        print(f"{i:>3} {ex['id']:>3} {ex['tier']:>4} {rep:>3} {r['turns']:>5} "
              f"{'ok' if strict else 'FAIL':>6} {r['wall_clock_s']:>7.2f} "
              f"{r['gen_tok_per_s']:>6.1f}  {', '.join(notes)}", flush=True)

    stream.close()
    header["vram_gib_peak"] = peak
    header["wall_clock_total_s"] = round(sum(r["wall_clock_s"] for r in rows), 1)
    # rewrite with the finalized header (peak VRAM / total wall clock are only known now)
    with open(out_path, "w") as f:
        f.write(json.dumps({"header": header}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ok = sum(r["strict_correct"] for r in rows)
    mins = header["wall_clock_total_s"] / 60
    print("-" * 84)
    print(f"strict accuracy : {ok}/{len(rows)} = {100*ok/len(rows):.1f}%")
    print(f"wall clock      : {mins:.1f} min")
    print(f"correct tasks/min: {ok/mins:.2f}")
    print(f"VRAM peak       : {peak} GiB")
    print(f"needs review    : {sum(1 for r in rows if r.get('needs_review'))}")
    print(f"harness errors  : {sum(1 for r in rows if r.get('harness_error'))}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
