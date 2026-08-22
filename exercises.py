"""The 13 benchmark exercises: prompts (verbatim from tool-calling-benchmark.md) + scorers.

A scorer receives the flattened run record and returns a dict of metric fields.
Everything mechanical is scored automatically; anything requiring human judgement
sets needs_review=True rather than guessing.
"""
SYSTEM = "You are a helpful assistant with access to tools. Today's date is 2026-08-22."
TODAY = "2026-08-22"


def _names(calls):
    return [c["name"] for c in calls]


def _first(calls, name):
    for c in calls:
        if c["name"] == name:
            return c
    return None


# ---- T1 -------------------------------------------------------------------
def s1(r):
    c = _first(r["calls"], "get_weather")
    tool_ok = bool(c) and _names(r["calls"])[:1] == ["get_weather"]
    args_ok = bool(c) and str(c["args"].get("city", "")).strip().lower() == "tokyo" \
        and c["args"].get("unit") == "fahrenheit"
    return {"tool_correct": tool_ok, "args_correct": args_ok,
            "chain_complete": tool_ok and args_ok}


def s2(r):
    c = _first(r["calls"], "create_task")
    if not c:
        return {"tool_correct": False, "args_correct": False, "chain_complete": False}
    a = c["args"]
    due = str(a.get("due", "")).strip()
    tags = a.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    tags_ok = {str(t).strip().lower() for t in tags} == {"finance", "q3", "review"}
    # both readings of "next Friday" accepted - today is a Saturday, so it is ambiguous
    due_ok = due in ("2026-08-28", "2026-09-04")
    args_ok = due_ok and a.get("priority") == "high" and tags_ok
    return {"tool_correct": True, "args_correct": args_ok, "chain_complete": args_ok,
            "due_choice": due if due_ok else f"OTHER:{due}"}


def s3(r):
    c = _first(r["calls"], "calculator")
    if not c:
        return {"tool_correct": False, "args_correct": False, "chain_complete": False}
    expr = str(c["args"].get("expression", ""))
    operands_kept = all(tok in expr for tok in ("1284.75", "3.6", "91.4", "7"))
    # a pre-computed answer would carry no operators
    not_precomputed = any(op in expr for op in "*+/x×÷")
    ok = operands_kept and not_precomputed
    return {"tool_correct": True, "args_correct": ok, "chain_complete": ok,
            "expression": expr}


# ---- T2 -------------------------------------------------------------------
def s4(r):
    c = _first(r["calls"], "sql_query")
    if not c:
        return {"tool_correct": False, "args_correct": False, "chain_complete": False}
    sql_ran = any(res.get("name") == "sql_query" and "error" not in res.get("parsed", {})
                  for res in r["results"])
    got_truth = "5413.65" in r["final_text"] or any(
        "5413.65" in str(res.get("content", "")) for res in r["results"])
    return {"tool_correct": True, "args_correct": sql_ran,
            "chain_complete": sql_ran and got_truth}


def s5(r):
    names = _names(r["calls"])
    ok = "calculator" in names and "sql_query" not in names
    return {"tool_correct": ok, "args_correct": ok, "chain_complete": ok,
            "wrong_tool": "sql_query" if "sql_query" in names else None}


def _no_tool(r):
    n = len(r["calls"])
    return {"tool_correct": n == 0, "args_correct": n == 0, "chain_complete": n == 0,
            "overcall": n > 0, "called": _names(r["calls"])}


s6 = s7 = _no_tool


def s8(r):
    names = _names(r["calls"])
    sent = "send_email" in names
    text = r["final_text"]
    looks_drafted = len(text) > 120 and any(
        k in text.lower() for k in ("subject", "dear", "hi ", "hello", "regards", "sincerely"))
    return {"tool_correct": not sent, "args_correct": not sent,
            "chain_complete": (not sent) and looks_drafted,
            "overcall": sent, "trap_fired": sent,
            "needs_review": (not sent) and not looks_drafted}


# ---- T3 -------------------------------------------------------------------
def s9(r):
    names = _names(r["calls"])
    order_ok = "sql_query" in names and "get_weather" in names \
        and names.index("sql_query") < names.index("get_weather")
    w = _first(r["calls"], "get_weather")
    threaded = bool(w) and str(w["args"].get("city", "")).strip().lower() == "new york"
    return {"tool_correct": order_ok, "args_correct": threaded,
            "chain_complete": order_ok and threaded}


def s10(r):
    names = _names(r["calls"])
    order_ok = "sql_query" in names and "calculator" in names \
        and names.index("sql_query") < names.index("calculator")
    c = _first(r["calls"], "calculator")
    expr = str(c["args"].get("expression", "")) if c else ""
    threaded = "86.5" in expr and ("0.08" in expr or "8" in expr)
    got_truth = "6.92" in r["final_text"] or any(
        "6.92" in str(res.get("content", "")) for res in r["results"])
    return {"tool_correct": order_ok, "args_correct": threaded,
            "chain_complete": order_ok and threaded and got_truth}


def s11(r):
    want = {"tokyo", "london", "singapore"}
    got = {str(c["args"].get("city", "")).strip().lower()
           for c in r["calls"] if c["name"] == "get_weather"}
    ok = want <= got
    # parallel = all three issued from a single assistant turn
    per_turn = r["calls_per_turn"]
    parallel = any(len([c for c in t if c["name"] == "get_weather"]) >= 3 for t in per_turn)
    return {"tool_correct": ok, "args_correct": ok, "chain_complete": ok,
            "issued_parallel": parallel, "weather_turns":
                sum(1 for t in per_turn if any(c["name"] == "get_weather" for c in t))}


# ---- T4 -------------------------------------------------------------------
def s12(r):
    sc = [c for c in r["calls"] if c["name"] == "get_stock_price"]
    tickers = [str(c["args"].get("ticker", "")).upper() for c in sc]
    recovered = "AAPL" in tickers
    # "within 2 extra turns" - the first call is turn 1, so the fix must land by turn 3
    idx = tickers.index("AAPL") if recovered else 99
    in_budget = recovered and r["turns"] <= 4
    got_truth = "243.02" in r["final_text"]
    return {"tool_correct": bool(sc), "args_correct": recovered,
            "chain_complete": recovered and in_budget and got_truth,
            "recovery_turns": r["turns"], "tickers_tried": tickers}


def s13(r):
    sc = [c for c in r["calls"] if c["name"] == "get_stock_price"]
    dates = [str(c["args"].get("date", "")) for c in sc]
    invented = [d for d in dates if d]
    asked = len(sc) == 0 and "?" in r["final_text"]
    # using the date supplied in the system prompt is a defensible inference,
    # not the same failure as fabricating an arbitrary one - track it separately
    used_today = [d for d in invented if d == TODAY]
    fabricated = [d for d in invented if d != TODAY]
    return {"tool_correct": asked, "args_correct": asked, "chain_complete": asked,
            "hallucinated_arg": bool(fabricated),
            "used_today_date": bool(used_today),
            "dates_supplied": invented,
            "needs_review": (not asked) and not invented}


EXERCISES = [
    dict(id=1,  tier="T1", prompt="What's the weather in Tokyo? Use fahrenheit.", score=s1),
    dict(id=2,  tier="T1", prompt="Create a task to review the Q3 budget, due next Friday, "
                                  "high priority, tagged finance, q3 and review.", score=s2),
    dict(id=3,  tier="T1", prompt="What's 1284.75 × 3.6 + 91.4 divided by 7?", score=s3),
    dict(id=4,  tier="T2", prompt="What was the total revenue from orders in March?", score=s4),
    dict(id=5,  tier="T2", prompt="If our March revenue was 5413.65 and we take a 12% "
                                  "platform cut, what do we keep?", score=s5),
    dict(id=6,  tier="T2", prompt="What's the capital of Australia?", score=s6),
    dict(id=7,  tier="T2", prompt="Why does iron rust?", score=s7),
    dict(id=8,  tier="T2", prompt="Draft an email to the vendor about the delayed shipment.",
         score=s8),
    dict(id=9,  tier="T3", prompt="Find the city of the customer who placed order #1042, "
                                  "then get me its weather.", score=s9),
    dict(id=10, tier="T3", prompt="Get the total for order #1042 and calculate 8% tax on it.",
         score=s10),
    dict(id=11, tier="T3", prompt="What's the weather in Tokyo, London, and Singapore?",
         score=s11),
    dict(id=12, tier="T4", prompt="What did APPL close at on 2026-08-21?", score=s12),
    dict(id=13, tier="T4", prompt="What's Tesla trading at?", score=s13),
]
