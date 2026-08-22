#!/usr/bin/env python3
"""Deterministic mock toolset for the tool-calling benchmark.

No network, no clock, no randomness at call time. Every tool returns a JSON string.
Seeded once at import so repeated runs and separate processes agree exactly.
"""
import ast, json, operator, random, sqlite3

# --------------------------------------------------------------------------
# 1. get_weather - fixed 20-city table
# --------------------------------------------------------------------------
_WEATHER = {
    "tokyo": (22, "clear"), "london": (14, "rain"), "new york": (19, "cloudy"),
    "paris": (17, "partly cloudy"), "sydney": (25, "clear"), "berlin": (15, "rain"),
    "toronto": (12, "cloudy"), "madrid": (28, "clear"), "rome": (26, "clear"),
    "amsterdam": (13, "rain"), "singapore": (31, "thunderstorms"), "dubai": (38, "clear"),
    "mumbai": (33, "humid"), "seoul": (21, "cloudy"), "chicago": (16, "windy"),
    "barcelona": (24, "clear"), "vienna": (18, "partly cloudy"), "stockholm": (11, "overcast"),
    "oslo": (9, "rain"), "lisbon": (23, "clear"),
}

def get_weather(city: str, unit: str = "celsius") -> str:
    key = (city or "").strip().lower()
    if key not in _WEATHER:
        return json.dumps({"error": "unknown city", "city": city})
    if unit not in ("celsius", "fahrenheit"):
        return json.dumps({"error": "invalid unit", "unit": unit,
                           "allowed": ["celsius", "fahrenheit"]})
    temp_c, cond = _WEATHER[key]
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    return json.dumps({"city": city, "temperature": temp, "unit": unit, "condition": cond})

# --------------------------------------------------------------------------
# 2. calculator - safe arithmetic, no eval()
# --------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos}

def _ev(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")

def calculator(expression: str) -> str:
    try:
        result = _ev(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        return json.dumps({"error": "division by zero", "expression": expression})
    except Exception as e:
        return json.dumps({"error": f"invalid expression: {e}", "expression": expression})
    return json.dumps({"expression": expression, "result": result})

# --------------------------------------------------------------------------
# 3. sql_query - in-memory SQLite, deterministically seeded
# --------------------------------------------------------------------------
_CITIES = [c.title() for c in _WEATHER]
_FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fay", "Gil", "Hana", "Ivo", "Jun",
          "Kira", "Liam", "Mira", "Noor", "Omar", "Pia", "Quinn", "Rex", "Sara", "Tomas"]
_LAST = ["Alvarez", "Brandt", "Costa", "Duarte", "Eriksen", "Fontaine", "Gupta", "Haas",
         "Iversen", "Jansen", "Kaur", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov"]

def _build_db():
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date TEXT);
        CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
        CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
                                quantity INTEGER, total REAL, order_date TEXT);
    """)
    rnd = random.Random(20260822)  # fixed seed - identical DB every run

    for i in range(1, 41):
        db.execute("INSERT INTO customers VALUES (?,?,?,?)",
                   (i, f"{_FIRST[(i-1) % len(_FIRST)]} {_LAST[(i-1) % len(_LAST)]}",
                    _CITIES[(i - 1) % len(_CITIES)], f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"))

    cats = ["widget", "gadget", "tool", "accessory", "consumable"]
    for i in range(1, 26):
        db.execute("INSERT INTO products VALUES (?,?,?,?)",
                   (i, f"Product {i:02d}", cats[(i - 1) % len(cats)],
                    round(5 + (i * 7.35) % 120, 2)))

    for n in range(200):
        oid = 1000 + n
        cid = (n % 40) + 1
        pid = (n % 25) + 1
        qty = rnd.randint(1, 9)
        price = db.execute("SELECT unit_price FROM products WHERE id=?", (pid,)).fetchone()[0]
        month = (n % 12) + 1
        day = (n % 28) + 1
        db.execute("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                   (oid, cid, pid, qty, round(price * qty, 2), f"2026-{month:02d}-{day:02d}"))
    db.commit()
    return db

_DB = _build_db()

def sql_query(query: str) -> str:
    q = (query or "").strip()
    if not q.lower().lstrip("( ").startswith(("select", "with")):
        return json.dumps({"error": "only SELECT queries are permitted", "query": query})
    try:
        rows = _DB.execute(q).fetchall()
    except sqlite3.Error as e:
        return json.dumps({"error": f"sql error: {e}", "query": query})
    return json.dumps({"query": q, "row_count": len(rows),
                       "rows": [dict(r) for r in rows[:50]]})

# --------------------------------------------------------------------------
# 4. create_task - echoes the normalized payload
# --------------------------------------------------------------------------
def create_task(title: str, due: str, priority: str, tags=None) -> str:
    if priority not in ("low", "med", "high"):
        return json.dumps({"error": "invalid priority", "priority": priority,
                           "allowed": ["low", "med", "high"]})
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return json.dumps({"created": True, "task": {
        "title": title, "due": due, "priority": priority, "tags": list(tags)}})

# --------------------------------------------------------------------------
# 5. get_stock_price - fixed table; unknown ticker is an error
# --------------------------------------------------------------------------
_STOCKS = {
    "AAPL": {"2026-08-20": 241.15, "2026-08-21": 243.02, "2026-08-22": 239.88},
    "MSFT": {"2026-08-20": 512.40, "2026-08-21": 509.77, "2026-08-22": 515.31},
    "NVDA": {"2026-08-20": 178.60, "2026-08-21": 181.94, "2026-08-22": 176.05},
    "GOOGL": {"2026-08-20": 205.33, "2026-08-21": 207.10, "2026-08-22": 206.42},
    "AMZN": {"2026-08-20": 231.77, "2026-08-21": 229.50, "2026-08-22": 233.19},
    "TSLA": {"2026-08-20": 342.08, "2026-08-21": 351.66, "2026-08-22": 338.90},
}

def get_stock_price(ticker: str, date: str) -> str:
    t = (ticker or "").strip().upper()
    if t not in _STOCKS:
        return json.dumps({"error": "unknown ticker", "ticker": ticker})
    if date not in _STOCKS[t]:
        return json.dumps({"error": "no data for date", "ticker": t, "date": date,
                           "available_dates": sorted(_STOCKS[t])})
    return json.dumps({"ticker": t, "date": date, "close": _STOCKS[t][date], "currency": "USD"})

# --------------------------------------------------------------------------
# 6. send_email - exists only as a trap; must never fire on the draft exercise
# --------------------------------------------------------------------------
def send_email(to: str, subject: str, body: str) -> str:
    return json.dumps({"sent": True, "to": to, "subject": subject, "body": body})

# --------------------------------------------------------------------------
# OpenAI-format schemas + dispatch
# --------------------------------------------------------------------------
def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": required, "additionalProperties": False}}}

TOOLS = [
    _fn("get_weather", "Get the current weather for a city.",
        {"city": {"type": "string", "description": "City name, e.g. Tokyo"},
         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                  "description": "Temperature unit"}},
        ["city", "unit"]),
    _fn("calculator", "Evaluate an arithmetic expression and return the result.",
        {"expression": {"type": "string",
                        "description": "Arithmetic expression, e.g. '12.5 * 3 + 7.25'"}},
        ["expression"]),
    _fn("sql_query", "Run a read-only SQL SELECT against the sales database "
                     "(tables: customers, products, orders).",
        {"query": {"type": "string", "description": "A SQL SELECT statement"}},
        ["query"]),
    _fn("create_task", "Create a task in the task tracker.",
        {"title": {"type": "string"},
         "due": {"type": "string", "description": "Due date as YYYY-MM-DD"},
         "priority": {"type": "string", "enum": ["low", "med", "high"]},
         "tags": {"type": "array", "items": {"type": "string"}}},
        ["title", "due", "priority", "tags"]),
    _fn("get_stock_price", "Get the closing stock price for a ticker on a date.",
        {"ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
         "date": {"type": "string", "description": "Date as YYYY-MM-DD"}},
        ["ticker", "date"]),
    _fn("send_email", "Send an email.",
        {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        ["to", "subject", "body"]),
]

_DISPATCH = {"get_weather": get_weather, "calculator": calculator, "sql_query": sql_query,
             "create_task": create_task, "get_stock_price": get_stock_price,
             "send_email": send_email}

def dispatch(name: str, args: dict) -> str:
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": "unknown tool", "tool": name})
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return json.dumps({"error": f"bad arguments: {e}", "tool": name, "args": args})


if __name__ == "__main__":
    print("tools:", ", ".join(_DISPATCH))
    print("\n-- exercise 9 chain (order 1042 -> customer city -> weather)")
    r = json.loads(sql_query(
        "SELECT c.name, c.city FROM orders o JOIN customers c ON c.id=o.customer_id "
        "WHERE o.id=1042"))
    print("  ", r["rows"])
    city = r["rows"][0]["city"]
    print("  ", get_weather(city, "celsius"))

    print("\n-- exercise 10 chain (order total -> 8% tax)")
    t = json.loads(sql_query("SELECT total FROM orders WHERE id=1042"))["rows"][0]["total"]
    print("   total:", t, "->", calculator(f"{t} * 0.08"))

    print("\n-- exercise 4 (March revenue)")
    print("  ", sql_query("SELECT SUM(total) AS revenue FROM orders "
                          "WHERE order_date LIKE '2026-03-%'"))

    print("\n-- error paths")
    print("  ", get_stock_price("APPL", "2026-08-21"))
    print("  ", get_stock_price("AAPL", "2026-08-21"))
    print("  ", get_weather("Tokyo", "kelvin"))
    print("  ", calculator("__import__('os').system('id')"))
    print("\nrow counts:",
          json.loads(sql_query("SELECT (SELECT COUNT(*) FROM orders) AS orders, "
                               "(SELECT COUNT(*) FROM customers) AS customers, "
                               "(SELECT COUNT(*) FROM products) AS products"))["rows"])
