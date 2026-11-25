"""Lab 3, Steps 2 and 4: MCP server for the pharmacy stock.

READ tools  (percepts / sensors):  check_stock, search_items, list_expiring
WRITE tool  (action / actuator):   record_sale  (validated)

Test with:  npx @modelcontextprotocol/inspector python server.py
"""
import csv
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pharmacy-stock")

# Resolve relative to this file so it works from any directory.
DATA = Path(__file__).parent / "data" / "pharmacy_stock.csv"
FIELDS = ["item", "strength", "form", "quantity", "unit_price_ghs", "expiry_date"]


def load() -> list[dict]:
    with open(DATA, newline="") as f:
        return list(csv.DictReader(f))


def save(rows: list[dict]) -> None:
    """Write to a temp file then swap it in, so a crash can't corrupt the CSV."""
    fd, tmp = tempfile.mkstemp(dir=DATA.parent, suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, DATA)


# ---------------------------------------------------------------- read tools
@mcp.tool()
def check_stock(item: str, strength: str = "") -> list[dict]:
    """Return stock rows whose item name contains `item` (and strength, if given).
    Use this for questions like 'Do we have amoxicillin 500mg?' or 'What does
    paracetamol cost?'. Each row has quantity, unit_price_ghs and expiry_date."""
    rows = [r for r in load() if item.lower() in r["item"].lower()]
    if strength:
        rows = [r for r in rows if strength.lower() in r["strength"].lower()]
    return rows


@mcp.tool()
def search_items(query: str) -> list[dict]:
    """Free-text search across item name, strength and form (e.g. 'tablet',
    'antibiotic name fragment'). Use when the exact drug name is unknown."""
    q = query.lower()
    return [r for r in load()
            if q in r["item"].lower() or q in r["strength"].lower() or q in r["form"].lower()]


@mcp.tool()
def list_expiring(days: int) -> list[dict]:
    """List items that expire within the next `days` days from today
    (already-expired items are included and marked)."""
    cutoff = date.today() + timedelta(days=days)
    out = []
    for r in load():
        try:
            exp = datetime.strptime(r["expiry_date"], "%Y-%m-%d").date()
        except ValueError:
            continue  # skip rows with a malformed date
        if exp <= cutoff:
            out.append({**r, "expired": exp < date.today()})
    return out


# --------------------------------------------------------------- write tool
@mcp.tool()
def record_sale(item: str, strength: str, qty: int) -> str:
    """Record a sale and reduce stock. ONLY call this when the user clearly asks
    to sell/record a sale. `item` and `strength` must identify exactly one row
    (e.g. item='Amoxicillin', strength='500mg'). `qty` is a positive number of
    packs. Returns a confirmation or a refusal message explaining why."""
    # Safety checks: never trust the model (or the prompt) to enforce these.
    if qty <= 0:
        return "REFUSED: quantity must be a positive whole number."

    rows = load()
    matches = [r for r in rows
               if r["item"].lower() == item.lower()
               and r["strength"].lower() == strength.lower()]
    if not matches:
        return f"REFUSED: no item '{item} {strength}' found. Check the name and strength."
    if len(matches) > 1:
        return "REFUSED: more than one row matches; be more specific."

    row = matches[0]
    available = int(row["quantity"])
    if qty > available:
        return f"REFUSED: only {available} pack(s) of {row['item']} {row['strength']} in stock; cannot sell {qty}."

    try:
        if datetime.strptime(row["expiry_date"], "%Y-%m-%d").date() < date.today():
            return f"REFUSED: {row['item']} {row['strength']} has expired ({row['expiry_date']})."
    except ValueError:
        pass

    row["quantity"] = str(available - qty)
    save(rows)
    total = qty * float(row["unit_price_ghs"])
    return (f"OK: sold {qty} x {row['item']} {row['strength']} for GHS {total:.2f}. "
            f"Remaining stock: {row['quantity']}.")


if __name__ == "__main__":
    mcp.run()