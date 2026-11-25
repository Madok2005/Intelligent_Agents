

import csv
import json
import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()
MODEL = os.getenv("LLM_MODEL")
DATA = "data/pharmacy_stock.csv"


def check_stock(item: str, strength: str = "") -> list[dict]:
    """Plain Python function: the 'sensor'."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.DictReader(f) if item.lower() in r["item"].lower()]
    if strength:
        rows = [r for r in rows if strength.lower() in r["strength"].lower()]
    return rows


# The model never runs this. It only asks for it; our code executes it.
TOOLS = {"check_stock": check_stock}

tool_schemas = [{
    "type": "function",
    "function": {
        "name": "check_stock",
        "description": (
            "Look up stock rows in the pharmacy inventory by drug name "
            "(partial match) and optionally strength, e.g. '500mg'. "
            "Returns quantity, unit price in GHS and expiry date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Drug name, e.g. 'amoxicillin'"},
                "strength": {"type": "string", "description": "Optional strength, e.g. '500mg'"},
            },
            "required": ["item"],
        },
    },
}]


def main():
    if not MODEL:
        raise SystemExit("LLM_MODEL is not set. Check your .env file.")

    messages = [
        {"role": "system", "content": "You are a pharmacy stock assistant. Use tools for stock facts."},
        {"role": "user", "content": "Do we have amoxicillin 500mg?"},
    ]

    resp = completion(model=MODEL, messages=messages, tools=tool_schemas, temperature=0)
    msg = resp.choices[0].message
    print("Model asked for:", msg.tool_calls)          # <- an ACTION decided by the model

    messages.append(msg.model_dump())
    for tc in msg.tool_calls or []:
        args = json.loads(tc.function.arguments)
        result = TOOLS[tc.function.name](**args)        # <- our code executes it
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(result)})  # <- a PERCEPT

    resp = completion(model=MODEL, messages=messages, tools=tool_schemas, temperature=0)
    print("Final answer:", resp.choices[0].message.content)


if __name__ == "__main__":
    main()