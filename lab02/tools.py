"""lab02/tools.py - the agent's actuators (Step 2).

The model never runs these. It only writes text naming a tool;
react_agent.py parses that text and calls the function.
"""
import ast
import operator as op
import re

import requests

# ---------- Calculator (safe: uses ast, never eval) ----------
OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}


def _eval(node):
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate plain arithmetic such as '4*185 + 3*95'."""
    try:
        expr = re.sub(r"(?i)ghs|₵|\$", "", expression).replace(",", "")
        return str(_eval(ast.parse(expr.strip(), mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


# ---------- Wikipedia lookup ----------
def wiki_lookup(title: str) -> str:
    """Return the first 800 characters of a Wikipedia page summary."""
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + title.strip().replace(" ", "_"))
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "DCIT403-lab"})
    except requests.RequestException as e:
        return f"Error: could not reach Wikipedia ({e})."
    if r.status_code != 200:
        return f"No Wikipedia page found for '{title}'."
    return r.json().get("extract", "")[:800]


# Registry used by the agent loop. "finish" is handled in the loop itself.
TOOLS = {
    "calculator": calculator,
    "wiki_lookup": wiki_lookup,
}


if __name__ == "__main__":
    # Step 2 checkpoint: test the tools BEFORE connecting the model.
    print(calculator("4*185 + 3*95"))         # expect 1025
    print(calculator("1,200 - 1,025"))        # expect 175
    print(calculator("__import__('os')"))     # expect an Error message
    print(wiki_lookup("Bono East Region")[:200])
    print(wiki_lookup("Zzzz not a real page"))
   