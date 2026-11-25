"""Lab 3, Steps 3 and 5: an agent that DISCOVERS tools from the MCP server.

Interactive:   python client_agent.py
Batch (six Qs): python client_agent.py --batch
Run from inside lab03/.  The agent never imports the tool functions.
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from litellm import acompletion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
MODEL = os.getenv("LLM_MODEL")
MAX_STEPS = 6  # free-tier protection: never loop forever
LOG = Path("evidence/tool_calls.jsonl")

SYSTEM = (
    "You are a community pharmacy stock assistant. Use the tools for every "
    "stock, price, expiry or sale question; never guess. Only record a sale "
    "when the user explicitly asks. If a tool refuses, relay the reason "
    "clearly. Keep answers short."
)

BATCH_QUESTIONS = [
    "Do we have amoxicillin 500mg?",
    "Which items expire within 60 days?",
    "What do 2 packs of paracetamol cost?",
    "Search for anything in tablet form.",
    "Sell 2 packs of paracetamol 500mg.",      # valid sale
    "Sell 5000 packs of amoxicillin 500mg.",   # oversale -> must be refused
]

params = StdioServerParameters(command=sys.executable, args=["server.py"])


def log_call(name: str, args: dict, result: str, ms: float) -> None:
    LOG.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": name,
        "arguments": args,
        "result_summary": result[:200],
        "duration_ms": round(ms, 1),
    }
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


async def run_turn(session: ClientSession, schemas: list, messages: list) -> str:
    """One user turn: model -> tool_calls -> session.call_tool -> model ..."""
    for _ in range(MAX_STEPS):
        resp = await acompletion(model=MODEL, messages=messages,
                                 tools=schemas, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:                       # no action requested: final reply
            return msg.content or ""

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            start = time.perf_counter()
            result = await session.call_tool(tc.function.name, args)   # we execute
            ms = (time.perf_counter() - start) * 1000
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            log_call(tc.function.name, args, text, ms)
            print(f"  [tool] {tc.function.name}({args}) -> {text[:80]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

    return "Sorry, I hit my step limit without finishing."


async def main():
    if not MODEL:
        raise SystemExit("LLM_MODEL is not set. Check your .env file.")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print("Discovered tools:", [t.name for t in tools])

            schemas = [{"type": "function", "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            }} for t in tools]

            messages = [{"role": "system", "content": SYSTEM}]

            if "--batch" in sys.argv:
                for q in BATCH_QUESTIONS:
                    print(f"\nYou: {q}")
                    messages.append({"role": "user", "content": q})
                    print("Bot:", await run_turn(session, schemas, messages))
                return

            while True:
                q = input("\nYou: ").strip()
                if q.lower() in {"quit", "exit"}:
                    break
                messages.append({"role": "user", "content": q})
                print("Bot:", await run_turn(session, schemas, messages))


if __name__ == "__main__":
    asyncio.run(main())