

import glob
import os
import re
import sys

from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import BadRequestError

from tools import TOOLS

load_dotenv()                                  # reads .env (LLM_MODEL, keys)
load_dotenv(os.path.join("..", ".env"))        # also try the repo root
MODEL = os.getenv("LLM_MODEL")

# ---------------------------------------------------------------
# Step 3a: the ReAct system prompt
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant for a trader at Makola Market in Ghana.
You solve problems step by step using tools.

Available actions:
- calculator: evaluates arithmetic. Input must be plain arithmetic only, for example 4*185 + 3*95
- wiki_lookup: gets a short Wikipedia summary. Input is a page title, for example Bono East Region
- finish: use this when you have the final answer. Action Input is the final answer.

On EVERY step reply in exactly this format and nothing else:

Thought: <your reasoning about what to do next>
Action: <calculator | wiki_lookup | finish>
Action Input: <input for the action>

Rules:
- Reply in PLAIN TEXT only. Do NOT use function calling, tool calls or JSON.
  The actions above are just words you write in the format shown.
- Always begin your reply with "Thought:".
- Write ONE action per reply, then stop and wait for the Observation.
- Never write an Observation yourself. It will be given to you.
- Never do arithmetic in your head. Always use the calculator.
- If a tool returns an error, try a different input or explain the problem in finish.

Example of one correct reply (plain text, nothing else):

Thought: I need the total cost of 2 items at 10 each and 3 items at 5 each, so I will use the calculator.
Action: calculator
Action Input: 2*10 + 3*5
"""

# ---------------------------------------------------------------
# Step 3b: the parser
# ---------------------------------------------------------------
ACTION_RE = re.compile(r"Action:\s*`?(\w+)`?\s*\nAction Input:\s*(.*)", re.DOTALL)


def parse(text):
    """Return (action, action_input), or (None, None) if the reply is malformed."""
    m = ACTION_RE.search(text)
    if not m:
        return None, None
    action = m.group(1).strip().lower()
    raw = m.group(2).strip()
    if action == "finish":
        # Keep the full answer, but cut it off if the model starts another step
        raw = re.split(r"\n\s*(?:Thought|Action|Observation):", raw)[0]
    else:
        # A tool input is ONE line. Anything after it is the model running ahead.
        raw = raw.splitlines()[0] if raw else ""
    return action, raw.strip().strip("`\"'").strip()


# ---------------------------------------------------------------
# Trace helpers (Step 4): print AND save every run
# ---------------------------------------------------------------
def next_trace_path():
    os.makedirs("evidence", exist_ok=True)
    n = len(glob.glob("evidence/trace_*.txt")) + 1
    return f"evidence/trace_{n}.txt"


class Trace:
    def __init__(self):
        self.lines = []

    def log(self, text):
        print(text)
        self.lines.append(text)

    def save(self):
        path = next_trace_path()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\n[trace saved to {path}]")


# ---------------------------------------------------------------
# Step 4: the loop (with Step 5 robustness)
# ---------------------------------------------------------------
FORMAT_REMINDER = (
    "Your reply could not be parsed. Reply in exactly this format:\n"
    "Thought: <reasoning>\n"
    "Action: <calculator | wiki_lookup | finish>\n"
    "Action Input: <input>"
)


def call_model(messages) -> str:
    """Call the LLM. If the provider rejects a native tool call (Groq's
    'tool_use_failed' 400), return that text as a malformed reply instead of
    crashing, so the loop can send the corrective Observation."""
    try:
        resp = completion(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],   # stops the model inventing its own results
        )
        return (resp.choices[0].message.content or "").strip()
    except BadRequestError as e:
        err = str(e)
        if "tool_use_failed" not in err:
            raise                    # a different 400: let it show
        m = re.search(r'failed_generation"\s*:\s*"(.*?)(?<!\\)"\s*\}\s*\}', err, re.DOTALL)
        raw = (m.group(1) if m else err[:300]).replace('\\"', '"')
        return f"[provider blocked a native tool call] {raw}"


def run_agent(question: str, max_steps: int = 6) -> str:
    if not MODEL:
        msg = "LLM_MODEL is not set. Check your .env file."
        print(msg)
        return msg

    trace = Trace()
    trace.log(f"Question: {question}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = None

    for step in range(1, max_steps + 1):
        trace.log(f"--- Step {step} ---")

        reply = call_model(messages)
        if not reply:                      # some providers reject empty messages
            reply = "(empty reply)"
        trace.log(reply)

        # Keep the model's own words in the history (its "beliefs")
        messages.append({"role": "assistant", "content": reply})

        action, action_input = parse(reply)

        # Failure 1: unparseable reply -> corrective Observation
        if action is None:
            obs = FORMAT_REMINDER
            trace.log(f"Observation: {obs}\n")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            continue

        # Finish: the agent decided it is done
        if action == "finish":
            answer = action_input
            trace.log(f"\nFinal answer: {answer}")
            break

        # Failure 2: unknown tool name -> corrective Observation
        if action not in TOOLS:
            obs = (f"Error: unknown action '{action}'. "
                   f"Valid actions are: {', '.join(TOOLS)}, finish.")
        else:
            obs = TOOLS[action](action_input)   # YOUR CODE runs the tool, not the model

        trace.log(f"Observation: {obs}\n")
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    # Failure 3: ran out of steps
    if answer is None:
        answer = f"Stopped after {max_steps} steps without a final answer."
        trace.log(f"\n{answer}")

    trace.save()
    return answer


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "I have GHS 1,200. I want 4 bags of rice at GHS 185 each and "
        "3 gallons of oil at GHS 95 each. How much will I have left?"
    )
    run_agent(q)