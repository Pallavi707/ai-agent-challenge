"""
This is an AI agent that writes a parser, a program that reads a bank PDF and 
turns it into a clean table. It first asks Gemini AI to generate the parser code.
Then, it runs tests to see if the output matches the correct CSV. If the tests fail, 
it shows the errors to Gemini and asks it to fix the code automatically. 
This process repeats until it works or gives up after a few tries.
"""

# Importing libraries
import argparse, os, re, subprocess, sys
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# LangGraph
from langgraph.graph import StateGraph, END

# make sure project root is in path
sys.path.append(os.path.abspath("."))

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "models/gemini-2.5-flash"

# Paths
PARSER_PATH = Path("custom_parser/icici_parser.py")
TEST_CMD = ["pytest", "-q"]

SYSTEM_PROMPT = """You are a coding agent that writes a PDF bank-statement parser.

TARGET FILE TO WRITE: custom_parser/icici_parser.py
CONTRACT:
    def parse(pdf_path: str) -> pandas.DataFrame
MUST MATCH EXACTLY: data/icici/result.csv via DataFrame.equals in tests/test_icici.py

REQUIREMENTS:
- Use Python only. No network.
- Use pdfplumber to read the PDF (tables + text).
- Must include at top:
      import pandas as pd
      import pdfplumber
      import re
      from datetime import datetime
- Deterministic, robust to whitespace and line breaks.
- Return a DataFrame with the SAME columns and ordering as result.csv.
- Output ONLY the full python source of icici_parser.py (no prose).
"""

# --- Helper functions ---
def llm(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL)
    resp = model.generate_content(prompt)
    return resp.text

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

def enforce_imports(code: str) -> str:
    """Ensure required imports are present."""
    required = [
        "import pandas as pd",
        "import pdfplumber",
        "import re",
        "from datetime import datetime"
    ]
    for imp in required:
        if imp not in code:
            code = imp + "\n" + code
    return code

def run_pytest() -> Tuple[int, str]:
    p = subprocess.Popen(TEST_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out

def write_parser(code: str):
    code = enforce_imports(code)  # <-- enforce imports here
    PARSER_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARSER_PATH.write_text(code, encoding="utf-8")

# --- LangGraph nodes ---
def generate_parser(state: dict):
    target = state["target"]
    user_ctx = f"""
BANK: {target}
PDF PATH: data/{target}/{target} sample.pdf
EXPECTED CSV: data/{target}/result.csv

Write the COMPLETE Python source for custom_parser/{target}_parser.py implementing parse(pdf_path).
"""
    print("\n[T1] Agent generating parser with Gemini…")
    code = llm(SYSTEM_PROMPT + "\n\n" + user_ctx)
    write_parser(_extract_code(code))
    state["attempt"] = 1
    return state

def run_tests(state: dict):
    print(f"\n[T4] Running tests on generated parser (attempt {state['attempt']}) …")
    rc, out = run_pytest()
    print(out)
    state["rc"] = rc
    state["test_output"] = out
    return state

def check_result(state: dict):
    if state["rc"] == 0:
        print("[T3+T4] Parser contract satisfied and tests passed.\n")
        return {END: state}

    if state["attempt"] >= state["max_attempts"]:
        print("Still failing after max attempts.\n")
        return {END: state}

    fix_prompt = f"""{SYSTEM_PROMPT}

The current implementation FAILED tests. Full pytest output:

<TEST_OUTPUT>
{state['test_output']}
</TEST_OUTPUT>

Rewrite the ENTIRE file custom_parser/{state['target']}_parser.py to fix the failure.
Output ONLY the full python source (no backticks, no prose).
"""
    print(f"[T1+T2] Parser failed tests, regenerating and self-fixing (attempt {state['attempt']+1}) …")
    fixed = llm(fix_prompt)
    write_parser(_extract_code(fixed))
    state["attempt"] += 1
    return {"run_tests": state}

# --- Main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="icici")
    ap.add_argument("--max_attempts", type=int, default=3)
    args = ap.parse_args()

    graph = StateGraph(dict)
    graph.add_node("generate_parser", generate_parser)
    graph.add_node("run_tests", run_tests)
    graph.add_node("check_result", check_result)

    graph.set_entry_point("generate_parser")
    graph.add_edge("generate_parser", "run_tests")
    graph.add_edge("run_tests", "check_result")

    app = graph.compile()
    app.invoke({"target": args.target, "max_attempts": args.max_attempts, "attempt": 0})

if __name__ == "__main__":
    main()
