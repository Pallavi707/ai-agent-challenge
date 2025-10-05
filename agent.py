"""
This is an AI agent that writes a parser, a program that reads a bank PDF and 
turns it into a clean table. It first asks Gemini AI to generate the parser code.
Then, it runs tests to see if the output matches the correct CSV. If the tests fail, 
it shows the errors to Gemini and asks it to fix the code automatically. 
This process repeats until it works or gives up after a few tries.
"""

# Importing libraries
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any
import re
import traceback

from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import numpy as np
from importlib import import_module
import importlib
from langgraph.graph import StateGraph, END

# Make sure project root is in path
sys.path.append(os.path.abspath("."))

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_CANDIDATES = ["models/gemini-2.5-flash"]

# --- System Prompt ---
SYSTEM_PROMPT = """You are a Python expert that writes PDF parsers for bank statements.

TASK: Write a parser for {target} bank statement PDF.

TARGET FILE: custom_parser/{target}_parser.py
FUNCTION: def parse(pdf_path: str) -> pd.DataFrame

REFERENCE DATA:
- PDF: {pdf_path}
- Expected CSV: {csv_path}

EXPECTED SCHEMA (from CSV):
{csv_schema}

YOUR APPROACH:
1. Use pdfplumber to extract tables from the PDF
2. Map extracted columns to match the CSV schema exactly
3. Handle data types correctly (dates as strings, amounts as float/pd.NA)
4. Return DataFrame with exact column names and order from CSV

REQUIREMENTS:
- Must import: pandas as pd, pdfplumber
- Process all pages of the PDF
- Use basic pdfplumber API: page.extract_tables() or page.extract_table()
- Do NOT use advanced settings, custom strategies, or cropping unless necessary
- If no data extracted: raise ValueError("no transactions extracted")
- Avoid type hints like 'float | pd.NA' - use Optional[float] instead

BASIC PATTERN:
```python
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        # process tables
```

OUTPUT: Python code only. No markdown, no explanations.
"""

REFLECTION_PROMPT = """The parser failed. Analyze and fix it.

ORIGINAL TASK:
{system_prompt}

YOUR CODE:
```python
{current_code}
```

TEST FAILURE:
{test_output}

Debug the failure, identify the root cause, and rewrite the parser with fixes.

OUTPUT: Complete Python code only.
"""

# --- Validation functions ---
def validate_parser_code(code: str, target: str, expected_columns: list) -> tuple[bool, str]:
    """
    Validate generated code before writing to file.
    Checks for required function signature, imports, and column references.
    """
    
    # Check 1: Must have parse function
    if not re.search(r"def\s+parse\s*\([^)]*pdf_path[^)]*\)", code):
        return False, "Missing function: def parse(pdf_path: str) -> pd.DataFrame"
    
    # Check 2: Must import required libraries
    required = ["import pandas as pd", "import pdfplumber"]
    missing = [imp for imp in required if imp not in code]
    if missing:
        return False, f"Missing imports: {', '.join(missing)}"
    
    # Check 3: Should return DataFrame
    if "return" not in code or "DataFrame" not in code:
        return False, "Function must return a pandas DataFrame"
    
    # Check 4: Should reference expected columns (soft check)
    col_mentions = sum(1 for col in expected_columns if col in code)
    if col_mentions == 0:
        return False, f"Code doesn't reference any expected columns: {expected_columns}"
    
    return True, "Validation passed"

# --- Helper functions ---
def llm(prompt: str, state: Dict[str, Any]) -> str:
    """
    Call Gemini API with fallback models.
    Tries each model in MODEL_CANDIDATES until one succeeds.
    Logs which model was used in state.
    """
    last_exc = None
    for model_name in MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            result = (resp.text or "").strip()
            state["last_model"] = model_name
            return result
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All models failed. Last error: {last_exc}")

def extract_code(text: str) -> str:
    """
    Extract code from markdown fences or return as-is.
    Handles both ```python and ``` code blocks.
    """
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def ensure_imports(code: str) -> str:
    """
    Add missing critical imports at the top of the code.
    Checks existing imports to avoid duplicates.
    """
    imports = [
        "import pandas as pd",
        "import pdfplumber", 
        "import re",
        "import numpy as np",
        "from typing import Optional"
    ]
    existing_imports = [line for line in code.split('\n') if line.strip().startswith(('import ', 'from '))]
    
    for imp in imports:
        if not any(imp in existing for existing in existing_imports):
            code = f"{imp}\n{code}"
    
    return code

def write_parser(target: str, code: str, attempt: int) -> None:
    """
    Write parser code to file and archive attempt for debugging.
    Creates custom_parser directory and artifacts directory for version history.
    """
    parser_dir = Path("custom_parser")
    parser_dir.mkdir(exist_ok=True)
    (parser_dir / "__init__.py").touch()
    
    # Write main parser
    parser_path = parser_dir / f"{target}_parser.py"
    parser_path.write_text(ensure_imports(code), encoding="utf-8")
    
    # Archive attempt for debugging
    archive_dir = Path("artifacts") / target
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"attempt_{attempt}.py").write_text(code, encoding="utf-8")
    
    print(f"  Written to {parser_path} (archived as attempt_{attempt}.py)")

def read_current_code(target: str) -> str:
    """
    Read current parser code from file.
    Returns empty string if file doesn't exist.
    """
    parser_path = Path(f"custom_parser/{target}_parser.py")
    if parser_path.exists():
        return parser_path.read_text()
    return ""

def get_csv_schema(csv_path: Path) -> str:
    """
    Extract schema info from expected CSV to guide the agent.
    Shows column names, sample data, and inferred types.
    """
    df = pd.read_csv(csv_path, nrows=3)
    schema = f"Columns: {list(df.columns)}\n"
    schema += f"Sample data (first 3 rows):\n{df.to_string()}\n"
    schema += f"\nData types expected:\n"
    for col in df.columns:
        sample_val = df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else "empty"
        schema += f"  - {col}: {type(sample_val).__name__} (example: {sample_val})\n"
    return schema

# --- Testing functions ---
def run_test(target: str) -> tuple[int, str]:
    """
    Run parser and compare output with expected CSV.
    Returns (0, message) on success, (1, error_message) on failure.
    Performs normalization and type-aware comparison.
    """
    pdf_path = Path(f"data/{target}/{target} sample.pdf")
    csv_path = Path(f"data/{target}/result.csv")
    
    if not pdf_path.exists():
        return 1, f"PDF not found: {pdf_path}"
    if not csv_path.exists():
        return 1, f"CSV not found: {csv_path}"
    
    try:
        # Force reload to pick up code changes
        module_name = f"custom_parser.{target}_parser"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        parser = import_module(module_name)
        
        # Run parser
        df_actual = parser.parse(str(pdf_path))
        df_expected = pd.read_csv(csv_path)
        
        # Normalize both DataFrames for comparison
        # Strip whitespace from string columns
        for col in df_actual.columns:
            if df_actual[col].dtype == 'object':
                df_actual[col] = df_actual[col].astype(str).str.strip()
        
        for col in df_expected.columns:
            if df_expected[col].dtype == 'object':
                df_expected[col] = df_expected[col].astype(str).str.strip()
        
        # Replace various null representations with pd.NA
        df_actual = df_actual.replace(['nan', 'None', ''], pd.NA)
        df_expected = df_expected.replace(['nan', 'None', ''], pd.NA)
        
        # Validate structure
        if list(df_actual.columns) != list(df_expected.columns):
            return 1, f"Column mismatch!\nExpected: {list(df_expected.columns)}\nGot: {list(df_actual.columns)}"
        
        if len(df_actual) != len(df_expected):
            return 1, f"Row count mismatch: expected {len(df_expected)}, got {len(df_actual)}\n\nFirst 3 rows of actual output:\n{df_actual.head(3)}"
        
        # Compare row by row (first 10 rows for detailed feedback)
        errors = []
        for i in range(min(10, len(df_expected))):
            for col in df_expected.columns:
                exp_val = df_expected.iloc[i][col]
                act_val = df_actual.iloc[i][col]
                
                # Handle NaN comparison
                exp_is_na = pd.isna(exp_val)
                act_is_na = pd.isna(act_val)
                
                if exp_is_na and act_is_na:
                    continue
                
                if exp_is_na != act_is_na:
                    errors.append(f"  Row {i}, {col}: expected '{exp_val}', got '{act_val}'")
                    continue
                
                # Type-aware comparison
                if isinstance(exp_val, (int, float)) and not exp_is_na:
                    try:
                        if abs(float(exp_val) - float(act_val)) > 0.01:
                            errors.append(f"  Row {i}, {col}: expected {exp_val}, got {act_val}")
                    except (ValueError, TypeError):
                        errors.append(f"  Row {i}, {col}: expected number {exp_val}, got '{act_val}'")
                else:
                    # String comparison
                    if str(exp_val) != str(act_val):
                        errors.append(f"  Row {i}, {col}: expected '{exp_val}', got '{act_val}'")
        
        if errors:
            error_msg = f"Data mismatch in {len(errors)} field(s):\n" + "\n".join(errors[:15])
            if len(errors) > 15:
                error_msg += f"\n  ... and {len(errors) - 15} more errors"
            return 1, error_msg
        
        return 0, "All tests passed!"
        
    except Exception as e:
        tb = traceback.format_exc()
        return 1, f"Runtime error:\n{str(e)}\n\nTraceback:\n{tb}"

# --- LangGraph nodes ---
def generate_initial_parser(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Generate first version of parser.
    Reads CSV schema and asks LLM to generate parser code.
    """
    target = state["target"]
    pdf_path = f"data/{target}/{target} sample.pdf"
    csv_path = f"data/{target}/result.csv"
    
    print(f"\n{'='*60}")
    print(f"Generating parser for {target.upper()} bank...")
    print(f"{'='*60}")
    
    # Get expected schema from CSV
    csv_schema = get_csv_schema(Path(csv_path))
    expected_columns = list(pd.read_csv(csv_path, nrows=1).columns)
    state["expected_columns"] = expected_columns
    
    prompt = SYSTEM_PROMPT.format(
        target=target,
        pdf_path=pdf_path,
        csv_path=csv_path,
        csv_schema=csv_schema
    )
    
    # Generate code
    code = extract_code(llm(prompt, state))
    
    # Validate before writing
    valid, msg = validate_parser_code(code, target, expected_columns)
    if not valid:
        print(f"Validation warning: {msg}")
        print("  Proceeding anyway, will catch errors in testing...")
    
    write_parser(target, code, attempt=1)
    
    state["attempt"] = 1
    return state

def test_parser(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Run tests on generated parser.
    Compares parser output with expected CSV.
    """
    print(f"\nTesting parser (attempt {state['attempt']}/3)...")
    
    rc, output = run_test(state["target"])
    state["rc"] = rc
    state["test_output"] = output
    
    print(output)
    return state

def decide_next(state: Dict[str, Any]) -> str:
    """
    Decision node: determine next action based on test results.
    Returns 'success' if tests pass, 'retry' if attempts remain, 'failure' otherwise.
    """
    if state["rc"] == 0:
        return "success"
    elif state["attempt"] < 3:
        return "retry"
    else:
        return "failure"

def self_correct(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Self-correct based on test failures.
    Shows LLM the current code and test errors, asks for fixes.
    """
    print(f"\nSelf-correcting (attempt {state['attempt'] + 1}/3)...")
    
    target = state["target"]
    current_code = read_current_code(target)
    pdf_path = f"data/{target}/{target} sample.pdf"
    csv_path = f"data/{target}/result.csv"
    csv_schema = get_csv_schema(Path(csv_path))
    
    system_prompt = SYSTEM_PROMPT.format(
        target=target,
        pdf_path=pdf_path,
        csv_path=csv_path,
        csv_schema=csv_schema
    )
    
    prompt = REFLECTION_PROMPT.format(
        system_prompt=system_prompt,
        current_code=current_code,
        test_output=state["test_output"]
    )
    
    code = extract_code(llm(prompt, state))
    
    state["attempt"] += 1
    write_parser(target, code, attempt=state["attempt"])
    
    return state

def report_success(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Report successful parser generation.
    Shows summary of generated parser and test results.
    """
    print(f"\n{'='*60}")
    print(f"SUCCESS! Parser works after {state['attempt']} attempt(s)")
    print(f"{'='*60}")
    print(f"\nGenerated parser: custom_parser/{state['target']}_parser.py")
    print(f"Function: parse(pdf_path: str) -> pd.DataFrame")
    print(f"Columns: {state['expected_columns']}")
    print(f"Model used: {state.get('last_model', 'unknown')}")
    return state

def report_failure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Report failure after max attempts.
    Shows last error and points to debug artifacts.
    """
    print(f"\n{'='*60}")
    print(f"FAILED after 3 attempts")
    print(f"{'='*60}")
    print(f"\nLast error:\n{state['test_output']}")
    print(f"\nDebug artifacts saved in: artifacts/{state['target']}/")
    print(f"Manual debugging required for: custom_parser/{state['target']}_parser.py")
    return state

# --- Main ---
def main():
    """
    Main entry point. Parses CLI arguments and runs the agent workflow.
    The agent generates a parser, tests it, and self-corrects up to 3 times.
    """
    parser = argparse.ArgumentParser(
        description="AI Agent: Generates bank statement parsers that learn from test failures"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Bank identifier (e.g., icici, sbi, hdfc). Must match folder name in data/"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip generation, only run tests on existing parser (useful for CI)"
    )
    args = parser.parse_args()
    
    # Validate data folder exists
    data_dir = Path(f"data/{args.target}")
    if not data_dir.exists():
        print(f"Error: Data folder not found: {data_dir}")
        print(f"   Expected structure:")
        print(f"   data/{args.target}/{args.target} sample.pdf")
        print(f"   data/{args.target}/result.csv")
        sys.exit(1)
    
    # Offline mode: just run tests
    if args.no_llm:
        print(f"Running tests only (--no-llm mode)")
        rc, output = run_test(args.target)
        print(output)
        sys.exit(rc)
    
    # Build workflow graph
    workflow = StateGraph(dict)
    
    # Add nodes
    workflow.add_node("generate", generate_initial_parser)
    workflow.add_node("test", test_parser)
    workflow.add_node("correct", self_correct)
    workflow.add_node("success", report_success)
    workflow.add_node("failure", report_failure)
    
    # Define edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "test")
    
    workflow.add_conditional_edges(
        "test",
        decide_next,
        {
            "success": "success",
            "retry": "correct",
            "failure": "failure"
        }
    )
    
    workflow.add_edge("correct", "test")
    workflow.add_edge("success", END)
    workflow.add_edge("failure", END)
    
    # Compile and run workflow
    app = workflow.compile()
    initial_state = {
        "target": args.target,
        "attempt": 0,
        "rc": 1,
        "test_output": "",
        "expected_columns": [],
        "last_model": ""
    }
    
    try:
        app.invoke(initial_state)
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()