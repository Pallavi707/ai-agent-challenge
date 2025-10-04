# tests/test_icici.py
"""
Assignment Check: ICICI Parser
- T3: Ensures the parser implements parse(pdf_path) -> pd.DataFrame
- T4: Ensures parser output matches expected CSV exactly (via DataFrame.equals)
"""
import pandas as pd
from importlib import import_module
from pathlib import Path
import sys, os

# Ensure repo root is on path
sys.path.append(os.path.abspath("."))

# Input and expected output files
PDF_PATH = "data/icici/icici sample.pdf"
CSV_PATH = "data/icici/result.csv"

def load_expected():
    """Loads the expected CSV output (result.csv) into a pandas DataFrame.
    This DataFrame is used as the ground truth for comparing the parser’s output."""
    return pd.read_csv(Path(CSV_PATH))

def test_icici_parser_contract():
    """# Dynamically imports the generated ICICI parser module 
    and checks that it has a `parse` function (T3). Runs the parser on the sample PDF, 
    aligns its output with the expected DataFrame, and converts dtypes if needed. Finally, 
    verifies that the output matches the expected CSV exactly using DataFrame.equals (T4)."""
    
    parser_mod = import_module("custom_parser.icici_parser")

    # Contract check (T3)
    assert hasattr(parser_mod, "parse"), "parse(pdf_path) not found in icici_parser.py"

    # Run parser
    out_df = parser_mod.parse(PDF_PATH)
    exp_df = load_expected()

    # Align columns & dtypes for fair comparison
    out_df = out_df[exp_df.columns]
    for c in exp_df.columns:
        try:
            out_df[c] = out_df[c].astype(exp_df[c].dtype)
        except Exception:
            pass

    # Exact output check (T4)
    assert out_df.equals(exp_df), "Parsed DataFrame does not match expected CSV"

    print("Parser contract satisfied, output matches expected CSV")
