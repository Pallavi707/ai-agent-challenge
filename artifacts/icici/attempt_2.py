import numpy as np
import re
import pandas as pd
import pdfplumber
from typing import Optional

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses an ICICI bank statement PDF and extracts transaction details into a DataFrame.

    Args:
        pdf_path (str): The path to the ICICI bank statement PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted transactions with columns:
                      ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].

    Raises:
        ValueError: If no transactions could be extracted from the PDF.
    """
    all_transactions = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            
            for table in tables:
                if not table:
                    continue

                header_row = table[0]
                
                # Normalize header names (remove newlines, strip spaces)
                # pdfplumber often extracts multi-line headers like "Debit\nAmt" as "Debit Amt"
                normalized_header = [h.replace('\n', ' ').strip() if h is not None else '' for h in header_row]

                # Identify the transaction table by checking for key column names
                # Based on the EXPECTED SCHEMA, the PDF headers, once normalized, are expected
                # to be 'Date', 'Description', 'Debit Amt', 'Credit Amt', and 'Balance'.
                # The previous error occurred because it was looking for 'Debit' and 'Credit'
                # instead of 'Debit Amt' and 'Credit Amt'.
                if 'Date' in normalized_header and \
                   'Description' in normalized_header and \
                   'Debit Amt' in normalized_header and \
                   'Credit Amt' in normalized_header and \
                   'Balance' in normalized_header:
                    
                    # Get column indices for reliable data extraction using the exact expected names
                    date_idx = normalized_header.index('Date')
                    desc_idx = normalized_header.index('Description')
                    debit_idx = normalized_header.index('Debit Amt')
                    credit_idx = normalized_header.index('Credit Amt')
                    balance_idx = normalized_header.index('Balance')
                    
                    # Process rows after the header
                    for row in table[1:]:
                        if not row or not any(row): # Skip empty or entirely None rows
                            continue
                        
                        # The maximum index ensures the row must be at least this long to access all required columns
                        required_len = max(date_idx, desc_idx, debit_idx, credit_idx, balance_idx) + 1
                        if len(row) < required_len:
                            # This row is likely malformed or an irrelevant table artifact, skip it.
                            continue 
                        
                        # Helper function to convert string to float or pd.NA
                        def to_float_or_na(val: Optional[str]) -> Optional[float]:
                            if val is None or str(val).strip() == '':
                                return pd.NA # Use pandas' native missing value for nullable numeric types
                            try:
                                # Remove commas (common in large numbers) and extra spaces, then convert to float
                                return float(str(val).replace(',', '').strip())
                            except ValueError:
                                # Return pd.NA for values that cannot be converted to float
                                return pd.NA
                        
                        # Extract and clean Date and Description values
                        date_val_raw = row[date_idx]
                        desc_val_raw = row[desc_idx]
                        date_val = str(date_val_raw).strip() if date_val_raw is not None else ''
                        desc_val = str(desc_val_raw).strip() if desc_val_raw is not None else ''

                        # Extract and convert Debit, Credit, Balance values to float or pd.NA
                        debit_val = row[debit_idx]
                        credit_val = row[credit_idx]
                        balance_val = row[balance_idx]
                        
                        debit_amt: Optional[float] = to_float_or_na(debit_val)
                        credit_amt: Optional[float] = to_float_or_na(credit_val)
                        balance_amt: Optional[float] = to_float_or_na(balance_val)
                        
                        # Append the extracted transaction to the list
                        all_transactions.append({
                            'Date': date_val,
                            'Description': desc_val,
                            'Debit Amt': debit_amt,
                            'Credit Amt': credit_amt,
                            'Balance': balance_amt
                        })
                        
    if not all_transactions:
        raise ValueError("no transactions extracted")

    # Create DataFrame from the collected transactions
    df = pd.DataFrame(all_transactions)
    
    # Ensure the DataFrame columns are in the exact specified order
    df = df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]
    
    # Data types should be correct based on extraction logic:
    # 'Date' and 'Description' will be object (string)
    # 'Debit Amt', 'Credit Amt', 'Balance' will be Float64 (pandas nullable float type due to pd.NA usage)
    
    return df