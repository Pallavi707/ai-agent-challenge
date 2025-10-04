import pandas as pd
import pdfplumber
import re
from datetime import datetime

def parse(pdf_path: str) -> pd.DataFrame:
    all_transactions_data = [] # To store dictionaries for DataFrame construction
    
    # Define a robust mapping from various possible PDF header strings to target column names
    # Using regex patterns for flexibility and case-insensitivity
    header_patterns = {
        r"date": "Date",
        r"txn\.?\s*date": "Date",
        r"transaction\s*date": "Date",
        r"value\s*date": "Date",
        
        r"description": "Description",
        r"narration": "Description",
        r"particulars": "Description",
        r"details": "Description",
        
        r"(withdrawal|debit)\s*(amt)?": "Debit Amt", # amt? to catch 'Debit Amt' or 'Debit'
        r"withdrawals": "Debit Amt",
        r"dr\.?": "Debit Amt", # Dr. or Dr
        
        r"(deposit|credit)\s*(amt)?": "Credit Amt", # amt? to catch 'Credit Amt' or 'Credit'
        r"deposits": "Credit Amt",
        r"cr\.?": "Credit Amt", # Cr. or Cr
        
        r"balance": "Balance",
        r"closing\s*balance": "Balance",
        r"running\s*balance": "Balance",
    }

    # Common date formats to try for ICICI statements
    date_formats = [
        "%d-%b-%Y", # e.g., 01-Jan-2023
        "%d/%m/%Y", # e.g., 01/01/2023
        "%d.%m.%Y", # e.g., 01.01.2023
        "%Y-%m-%d", # In case it's already in this format or an intermediate state
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract tables using strategies that work well for structured data
            tables = page.extract_tables({
                "vertical_strategy": "lines", # Use explicit lines for column detection
                "horizontal_strategy": "lines", # Use explicit lines for row detection
                "snap_tolerance": 3, # Tolerance for snapping to lines
                "join_tolerance": 3, # Tolerance for joining words in a cell
            })

            for table in tables:
                if not table or not table[0]: # Skip empty tables or tables with no header row
                    continue

                # Clean header names: remove extra spaces, newlines, convert to lowercase for robust matching
                cleaned_header = [re.sub(r'\s+', ' ', str(h or '').strip()).lower() for h in table[0]]
                
                # Map cleaned PDF header column names to target column names
                col_idx_map = {} # Stores {target_col_name: index_in_pdf_table}
                for i, pdf_col_name_cleaned in enumerate(cleaned_header):
                    for pattern, target_col_name in header_patterns.items():
                        if re.search(pattern, pdf_col_name_cleaned):
                            col_idx_map[target_col_name] = i
                            break # Found a match for this PDF column, move to next PDF column
                
                # A table is considered a transaction table if it has at least 'Date', 'Description', and one amount column
                required_cols_found = 'Date' in col_idx_map and 'Description' in col_idx_map and \
                                      ('Debit Amt' in col_idx_map or 'Credit Amt' in col_idx_map or 'Balance' in col_idx_map)
                
                if not required_cols_found:
                    continue # Skip tables that don't match transaction table criteria

                # Process data rows (skipping the header row which is table[0])
                for row_data in table[1:]:
                    # Clean row data: remove extra spaces, newlines
                    cleaned_row = [re.sub(r'\s+', ' ', str(cell or '').strip()) for cell in row_data]
                    
                    # Heuristic to skip empty rows or rows that are merely repeated headers on new pages
                    # or footers, or rows with too little relevant data.
                    if not any(cell for cell in cleaned_row): # Skip entirely empty rows
                        continue
                    
                    # Check if the row contains values that look like header names (e.g., "Date", "Description")
                    # This helps to filter out repeated headers in multi-page tables.
                    if any(re.search(pat, cell.lower()) for pat in ['date', 'description', 'amount', 'balance'] for cell in cleaned_row if cell):
                        continue

                    # Initialize a dictionary for the current transaction with all target columns set to None
                    transaction = {col: None for col in ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']}
                    
                    # Populate transaction dictionary using the identified column index map
                    for target_col, col_idx in col_idx_map.items():
                        if col_idx < len(cleaned_row):
                            transaction[target_col] = cleaned_row[col_idx]
                    
                    # --- Post-processing and validation for a single transaction ---
                    date_str = transaction.get('Date', '')
                    desc_str = transaction.get('Description', '')

                    is_date_parsed = False
                    for fmt in date_formats:
                        try:
                            # Attempt to parse date using known formats
                            parsed_date = datetime.strptime(date_str, fmt)
                            transaction['Date'] = parsed_date.strftime('%Y-%m-%d') # Standardize date format
                            is_date_parsed = True
                            break
                        except ValueError:
                            pass # Try next format if current one fails
                    
                    if not is_date_parsed:
                        # If date parsing failed, this row might be a continuation of the previous description.
                        # Check if it has a description and if numerical columns are empty/non-numerical.
                        if all_transactions_data and desc_str:
                            is_continuation = True
                            # Iterate through numerical columns to see if they contain valid numbers.
                            # If they do, it's likely a new (but malformed) transaction, not a description continuation.
                            for col_name in ['Debit Amt', 'Credit Amt', 'Balance']:
                                val = transaction.get(col_name)
                                # Check if the value looks like a number (contains digits, optional decimal/comma/minus)
                                if val and re.search(r'\d', str(val)) and re.match(r'^-?[\d,.]+$', str(val).replace(',', '')):
                                    is_continuation = False
                                    break
                            
                            if is_continuation:
                                # Append the current description to the last transaction's description
                                all_transactions_data[-1]['Description'] += '\n' + desc_str.strip()
                                continue # This row was a description continuation, so skip adding as a new transaction
                        
                        continue # Skip this row if date couldn't be parsed and it's not a description continuation

                    # Parse and convert amount columns to float
                    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
                        val_str = str(transaction.get(col, '')).strip()
                        # Remove non-numeric characters except for digits, decimal point, and leading hyphen
                        val_str = re.sub(r'[^\d.-]', '', val_str) 
                        
                        if val_str == '-' or not val_str: # Treat empty string or '-' as 0.0
                            transaction[col] = 0.0
                        else:
                            try:
                                transaction[col] = float(val_str)
                            except ValueError:
                                transaction[col] = 0.0 # Default to 0.0 if conversion fails
                    
                    # Ensure description is a string and remove leading/trailing whitespace
                    transaction['Description'] = desc_str.strip()
                    
                    # Add the fully processed transaction dictionary to our list
                    all_transactions_data.append(transaction)

    # Create a pandas DataFrame from the collected transaction data
    df = pd.DataFrame(all_transactions_data)

    # Ensure all target columns exist and are in the correct order, adding missing ones if necessary
    target_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    for col in target_columns:
        if col not in df.columns:
            if col == 'Date':
                df[col] = pd.NaT # Use NaT (Not a Time) for missing dates
            elif col == 'Description':
                df[col] = ''
            else:
                df[col] = 0.0 # Use 0.0 for missing numeric amounts
    
    # Reorder DataFrame columns to match the target specification
    df = df[target_columns]

    # Final type conversions and cleaning to ensure data integrity
    # Convert 'Date' column to datetime objects and then format to 'YYYY-MM-DD' string
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['Date'] = df['Date'].fillna('') # Fill NaT (if any) with empty string after formatting

    df['Description'] = df['Description'].fillna('').astype(str) # Fill NaN descriptions with empty string
    
    # Fill NaN values in numeric columns with 0.0 and ensure they are of float type
    for col in ['Debit Amt', 'Credit Amt', 'Balance']:
        df[col] = df[col].fillna(0.0).astype(float)

    return df