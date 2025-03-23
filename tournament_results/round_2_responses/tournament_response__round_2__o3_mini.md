Below is the “best‐of‐all‐worlds” full Python implementation. In this solution we first address encoding (using chardet for bytes, stripping BOM and normalizing line endings), then we split into nonempty lines and automatically detect the most likely delimiter (by counting candidates only outside quotes). We then standardize the input (even “fixing” mixed quotes in a lightweight way), parse the header and data rows, adjust rows whose length is off, and clean each field. The cleaning for each field first removes extra whitespace and outer quotes, handles missing/null values and booleans, then (if appropriate) tries both date and number normalization. Date normalization leverages dateutil’s parser (with fuzzy matching, plus a pre-cleaning to remove ordinal suffixes) so that dates like “May 3rd, 1992” become ISO‐formatted “YYYY‑MM‑DD”. Numeric fields are normalized to remove currency symbols and thousand separators (handling both US and European conventions). Finally, the cleaned rows are written out using csv.writer with a consistent comma delimiter and proper quoting. Comments explain key decisions and error‐handling.

Below is the complete code:

------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import io
import chardet
from collections import Counter
from dateutil import parser as date_parser

def normalize_csv(csv_data: str) -> str:
    """
    Clean and normalize messy CSV data.
    
    This function handles:
      - Character encoding issues (including BOM removal)
      - Inconsistent line endings
      - Automatic detection of delimiter (among comma, semicolon, tab, pipe)
      - Mixed quote styles (removing outer quotes)
      - Normalizing header names (lowercase with underscores)
      - Adjusting rows if the field count is off
      - Cleaning each field:
          • Trimming extra whitespace/quotes
          • Converting booleans (e.g. "yes"/"no")
          • Converting dates (using dateutil’s fuzzy parser into ISO YYYY-MM-DD)
          • Converting numbers (handling US vs European styles)
          • Returning empty string for missing/null-like values
    Returns:
        Clean CSV data as a string with comma-separated values.
    """
    # --- 1. Fix encoding, BOM, and line endings ---
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        encoding = detected.get('encoding') or 'utf-8'
        csv_data = csv_data.decode(encoding, errors='replace')
    csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    
    # --- 2. Split into nonempty lines ---
    lines = [line.strip() for line in csv_data.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # --- 3. Detect delimiter among common candidates (',', ';', '\t', '|') ---
    delimiter = detect_delimiter(lines)
    
    # --- 4. Parse CSV using the detected delimiter ---
    # Use csv.reader to allow proper handling of quotes (our cleaning will remove extraneous quotes later)
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter, skipinitialspace=True)
    rows = list(reader)
    if not rows:
        return ""
    
    # --- 5. Normalize header row ---
    header = [normalize_column_name(col) for col in rows[0]]
    
    # --- 6. Process each row (skip fully empty rows, adjust row length) ---
    cleaned_rows = [header]  # first row is header
    for row in rows[1:]:
        # Skip a row if all fields are blank after stripping
        if all(not cell.strip() for cell in row):
            continue
        
        row = adjust_row_length(row, len(header))
        
        # Clean each field based on its content (and if needed, by its column name)
        new_row = []
        for idx, field in enumerate(row):
            new_field = clean_field(field)
            new_field = normalize_value(new_field, header[idx] if idx < len(header) else f'column_{idx}')
            new_row.append(new_field)
        cleaned_rows.append(new_row)
    
    # --- 7. Write normalized rows back to CSV (use comma as delimiter) ---
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerows(cleaned_rows)
    
    return output.getvalue()


# ---------------- Helper Functions ----------------

def detect_delimiter(lines):
    """
    Detect the most likely delimiter by counting occurrences outside quoted regions.
    Considers common delimiters: comma, semicolon, tab, and pipe.
    """
    candidates = [',', ';', '\t', '|']
    delim_counts = {d: 0 for d in candidates}
    for line in lines[:5]:  # analyze first few lines
        for d in candidates:
            delim_counts[d] += count_outside_quotes(line, d)
    # Pick the delimiter with the highest total count
    return max(delim_counts, key=delim_counts.get) if max(delim_counts.values()) > 0 else ','

def count_outside_quotes(line, delimiter):
    """
    Count occurrences of a delimiter outside quoted segments.
    Quotes considered are single and double.
    """
    count = 0
    in_quotes = False
    quote_char = None
    for char in line:
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif char == delimiter and not in_quotes:
            count += 1
    return count

def normalize_column_name(name: str) -> str:
    """
    Normalize a header column name to lowercase with underscores.
    Removes punctuation (except underscores) and extra whitespace.
    """
    name = name.strip().lower()
    # Replace non-word characters with underscore and reduce multiple underscores
    name = re.sub(r'[^\w\s]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')

def adjust_row_length(row, expected_length):
    """
    Change the row to have exactly expected_length fields.
      - If too few, pad with empty strings.
      - If too many, combine extra fields onto the last column.
    """
    if len(row) == expected_length:
        return row
    if len(row) < expected_length:
        return row + [''] * (expected_length - len(row))
    # When too many fields, combine extras into the last field.
    new_row = row[:expected_length-1]
    new_row.append(" ".join(row[expected_length-1:]))
    return new_row

def clean_field(field: str) -> str:
    """
    Clean extraneous whitespace and remove matching outer quotes (both single and double).
    """
    field = field.strip()
    if len(field) >= 2 and field[0] == field[-1] and field[0] in ('"', "'"):
        field = field[1:-1].strip()
    return field

def normalize_value(value: str, column_name: str) -> str:
    """
    Normalize an individual cell’s value.
      • If blank or null-like, return empty string.
      • Convert booleans to standardized lower-case "true"/"false".
      • If the field looks like a date (or belongs to a date column), try to convert to ISO (YYYY-MM-DD).
      • If it looks like a number, remove currency symbols and thousand separators and produce a standard number.
      • Otherwise, return the cleaned string.
    """
    # Handle missing/null values
    if not value or value.lower() in {"null", "none", "na", "n/a", "-"}:
        return ""
    
    # Normalize boolean values
    low = value.lower()
    if low in {"yes", "true", "y", "1"}:
        return "true"
    if low in {"no", "false", "n", "0"}:
        return "false"
    
    # Try date normalization if value contains alphabetic characters or common date delimiters,
    # or if the column name implies a date (e.g., contains "date", "birth", "updated")
    if (re.search(r'[A-Za-z]', value) or re.search(r'[\-/]', value)) or is_date_column(column_name):
        date_norm = try_normalize_date(value)
        if date_norm is not None:
            return date_norm
    
    # If the field contains digits, attempt number normalization.
    if re.search(r'\d', value):
        num_norm = try_normalize_number(value)
        if num_norm is not None:
            return num_norm
    
    # Otherwise, return the value as-is.
    return value

def is_date_column(col: str) -> bool:
    """Heuristic to see if a column name suggests date values."""
    return any(ind in col.lower() for ind in ['date', 'day', 'month', 'year', 'time', 'birth', 'updated', 'created'])

def try_normalize_date(text: str):
    """
    Attempt to parse text as a date (using fuzzy matching).
    Removes ordinal suffixes (e.g. "3rd" becomes "3") before parsing.
    Returns date formatted as YYYY-MM-DD if successful; otherwise, returns None.
    """
    try:
        cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)
        dt = date_parser.parse(cleaned, fuzzy=True)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None

def try_normalize_number(text: str):
    """
    Attempt to normalize a number:
      - Remove common currency symbols [$€£] and whitespace.
      - Handle both US style numbers (with comma as thousand separators)
        and European style (dot as thousand separator and comma as decimal separator).
    Returns the standard number string if parsed successfully; otherwise, None.
    """
    cleaned = re.sub(r'[\s$€£]', '', text)
    # If both comma and dot exist, decide by the rightmost symbol
    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind(',') > cleaned.rfind('.'):
            # Likely European style: thousand separator is dot, decimal is comma.
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # Likely US style: remove commas.
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned and '.' not in cleaned:
        # Ambiguous: if comma appears and if there is only one occurrence at the appropriate position,
        # try European-style conversion.
        if cleaned.count(',') == 1 and len(cleaned.split(',')[-1]) in {1,2}:
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
            
    try:
        num = float(cleaned)
        # Return as integer if no fractional part, otherwise as float string
        if num.is_integer():
            return str(int(num))
        return str(num)
    except Exception:
        return None

# ---------------- Example Usage ----------------
if __name__ == '__main__':
    # Example 1
    csv_example1 = '''
Name, Age, Birth Date, Salary 
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
'''
    print("Normalized CSV Example 1:")
    print(normalize_csv(csv_example1))
    print("-------------")
    
    # Example 2
    csv_example2 = '''
Product Name|Price|In Stock|Last Updated
"Wireless Headphones"|"$129.99"|"Yes"|"2023-01-15"
'Smart Watch';€199,95;'no';'01/22/2023'
"USB-C Cable",$19.99,true,"February 3, 2023"
'''
    print("Normalized CSV Example 2:")
    print(normalize_csv(csv_example2))
    print("-------------")
    
    # Example 3
    csv_example3 = '''
customer_id,first_name,last_name,email,purchase_amount,purchase_date
1001,John,Doe,john.doe@example.com,"1,240.50","01/15/2023"
1002,Jane,Smith,jane.smith@example.com,"985,75","2023-01-20"
1003,"David, Jr.",Johnson,"david.johnson@example.com","2.399,00","Jan 25, 2023"
'''
    print("Normalized CSV Example 3:")
    print(normalize_csv(csv_example3))

------------------------------------------------------------

Explanation of key design decisions:
1. Encoding issues and BOMs are handled immediately so that further processing sees consistent text.
2. Delimiter detection works by counting candidate symbols only outside quoted regions.
3. Header names are normalized (trimmed, lowercased, and non-alphanumeric characters replaced) so downstream applications have consistent keys.
4. Each row is adjusted if it contains too few or too many fields.
5. Each field is “cleaned” by stripping extraneous quotes/whitespace. Then we check for booleans, try date parsing (after removing ordinal suffixes), and number normalization (handling both US and European formats).
6. Finally, csv.writer produces a clean, comma‐delimited output with proper quoting.

This hybrid implementation brings together the strengths of all provided solutions and is robust, clear, and efficient.