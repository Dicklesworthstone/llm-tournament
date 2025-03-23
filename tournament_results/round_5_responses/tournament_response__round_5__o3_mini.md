Below is the complete final implementation. I designed this solution by reviewing ideas from multiple submissions. In particular, I combined:

• Robust encoding detection (with BOM removal and normalization)  
• Delimiter detection by counting separators outside quotes (with a fallback to multiple‐spaces)  
• Standardization of delimiters (all replaced with commas) and quote normalization (all to double quotes)  
• Parsing via csv.reader with a manual fallback if necessary  
• Header “normalization” (lowercase with underscores)  
• Field cleaning that trims whitespace/quotes, converts “null” strings to empty values, normalizes booleans, reformats dates to ISO (YYYY‐MM‐DD) using dateutil’s fuzzy parser, and converts numbers (handling US and European formats)  
• Adjustment of row lengths by padding or intelligently combining extra fields  

The code also includes extensive inline comments describing the design decisions. You can use the code as is without any modification.

-------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union, Any

def normalize_csv(csv_data: Union[str, bytes]) -> str:
    """
    Clean and normalize messy CSV data.

    The function performs the following:
      1. Handles character encoding (detect bytes and BOM, normalizes line-endings).
      2. Splits the input into non-empty lines.
      3. Detects the delimiter by counting occurrences outside quotes.
      4. Standardizes lines so that all delimiters become commas and quotes become double quotes.
      5. Parses the CSV data, normalizes header names (lowercased, underscores), and adjusts row lengths.
      6. Cleans each field by trimming extra whitespace/quotes, converting null/boolean values,
         normalizing dates to ISO format, and converting numeric formats.
      7. Reconstructs the data using csv.writer to guarantee proper escaping.

    Args:
        csv_data: CSV data as a string or bytes.

    Returns:
        Clean, normalized CSV data as a string.
    """
    # 1. Handle encoding (works even if csv_data is bytes) and remove BOM.
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        encoding = detected.get('encoding') or 'utf-8'
        csv_data = csv_data.decode(encoding, errors='replace')
    csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Split into non-empty lines.
    lines = [line for line in csv_data.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # 3. Detect delimiter.
    delimiter = detect_delimiter(lines)
    
    # 4. Standardize the delimiters and quote characters.
    standardized_lines = standardize_delimiters(lines, delimiter)
    
    # 5. Parse CSV rows.
    rows = parse_csv_rows(standardized_lines, delimiter)
    if not rows or len(rows) < 1:
        return ""
    
    # 6. Normalize header row.
    header = [normalize_column_name(col) for col in rows[0]]
    
    # 7. Process and clean each row.
    normalized_rows = [header]
    for row in rows[1:]:
        if not row or all(not cell.strip() for cell in row):
            continue  # skip completely blank rows
        
        adjusted = adjust_row_length(row, len(header))
        if not adjusted:
            continue
        
        cleaned = [clean_field(adjusted[i], header[i] if i < len(header) else f"column_{i}") 
                      for i in range(len(adjusted))]
        normalized_rows.append(cleaned)
    
    # 8. Write back to CSV string.
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerows(normalized_rows)
    
    return output.getvalue()

def detect_delimiter(lines: List[str]) -> str:
    """
    Analyze the first few lines and count candidate delimiters outside quotes.
    Returns the most consistent delimiter; if none is found, fallback to multiple spaces.

    Args:
        lines: List of input CSV lines.
        
    Returns:
        A delimiter character or pattern (r'\s+') if multiple spaces seem to be used.
    """
    common_delimiters = [',', ';', '\t', '|']
    delim_counts: Dict[str, List[int]] = {d: [] for d in common_delimiters}
    
    for line in lines[:min(10, len(lines))]:
        for delim in common_delimiters:
            cnt = count_delimiters_outside_quotes(line, delim)
            if cnt > 0:
                delim_counts[delim].append(cnt)
    
    best_delimiter = ','
    best_score = 0
    for delim, counts in delim_counts.items():
        if not counts:
            continue
        freq = Counter(counts).most_common(1)[0]  # (most_common_count, frequency)
        count_val, frequency = freq
        score = count_val * frequency  # combination of count and consistency
        if score > best_score:
            best_score = score
            best_delimiter = delim

    # Fallback if no traditional delimiter is found: check for multiple spaces.
    if best_score == 0:
        for line in lines[:min(5, len(lines))]:
            if re.search(r'\s{2,}', line):
                return r'\s+'
    return best_delimiter

def count_delimiters_outside_quotes(line: str, delimiter: str) -> int:
    """
    Count the occurrences of a delimiter outside quoted parts.
    
    Args:
        line: A single CSV line.
        delimiter: The delimiter candidate.
    
    Returns:
        The count of delimiter occurrences outside any quotes.
    """
    count = 0
    in_quotes = False
    quote_char = None
    escaped = False
    
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
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

def standardize_delimiters(lines: List[str], primary_delimiter: str) -> List[str]:
    """
    Convert all delimiters in every line to comma and turn any quotes into standard double quotes.
    
    If the detected delimiter is a multi-space pattern (r'\s+'),
    then split on two or more spaces.
    
    Args:
        lines: List of CSV line strings.
        primary_delimiter: The detected delimiter from detect_delimiter().
    
    Returns:
        List of standardized CSV line strings.
    """
    standardized = []
    for line in lines:
        # When using multi-space-delimiter:
        if primary_delimiter == r'\s+' and not any(d in line for d in [',', ';', '\t', '|']):
            # Split when two or more spaces occur.
            fields = re.split(r'\s{2,}', line.strip())
            standardized.append(",".join(fields))
            continue
        
        new_line = ""
        in_quotes = False
        quote_char = None
        for char in line:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                    new_line += '"'  # start double quote
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                    new_line += '"'  # close with double quote
                else:
                    new_line += char
            elif char in [',', ';', '\t', '|'] and not in_quotes:
                new_line += ','  # force comma delimiter
            else:
                new_line += char
        standardized.append(new_line)
    return standardized

def parse_csv_rows(lines: List[str], detected_delimiter: str) -> List[List[str]]:
    """
    Parse CSV rows using Python's csv.reader. If the delimiter is r'\s+', treat the data as already separated.
    
    Args:
        lines: List of standardized CSV lines.
        detected_delimiter: The delimiter detected.
    
    Returns:
        List of rows, where each row is a list of cell values.
    """
    csv_text = "\n".join(lines)
    actual_delim = ',' if detected_delimiter == r'\s+' else detected_delimiter
    try:
        reader = csv.reader(io.StringIO(csv_text), delimiter=actual_delim)
        # Also strip extra whitespace from each cell.
        return [[cell.strip() for cell in row] for row in reader]
    except Exception:
        # Fallback: manually split each line.
        rows = []
        for line in lines:
            row = [cell.strip() for cell in line.split(actual_delim)]
            rows.append(row)
        return rows

def normalize_column_name(column: str) -> str:
    """
    Normalize header names: remove extra quotes/whitespace, convert to lowercase,
    and replace non-alphanumerics with underscores.
    
    Args:
        column: The original column name.
    
    Returns:
        Normalized column name.
    """
    column = column.strip()
    if (column.startswith('"') and column.endswith('"')) or (column.startswith("'") and column.endswith("'")):
        column = column[1:-1].strip()
    normalized = re.sub(r'[^\w\s]', '_', column)
    normalized = re.sub(r'\s+', '_', normalized).lower()
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    return normalized or "column"

def adjust_row_length(row: List[str], expected_length: int) -> Optional[List[str]]:
    """
    Ensure the row has exactly expected_length fields:
      - If too few, pad with empty strings.
      - If too many, try to combine fields (e.g. if quoted text was split).
    
    Args:
        row: The parsed row.
        expected_length: The number of header columns.
    
    Returns:
        A row list with exactly expected_length elements.
    """
    if len(row) == expected_length:
        return row
    if len(row) < expected_length:
        return row + [""] * (expected_length - len(row))
    
    # For too many fields, try to combine extra fields into the last column.
    new_row = row[:expected_length-1]
    combined = " ".join(row[expected_length-1:])
    new_row.append(combined)
    if len(new_row) == expected_length:
        return new_row
    return new_row[:expected_length]

def clean_field(field: str, column_name: str) -> str:
    """
    Clean a field by:
      - Removing extra whitespace and outer quotes.
      - Converting common null (or missing) values to empty strings.
      - Normalizing booleans.
      - Converting dates (if the text looks like a date or the column indicates a date) to ISO format.
      - Converting numeric values (US or European style) to a standardized format.
    
    Args:
        field: The cell value to clean.
        column_name: The name of its column (for type hints).
    
    Returns:
        The cleaned field value.
    """
    field = field.strip()
    # If field is wrapped in quotes, remove them.
    if len(field) >= 2 and field[0] == field[-1] and field[0] in ['"', "'"]:
        field = field[1:-1].strip()
    
    # Empty and null-like values.
    if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
        return ""
    
    # Normalize booleans.
    low_field = field.lower()
    if low_field in ['true', 'yes', 'y', '1']:
        return "true"
    if low_field in ['false', 'no', 'n', '0']:
        return "false"
    
    # If the field looks like a date or the header indicates a date, try to parse it.
    if is_date_column(column_name) or looks_like_date(field):
        dt = parse_date(field)
        if dt:
            return dt.strftime('%Y-%m-%d')
    
    # If the field looks numeric, try to format it.
    if looks_like_numeric(field):
        num = format_number(field)
        if num is not None:
            return num
    
    return field

def is_date_column(column_name: str) -> bool:
    """
    Heuristic: if the column name contains words suggestive of dates, return True.
    
    Args:
        column_name: Name of the column.
        
    Returns:
        True if the column is likely a date.
    """
    keywords = ['date', 'day', 'month', 'year', 'time', 'birth', 'updated', 'created']
    return any(kw in column_name.lower() for kw in keywords)

def looks_like_date(field: str) -> bool:
    """
    Use regular expressions to guess if a field appears to contain a date.
    
    Args:
        field: The cell content.
    
    Returns:
        True if it matches typical date patterns.
    """
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',       # e.g. 04/25/1991 or 25-12-2023
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',          # e.g. 1991-04-25
        r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # e.g. May 3rd, 1992
        r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'     # e.g. 3rd May 1992
    ]
    return any(re.search(p, field) for p in patterns)

def parse_date(date_str: str) -> Optional[Any]:
    """
    Parse a date string using dateutil's parser (with fuzzy matching).
    Also removes ordinal suffixes (e.g., "3rd" becomes "3").
    
    Args:
        date_str: A date string.
    
    Returns:
        A datetime object if successful or None.
    """
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    try:
        return date_parser.parse(cleaned, fuzzy=True)
    except Exception:
        return None

def looks_like_numeric(field: str) -> bool:
    """
    Determine if a field appears to have a numeric value.
    Currency symbols and spaces are removed first.
    
    Args:
        field: The cell content.
    
    Returns:
        True if the field contains digits and allowed numeric punctuation.
    """
    cleaned = re.sub(r'[$€£\s]', '', field)
    return bool(re.match(r'^[+-]?\d+([.,]\d+)*$', cleaned))

def format_number(num_str: str) -> Optional[str]:
    """
    Normalize numbers with both US and European formats.
    
    Steps:
      - Remove currency symbols and spaces.
      - If both comma and period exist, decide which one is the decimal separator.
      - Convert the cleaned string to float and then back to a string (dropping unnecessary zeros).
    
    Args:
        num_str: The numeric string.
        
    Returns:
        The normalized number string or None on failure.
    """
    cleaned = re.sub(r'[$€£\s]', '', num_str)
    # If both delimiters exist, decide based on which comes last.
    if ',' in cleaned and '.' in cleaned:
        if cleaned.rfind(',') > cleaned.rfind('.'):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned and '.' not in cleaned:
        # If one comma is followed by exactly 2 digits, assume decimal; otherwise remove comma.
        if re.search(r',\d{1,2}$', cleaned):
            cleaned = cleaned.replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    try:
        number = float(cleaned)
        if number.is_integer():
            return str(int(number))
        else:
            # Format to remove trailing zeros while keeping up to 6 decimals.
            formatted = f"{number:.6f}".rstrip('0').rstrip('.')
            return formatted
    except ValueError:
        return None

# ----------------------
# Example usage (for testing):
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
    
# End of implementation

-------------------------

Explanation of Key Elements:
1. Encoding and BOM removal ensure we work with clean Unicode text.
2. Delimiter detection looks at the first few nonempty lines counting only those characters outside of quotes.
3. Standardization homogenizes the data (all delimiters become commas and quotes are normalized).
4. The header is “normalized” by removing extraneous punctuation and lowercasing with underscores.
5. The row length is adjusted to the header’s length (padding or merging extra fields as needed).
6. Each field is then “cleaned” to trim quotes/whitespace, convert null/boolean values, reformat dates (with dateutil’s parser), and normalize number formats to a standard form.
7. Finally, csv.writer is used to generate the output ensuring proper escaping.

This solution is robust, efficient, maintainable, and handles a wide variety of messy CSV inputs.