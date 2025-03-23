#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

class o3_miniRound4Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return normalize_csv(input_text)

def normalize_csv(csv_data: str) -> str:
    """
    Clean and normalize messy CSV data.
    
    Issues addressed:
      • Ensures correct character encoding (using chardet if needed) and BOM removal
      • Normalizes line endings and splits only non-empty lines
      • Automatically detects delimiters (commas, semicolons, tabs, pipes or multi‐spaces)
      • Standardizes quote style to double quotes and converts all delimiters to commas
      • Uses csv.reader for robust parsing, then normalizes header names (to lowercase with underscores)
      • Adjusts row length when needed and cleans each field—trimming whitespace/quotes,
        converting null-like strings, booleans, dates (to YYYY-MM-DD) and numbers (US/European style)
      • Robust error handling for edge cases.
      
    Args:
        csv_data: String containing messy CSV data (or bytes)
        
    Returns:
        Clean, normalized CSV data as a string.
    """
    
    # 1. Handle encoding (if bytes) and remove BOM; normalize line endings.
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        encoding = detected.get('encoding') or 'utf-8'
        csv_data = csv_data.decode(encoding, errors='replace')
    csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Split into nonempty lines.
    lines = [line for line in csv_data.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # 3. Detect the likely delimiter.
    delimiter = detect_delimiter(lines)
    
    # 4. Standardize all lines:
    #    • Convert any common delimiter into a comma.
    #    • Standardize quotes to double quotes.
    standardized_lines = standardize_delimiters(lines, delimiter)
    
    # 5. Parse CSV rows using csv.reader for robust quote handling.
    rows = parse_csv_rows(standardized_lines, delimiter)
    if not rows or len(rows) < 1:
        return ""
    
    # 6. Normalize header names (lowercase, underscores, no extra quotes).
    header = [normalize_column_name(col) for col in rows[0]]
    
    # 7. Process and clean data rows.
    normalized_rows = [header]
    for row in rows[1:]:
        # Skip rows that are entirely empty.
        if not row or all(not cell.strip() for cell in row):
            continue
        
        # Adjust row length: pad if too short, or merge extra fields.
        row = adjust_row_length(row, len(header))
        if not row:
            continue
        
        # Clean each field based on its content and column hints.
        cleaned_row = []
        for i, field in enumerate(row):
            col_name = header[i] if i < len(header) else f"column_{i}"
            cleaned_row.append(clean_field(field, col_name))
        normalized_rows.append(cleaned_row)
    
    # 8. Write normalized rows into output CSV string.
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerows(normalized_rows)
    return output.getvalue()


def detect_delimiter(lines: List[str]) -> str:
    """
    Detect the most likely delimiter from candidate symbols (comma, semicolon, tab, pipe).
    We count delimiter characters that occur outside quoted sections in the first few lines.
    
    Returns:
        The delimiter character that scores best (or multi-space regex if none found).
    """
    candidates = [',', ';', '\t', '|']
    delim_counts: Dict[str, List[int]] = {d: [] for d in candidates}
    
    for line in lines[:min(10, len(lines))]:
        for delim in candidates:
            count = count_delimiters_outside_quotes(line, delim)
            if count > 0:
                delim_counts[delim].append(count)
    
    best_delim = ','
    best_score = 0
    for delim, counts in delim_counts.items():
        if not counts:
            continue
        ctr = Counter(counts)
        common_count, frequency = ctr.most_common(1)[0]
        score = frequency * common_count  # combination of consistency and frequency
        if score > best_score:
            best_score = score
            best_delim = delim

    # Special-case: if no clear delimiter found, try multiple spaces as delimiter.
    if best_score == 0:
        for line in lines[:min(5, len(lines))]:
            if re.search(r'\s{2,}', line):
                return r'\s+'
    
    return best_delim

def count_delimiters_outside_quotes(line: str, delim: str) -> int:
    """
    Returns the count of the given delimiter that occurs outside of quoted text.
    Supports both single and double quotes.
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
        elif char == delim and not in_quotes:
            count += 1
    return count

def standardize_delimiters(lines: List[str], primary_delimiter: str) -> List[str]:
    """
    Convert all common delimiters in each line into the primary delimiter (a comma),
    and standardize quote characters to double quotes.
    For a multi-space delimiter (r'\s+'), split on two or more spaces.
    """
    standardized = []
    
    for line in lines:
        # Special handling if we use a multi-space delimiter.
        if primary_delimiter == r'\s+' and not any(d in line for d in [',', ';', '\t', '|']):
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
                    new_line += '"'  # open with double quote
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                    new_line += '"'  # close with double quote
                else:
                    new_line += char
            elif char in [',', ';', '\t', '|'] and not in_quotes:
                new_line += ','  # use comma as standard delimiter
            else:
                new_line += char
        standardized.append(new_line)
    
    return standardized

def parse_csv_rows(lines: List[str], detected_delimiter: str) -> List[List[str]]:
    """
    Parse the standardized CSV lines into rows using csv.reader.
    If we are using a space-delimiter fallback, the delimiter is already applied.
    """
    actual_delim = ',' if detected_delimiter == r'\s+' else detected_delimiter
    csv_text = "\n".join(lines)
    try:
        reader = csv.reader(io.StringIO(csv_text), delimiter=actual_delim)
        # Strip whitespace from each field.
        return [[cell.strip() for cell in row] for row in reader]
    except Exception:
        # Fallback: simple splitting if csv.reader fails.
        rows = []
        for line in lines:
            row = line.split(actual_delim)
            rows.append([cell.strip() for cell in row])
        return rows

def normalize_column_name(colname: str) -> str:
    """
    Normalize header column name: trim quotes and whitespace,
    convert to lowercase and replace non-alphanumerics with underscores.
    """
    colname = colname.strip()
    if (colname.startswith('"') and colname.endswith('"')) or (colname.startswith("'") and colname.endswith("'")):
        colname = colname[1:-1].strip()
    
    # Replace non-alphanumeric (except underscore) with underscore.
    colname = re.sub(r'[^\w\s]', '_', colname)
    colname = re.sub(r'\s+', '_', colname)
    colname = re.sub(r'_+', '_', colname)
    return colname.strip('_').lower() or "column"

def adjust_row_length(row: List[str], expected: int) -> List[str]:
    """
    Adjust the row so that it has exactly expected number of fields.
      • If there are too few, pad with empty strings.
      • If there are too many, merge extra fields into the last column.
    """
    if len(row) == expected:
        return row
    if len(row) < expected:
        return row + [""] * (expected - len(row))
    
    # If too many fields, join extra fields into the final column.
    new_row = row[:expected - 1]
    combined = " ".join(row[expected - 1:])
    new_row.append(combined)
    return new_row

def clean_field(field: str, column_name: str) -> str:
    """
    Clean and normalize a single CSV field.
      • Trim whitespace and remove extraneous outer quotes.
      • Treat null-like values as empty.
      • Normalize boolean values.
      • If the field looks like a date (or the column name suggests it), convert it to ISO format.
      • If the field appears numeric, normalize number format.
    """
    field = field.strip()
    # Remove matching outer quotes
    if len(field) >= 2 and field[0] == field[-1] and field[0] in ['"', "'"]:
        field = field[1:-1].strip()
    
    # Handle missing or null-like
    if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
        return ""
    
    # Normalize boolean values
    low = field.lower()
    if low in ['true', 'yes', 'y', '1']:
        return "true"
    if low in ['false', 'no', 'n', '0']:
        return "false"
    
    # If the text looks like a date or the header indicates a date, convert to ISO.
    if looks_like_date(field) or is_date_column(column_name):
        dt = parse_date(field)
        if dt:
            return dt.strftime('%Y-%m-%d')
    
    # If the field looks numeric, try to format it.
    if looks_like_numeric(field):
        num = format_number(field)
        if num is not None:
            return num
    
    return field

def looks_like_date(field: str) -> bool:
    """
    Heuristic to determine if a field's content resembles a date.
    Checks several regex-based date patterns.
    """
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',           # 04/25/1991, 25-12-2023
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',               # 1991-04-25
        r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # May 3rd, 1992, March 12 1990
        r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'     # 3rd May 1992
    ]
    return any(re.search(p, field) for p in date_patterns)

def is_date_column(colname: str) -> bool:
    """
    Determine if the column name suggests it contains dates.
    """
    indicators = ['date', 'day', 'month', 'year', 'time', 'birth', 'updated', 'created']
    return any(ind in colname.lower() for ind in indicators)

def parse_date(date_str: str) -> Optional[date_parser]:
    """
    Attempt to parse a date string (fuzzy parsing after removing ordinal suffixes).
    Returns a datetime object if successful, else None.
    """
    # Remove ordinal suffixes (e.g. 3rd -> 3)
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
    try:
        return date_parser.parse(cleaned, fuzzy=True)
    except Exception:
        return None

def looks_like_numeric(field: str) -> bool:
    """
    Check if field appears to be numeric.
    Removes common currency symbols before checking.
    """
    cleaned = re.sub(r'[$€£\s]', '', field)
    return bool(re.search(r'\d', cleaned))

def format_number(num_str: str) -> Optional[str]:
    """
    Normalize a number string to standard format.
    Handles both US (1,234.56) and European (1.234,56) formats.
    Removes currency symbols and whitespace.
    """
    # Remove currency symbols and spaces.
    temp = re.sub(r'[$€£\s]', '', num_str)
    
    # If both comma and period present, decide which is the decimal marker.
    if ',' in temp and '.' in temp:
        if temp.rfind(',') > temp.rfind('.'):
            # Likely European: remove dots (thousands) and replace comma with period.
            temp = temp.replace('.', '')
            temp = temp.replace(',', '.')
        else:
            # Likely US: remove commas.
            temp = temp.replace(',', '')
    elif ',' in temp and '.' not in temp:
        # Ambiguous: if one comma and two digits after, treat as decimal.
        parts = temp.split(',')
        if len(parts) == 2 and len(parts[1]) in {1, 2}:
            temp = temp.replace(',', '.')
        else:
            temp = temp.replace(',', '')
    
    try:
        number = float(temp)
        # If number has no fractional part, return an integer.
        if number.is_integer():
            return str(int(number))
        else:
            s = f"{number:.6f}".rstrip('0').rstrip('.')
            return s
    except ValueError:
        return None