To create a robust `normalize_csv` function, we've combined the best elements from all provided solutions. This implementation efficiently handles encoding issues, delimiter detection, quote normalization, date and number parsing, and more. Let's explore the hybrid solution:

```python
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

def normalize_csv(csv_data: str) -> str:
    """
    Clean and normalize messy CSV data.
    
    Args:
        csv_data: String containing messy CSV data
        
    Returns:
        String containing cleaned, normalized CSV data
    """
    # Handle encoding issues
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        csv_data = csv_data.decode(detected['encoding'] or 'utf-8', errors='replace')
    
    # Remove BOM and normalize line endings
    csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    
    # Split lines and remove empty lines
    lines = [line for line in csv_data.split('\n') if line.strip()]
    if not lines:
        return ""
    
    # Detect and standardize delimiters
    delimiter = detect_delimiter(lines)
    standardized_lines = standardize_delimiters(lines, delimiter)
    
    # Parse CSV with detected delimiter
    rows = parse_csv_rows(standardized_lines)
    if not rows:
        return ""
    
    # Normalize the header
    header = [normalize_column_name(col) for col in rows[0]]
    
    # Process data rows
    normalized_rows = [header]
    for row in rows[1:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        adjusted_row = adjust_row_length(row, len(header))
        normalized_row = [clean_field(cell, header[i] if i < len(header) else f"column_{i}") for i, cell in enumerate(adjusted_row)]
        normalized_rows.append(normalized_row)
    
    # Write to CSV
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerows(normalized_rows)
    
    return output.getvalue()

def detect_delimiter(lines: List[str]) -> str:
    common_delimiters = [',', ';', '\t', '|']
    delimiter_counts = {d: [] for d in common_delimiters}
    
    for line in lines[:min(10, len(lines))]:
        for delim in common_delimiters:
            count = count_delimiters_outside_quotes(line, delim)
            delimiter_counts[delim].append(count)
    
    best_delimiter = max(delimiter_counts, key=lambda d: Counter(delimiter_counts[d]).most_common(1)[0][1])
    return best_delimiter

def count_delimiters_outside_quotes(line: str, delimiter: str) -> int:
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

def standardize_delimiters(lines: List[str], primary_delimiter: str) -> List[str]:
    standardized_lines = []
    
    for line in lines:
        new_line = ""
        in_quotes = False
        quote_char = None
        
        for char in line:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                    new_line += '"'
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                    new_line += '"'
                else:
                    new_line += char
            elif char in [',', ';', '\t', '|'] and not in_quotes:
                new_line += ','
            else:
                new_line += char
                
        standardized_lines.append(new_line)
    
    return standardized_lines

def parse_csv_rows(lines: List[str]) -> List[List[str]]:
    csv_text = '\n'.join(lines)
    reader = csv.reader(io.StringIO(csv_text), delimiter=',')
    return [row for row in reader]

def normalize_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r'[^\w\s]', '_', column)
    column = re.sub(r'\s+', '_', column)
    return column.strip('_')

def adjust_row_length(row: List[str], expected_length: int) -> List[str]:
    if len(row) < expected_length:
        return row + [""] * (expected_length - len(row))
    elif len(row) > expected_length:
        return row[:expected_length]
    return row

def clean_field(field: str, column_name: str) -> str:
    field = field.strip()
    if len(field) >= 2 and ((field.startswith('"') and field.endswith('"')) or (field.startswith("'") and field.endswith("'"))):
        field = field[1:-1].strip()
    
    if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
        return ""
    
    if field.lower() in ['true', 'yes', 'y', '1']:
        return "true"
    if field.lower() in ['false', 'no', 'n', '0']:
        return "false"
    
    if looks_like_date(field) or is_date_column(column_name):
        date_obj = parse_date(field)
        if date_obj:
            return date_obj.strftime('%Y-%m-%d')
    
    if looks_like_numeric(field):
        return format_number(field)
    
    return field

def looks_like_date(field: str) -> bool:
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',
        r'\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{2,4}'
    ]
    return any(re.search(pattern, field) for pattern in date_patterns)

def is_date_column(column_name: str) -> bool:
    date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
    return any(indicator in column_name.lower() for indicator in date_indicators)

def parse_date(date_str: str) -> Optional[date_parser.parser]:
    try:
        return date_parser.parse(date_str, fuzzy=True)
    except (date_parser.ParserError, ValueError):
        return None

def looks_like_numeric(field: str) -> bool:
    cleaned = re.sub(r'[^\d.,-]', '', field)
    return bool(re.match(r'^[+-]?\d+([.,]\d+)*$', cleaned))

def format_number(number_str: str) -> str:
    number_str = re.sub(r'[^\d.,-]', '', number_str)
    if ',' in number_str and '.' in number_str:
        if number_str.rindex(',') > number_str.rindex('.'):
            number_str = number_str.replace('.', '').replace(',', '.')
        else:
            number_str = number_str.replace(',', '')
    elif ',' in number_str:
        number_str = number_str.replace(',', '.')
    try:
        num = float(number_str)
        return f"{num:.2f}".rstrip('0').rstrip('.') if '.' in str(num) else str(int(num))
    except ValueError:
        return number_str

# Example usage
if __name__ == "__main__":
    csv_example = '''
Name, Age, Birth Date, Salary
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
'''
    print(normalize_csv(csv_example))
```

### Key Features:
1. **Encoding Handling**: Utilizes `chardet` to detect and handle encoding issues, ensuring the input is correctly interpreted.
2. **Delimiter Detection**: Considers common delimiters and counts occurrences outside quotes to determine the most consistent delimiter.
3. **Quote and Delimiter Standardization**: Normalizes quotes to double quotes and replaces all delimiters with commas for consistency.
4. **Header Normalization**: Converts headers to lowercase with underscores to ensure uniformity.
5. **Field Cleaning**: Handles booleans, dates, numbers, and null values, ensuring fields are correctly formatted.
6. **Row Length Adjustment**: Adjusts row lengths to match headers, padding or truncating as necessary.
7. **Output**: Uses `csv.writer` to output consistently formatted CSV data.

This implementation combines the best features from all provided solutions, offering a comprehensive approach to CSV normalization.