Here's a hybrid implementation of the `normalize_csv` function. It combines the best aspects of the solutions provided, ensuring correctness, efficiency, readability, and robustness. This implementation addresses all the specified issues with CSV data and is designed to handle a variety of edge cases gracefully.

```python
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union

def normalize_csv(csv_data: Union[str, bytes]) -> str:
    """
    Clean and normalize messy CSV data.
    
    Args:
        csv_data: String or bytes containing messy CSV data
        
    Returns:
        String containing cleaned, normalized CSV data
    """
    # Handle encoding issues
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        csv_data = csv_data.decode(detected['encoding'] or 'utf-8', errors='replace')
    
    # Remove BOM and normalize line endings
    csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into non-empty lines
    lines = [line for line in csv_data.split('\n') if line.strip()]
    if not lines:
        return ""

    # Detect and standardize delimiters
    delimiter = detect_delimiter(lines)
    standardized_lines = standardize_delimiters(lines, delimiter)
    
    # Parse CSV rows
    rows = parse_csv_rows(standardized_lines)
    if not rows or len(rows) < 1:
        return ""
    
    # Normalize header row
    header = [normalize_column_name(col) for col in rows[0]]
    
    # Process data rows
    normalized_rows = [header]
    for row in rows[1:]:
        # Skip completely empty rows
        if not row or all(not cell.strip() for cell in row):
            continue
        
        # Adjust row length to match header
        adjusted_row = adjust_row_length(row, len(header))
        if not adjusted_row:
            continue
        
        # Clean each field based on content and column name
        cleaned_row = [
            clean_field(field, header[i] if i < len(header) else f"column_{i}")
            for i, field in enumerate(adjusted_row)
        ]
        
        normalized_rows.append(cleaned_row)
    
    # Write normalized data back to CSV
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerows(normalized_rows)
    
    return output.getvalue()

def detect_delimiter(lines: List[str]) -> str:
    """
    Detect the most likely delimiter by analyzing patterns outside quoted sections.
    
    Args:
        lines: List of CSV data lines
        
    Returns:
        The most likely delimiter character
    """
    common_delimiters = [',', ';', '\t', '|']
    delimiter_counts: Dict[str, List[int]] = {d: [] for d in common_delimiters}
    
    # Analyze first few lines
    for line in lines[:min(10, len(lines))]:
        for delimiter in common_delimiters:
            count = count_delimiters_outside_quotes(line, delimiter)
            if count > 0:
                delimiter_counts[delimiter].append(count)
    
    # Find most consistent delimiter
    best_delimiter = ','  # Default to comma
    best_score = 0
    
    for delimiter, counts in delimiter_counts.items():
        if not counts:
            continue
        
        # Score based on consistency and frequency
        counter = Counter(counts)
        most_common, frequency = counter.most_common(1)[0]
        score = frequency * most_common
        
        if score > best_score:
            best_score = score
            best_delimiter = delimiter
    
    # Special case: check for multiple spaces as delimiter
    if best_score == 0:
        for line in lines[:min(5, len(lines))]:
            if re.search(r'\s{2,}', line):
                return r'\s+'
    
    return best_delimiter

def count_delimiters_outside_quotes(line: str, delimiter: str) -> int:
    """Count delimiters that appear outside of quoted sections."""
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
    """
    Standardize all lines to use comma as delimiter and double quotes for quoting.
    
    Args:
        lines: List of CSV data lines
        primary_delimiter: The detected delimiter
        
    Returns:
        List of standardized CSV lines
    """
    standardized_lines = []
    
    for line in lines:
        # Special handling for space-delimited data
        if primary_delimiter == r'\s+' and not any(d in line for d in [',', ';', '\t', '|']):
            fields = re.split(r'\s{2,}', line.strip())
            standardized_lines.append(','.join(fields))
            continue
            
        new_line = ""
        in_quotes = False
        quote_char = None
        
        for char in line:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                    new_line += '"'  # Standardize to double quotes
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                    new_line += '"'  # Standardize to double quotes
                else:
                    new_line += char
            elif char in [',', ';', '\t', '|'] and not in_quotes:
                new_line += ','  # Standardize to comma
            else:
                new_line += char
        
        standardized_lines.append(new_line)
    
    return standardized_lines

def parse_csv_rows(lines: List[str]) -> List[List[str]]:
    """
    Parse the standardized CSV lines into rows using csv.reader.
    
    Args:
        lines: List of standardized CSV data lines
        
    Returns:
        List of parsed CSV rows
    """
    csv_text = '\n'.join(lines)
    reader = csv.reader(io.StringIO(csv_text), delimiter=',')
    return [row for row in reader]

def normalize_column_name(name: str) -> str:
    """
    Normalize column name to lowercase with underscores.
    
    Args:
        name: Column name to normalize
        
    Returns:
        Normalized column name
    """
    # Remove quotes and extra whitespace
    name = name.strip()
    if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
        name = name[1:-1].strip()
    
    # Convert to lowercase and replace non-alphanumeric with underscores
    name = re.sub(r'[^\w\s]', '_', name.lower())
    name = re.sub(r'\s+', '_', name)
    
    # Remove consecutive and trailing underscores
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    return name if name else "column"  # Ensure we don't return empty string

def adjust_row_length(row: List[str], expected_length: int) -> List[str]:
    """
    Adjust row to match expected length by padding or combining fields.
    
    Args:
        row: Row to adjust
        expected_length: Expected number of fields
        
    Returns:
        Adjusted row
    """
    if len(row) == expected_length:
        return row
    
    # If row is too short, pad with empty strings
    if len(row) < expected_length:
        return row + [""] * (expected_length - len(row))
    
    # If row is too long, combine extra fields
    new_row = row[:expected_length - 1]
    combined = " ".join(row[expected_length - 1:])
    new_row.append(combined)
    return new_row

def clean_field(field: str, column_name: str) -> str:
    """
    Clean and normalize a single CSV field.
    
    Args:
        field: Field value to clean
        column_name: Name of the column (for type inference)
        
    Returns:
        Cleaned field value
    """
    # Remove outer quotes and trim whitespace
    field = field.strip()
    if len(field) >= 2 and ((field.startswith('"') and field.endswith('"')) or (field.startswith("'") and field.endswith("'"))):
        field = field[1:-1].strip()
    
    # Handle empty or null-like values
    if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
        return ""
    
    # Normalize boolean values
    if field.lower() in ['true', 'yes', 'y', '1']:
        return "true"
    if field.lower() in ['false', 'no', 'n', '0']:
        return "false"
    
    # Try to detect and normalize date fields
    if is_date_column(column_name) or looks_like_date(field):
        date_obj = parse_date(field)
        if date_obj:
            return date_obj.strftime('%Y-%m-%d')
    
    # Try to detect and normalize numeric fields
    if looks_like_numeric(field):
        normalized_number = format_number(field)
        if normalized_number is not None:
            return normalized_number
    
    return field

def is_date_column(column_name: str) -> bool:
    """Check if column name suggests it contains dates."""
    date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
    return any(indicator in column_name.lower() for indicator in date_indicators)

def looks_like_date(field: str) -> bool:
    """Check if field value looks like a date."""
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
        r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # Month DD, YYYY
        r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
    ]
    
    return any(re.search(pattern, field) for pattern in date_patterns)

def parse_date(date_str: str) -> Optional[date_parser.parser]:
    """Attempt to parse a date string to a datetime object."""
    try:
        # Remove ordinal suffixes (e.g., 1st, 2nd, 3rd)
        cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        return date_parser.parse(cleaned, fuzzy=True)
    except (date_parser.ParserError, ValueError):
        return None

def looks_like_numeric(field: str) -> bool:
    """Check if field appears to be numeric."""
    cleaned = re.sub(r'[$€£\s]', '', field)
    return bool(re.search(r'\d', cleaned))

def format_number(number_str: str) -> Optional[str]:
    """
    Normalize a number string to standard format.
    
    Args:
        number_str: Number string to normalize
        
    Returns:
        Normalized number string or None if parsing fails
    """
    # Remove currency symbols and spaces
    number_str = re.sub(r'[$€£\s]', '', number_str)
    
    # Handle different number formats
    if ',' in number_str and '.' in number_str:
        if number_str.rindex(',') > number_str.rindex('.'):
            # European format: remove dots and replace last comma with dot
            number_str = number_str.replace('.', '').replace(',', '.')
        else:
            # US format: remove commas
            number_str = number_str.replace(',', '')
    elif ',' in number_str:
        # If one comma and two digits after, treat as decimal
        parts = number_str.split(',')
        if len(parts) == 2 and len(parts[1]) in {1, 2}:
            number_str = number_str.replace(',', '.')
        else:
            number_str = number_str.replace(',', '')
    
    try:
        num = float(number_str)
        return f"{num:.2f}".rstrip('0').rstrip('.') if '.' in str(num) else str(int(num))
    except ValueError:
        return number_str
```

### Key Features:
- **Encoding Handling**: Detects and decodes the input correctly, removing BOM if present.
- **Delimiter Detection**: Automatically identifies the most consistent delimiter using analysis outside of quoted sections.
- **Quote Standardization**: Converts all quote characters to double quotes for consistency.
- **Header Normalization**: Cleans and formats headers to lowercase with underscores.
- **Field Cleaning**: Handles null-like values, boolean normalization, date parsing to ISO format, and number formatting for both US and European styles.
- **Row Length Adjustment**: Pads or combines fields to ensure row consistency with headers.
- **Robust Error Handling**: Gracefully handles errors during date and number parsing.

This implementation should work efficiently and correctly for a wide variety of messy CSV inputs, providing a clean and normalized output.