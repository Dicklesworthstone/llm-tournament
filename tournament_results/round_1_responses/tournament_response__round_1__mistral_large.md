To create a robust, efficient, and clean implementation of the `normalize_csv` function, we'll combine the best elements from each of the provided solutions. Let's break down the key steps and design decisions:

### Key Steps and Design Decisions

1. **Character Encoding Handling**:
   - Use `chardet` to detect and handle different character encodings (from claude37).

2. **Delimiter Detection**:
   - Analyze the first few rows to determine the most common delimiter (from gpt4o and claude37).

3. **Quote Normalization**:
   - Normalize different quote styles to a consistent style (double quotes) and handle escaped quotes within fields (from mistral_large).

4. **Date Parsing**:
   - Use the `dateutil` library to parse dates flexibly and convert them to ISO format (from gpt4o and o3_mini).

5. **Number Parsing**:
   - Use regular expressions to identify and standardize number formats, handling both US and European conventions (from claude37).

6. **Whitespace Trimming**:
   - Trim unnecessary whitespace from field values (from all solutions).

7. **Column Name Normalization**:
   - Convert column names to lowercase and replace spaces with underscores (from all solutions).

8. **Missing Values**:
   - Replace missing values with empty strings for consistency (from all solutions).

9. **Character Encoding**:
   - Ensure the data is decoded and encoded consistently using UTF-8 (from claude37).

10. **Output Formatting**:
    - Use `csv.writer` to ensure consistent CSV output with properly escaped fields (from all solutions).

### Implementation

Here's the combined implementation of the `normalize_csv` function:

```python
import csv
import re
import io
import chardet
from dateutil import parser
from collections import Counter

def detect_delimiter(csv_data):
    lines = csv_data.splitlines()
    delimiter_candidates = [',', ';', '\t', '|']
    delimiter_counts = Counter()
    for line in lines[:5]:  # Check the first 5 lines
        for delimiter in delimiter_candidates:
            delimiter_counts[delimiter] += line.count(delimiter)
    return delimiter_counts.most_common(1)[0][0]

def normalize_quotes(field):
    if field.startswith(("'", '"')) and field.endswith(("'", '"')):
        return f'"{field[1:-1]}"'
    return f'"{field}"'

def parse_date(date_str):
    try:
        return parser.parse(date_str).strftime('%Y-%m-%d')
    except (ValueError, OverflowError):
        return date_str

def parse_number(number_str):
    number_str = re.sub(r'[^\d.,-]', '', number_str)
    if ',' in number_str and '.' in number_str:
        if number_str.rindex(',') > number_str.rindex('.'):
            number_str = number_str.replace('.', '').replace(',', '.')
        else:
            number_str = number_str.replace(',', '')
    return number_str

def normalize_csv(csv_data: str) -> str:
    # Handle character encoding issues
    if isinstance(csv_data, bytes):
        detected = chardet.detect(csv_data)
        csv_data = csv_data.decode(detected['encoding'] or 'utf-8', errors='replace')

    # Remove BOM if present
    csv_data = csv_data.lstrip('\ufeff')

    # Normalize line endings
    csv_data = csv_data.replace('\r\n', '\n').replace('\r', '\n')

    # Detect delimiter
    delimiter = detect_delimiter(csv_data)

    # Normalize quotes and handle escaped quotes
    csv_data = re.sub(r"(?<!\\)'", '"', csv_data)
    csv_data = re.sub(r'(?<!\\)"', '"', csv_data)

    # Read CSV data
    reader = csv.reader(io.StringIO(csv_data), delimiter=delimiter)
    rows = list(reader)

    # Normalize column names
    header = [re.sub(r'\s+', '_', col.lower().strip()) for col in rows[0]]

    # Process rows
    normalized_rows = []
    for row in rows[1:]:
        if all(field.strip() == '' for field in row):
            continue
        normalized_row = []
        for field in row:
            field = field.strip()
            if field == '':
                normalized_row.append('')
            else:
                field = normalize_quotes(field)
                if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', field):
                    field = parse_date(field)
                elif re.match(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', field):
                    field = parse_number(field)
                normalized_row.append(field)
        normalized_rows.append(normalized_row)

    # Write normalized CSV data
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    writer.writerows(normalized_rows)

    return output.getvalue()

# Example usage
messy_csv = '''Name, Age, Birth Date, Salary
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
'''
cleaned_csv = normalize_csv(messy_csv)
print(cleaned_csv)
```

### Explanation of the Implementation

1. **Character Encoding Handling**:
   - The function first detects and handles different character encodings using `chardet`.

2. **Delimiter Detection**:
   - The `detect_delimiter` function analyzes the first few rows to determine the most common delimiter.

3. **Quote Normalization**:
   - The `normalize_quotes` function ensures that all fields are consistently quoted with double quotes.

4. **Date Parsing**:
   - The `parse_date` function uses the `dateutil` library to parse and reformat dates.

5. **Number Parsing**:
   - The `parse_number` function uses regular expressions to standardize number formats, handling both US and European conventions.

6. **Read and Write CSV**:
   - The `csv.reader` and `csv.writer` classes are used to read and write the CSV data, ensuring proper escaping and quoting.

7. **Handle Missing Values and Whitespace**:
   - Empty fields are handled by checking if the field is empty after stripping whitespace.

This implementation combines the best aspects of all provided solutions, ensuring correctness, efficiency, readability, and robustness.