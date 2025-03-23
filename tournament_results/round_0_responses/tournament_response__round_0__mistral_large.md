To tackle the problem of normalizing and cleaning messy CSV data, we need to address several key issues systematically. Here's a step-by-step breakdown of the approach:

### Step-by-Step Approach

1. **Detect Delimiters**:
   - Identify the most common delimiter used in the CSV data. This can be done by counting the occurrences of common delimiters (commas, semicolons, tabs, pipes) in the first few rows.

2. **Handle Quote Styles**:
   - Normalize different quote styles (double quotes, single quotes, no quotes) to a consistent style (double quotes).

3. **Normalize Date Formats**:
   - Convert various date formats to the ISO format (YYYY-MM-DD). This involves parsing dates using multiple potential formats and then reformatting them.

4. **Standardize Number Formats**:
   - Convert numbers with various thousand and decimal separators to a standard format (e.g., 1,000.00 or 1.000,00 to 1000.00).

5. **Remove Empty Rows**:
   - Identify and remove completely empty rows.

6. **Trim Whitespace**:
   - Trim unnecessary whitespace from field values.

7. **Normalize Column Names**:
   - Convert column names to lowercase and replace spaces with underscores.

8. **Handle Missing Values**:
   - Replace missing values with appropriate NULL values or empty strings.

9. **Fix Character Encoding Issues**:
   - Handle common encoding problems by ensuring the data is in a consistent encoding format (e.g., UTF-8).

### Key Design Decisions

1. **Delimiter Detection**:
   - Use a heuristic approach to detect the most common delimiter by analyzing the first few rows.

2. **Quote Normalization**:
   - Use regular expressions to normalize quotes and handle escaped quotes within fields.

3. **Date Parsing**:
   - Use the `dateutil` library to parse dates flexibly and convert them to ISO format.

4. **Number Parsing**:
   - Use regular expressions to identify and standardize number formats.

5. **Whitespace Trimming**:
   - Use string methods to trim whitespace from field values.

6. **Column Name Normalization**:
   - Use string methods to convert column names to a consistent format.

7. **Missing Values**:
   - Replace missing values with empty strings for consistency.

8. **Character Encoding**:
   - Ensure the data is decoded and encoded consistently using UTF-8.

### Implementation

Here's the implementation of the `normalize_csv` function:

```python
import csv
import re
from io import StringIO
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
    # Detect delimiter
    delimiter = detect_delimiter(csv_data)

    # Normalize quotes and handle escaped quotes
    csv_data = re.sub(r"(?<!\\)'", '"', csv_data)
    csv_data = re.sub(r'(?<!\\)"', '"', csv_data)

    # Read CSV data
    reader = csv.reader(StringIO(csv_data), delimiter=delimiter)
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
    output = StringIO()
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

1. **Detect Delimiter**:
   - The `detect_delimiter` function analyzes the first few rows to determine the most common delimiter.

2. **Normalize Quotes**:
   - The `normalize_quotes` function ensures that all fields are consistently quoted with double quotes.

3. **Parse Dates and Numbers**:
   - The `parse_date` function uses the `dateutil` library to parse and reformat dates.
   - The `parse_number` function uses regular expressions to standardize number formats.

4. **Read and Write CSV**:
   - The `csv.reader` and `csv.writer` classes are used to read and write the CSV data, ensuring proper escaping and quoting.

5. **Handle Missing Values and Whitespace**:
   - Empty fields are handled by checking if the field is empty after stripping whitespace.

This implementation should handle the common issues in messy CSV data and produce a clean, normalized CSV output.