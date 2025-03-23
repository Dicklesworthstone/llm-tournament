To tackle the CSV normalization and cleaning challenge, we need a systematic approach to handle various issues in messy CSV data. Here's a detailed explanation of the steps and considerations involved in implementing the `normalize_csv` function:

### Key Steps and Considerations

1. **Detecting Delimiters**: 
   - We need to determine the delimiter used in each line. This can be done by checking for common delimiters (e.g., commas, semicolons, tabs) and selecting the most frequent one.

2. **Handling Mixed Quote Styles**:
   - Since fields may be quoted with double quotes, single quotes, or not at all, we can use Python's `csv` module which can handle different quote styles if we define the quoting rules properly.

3. **Converting Date Formats**:
   - We will use the `dateutil` library to parse dates flexibly and convert them to the ISO format (`YYYY-MM-DD`).

4. **Normalizing Number Formats**:
   - We will remove any non-numeric symbols (commas, currency symbols) and convert numbers to a standard format using Python's `locale` module to handle different decimal and thousand separators.

5. **Removing Empty Rows**:
   - Skip rows that are completely empty after stripping whitespace.

6. **Trimming Extra Whitespace**:
   - Strip leading and trailing whitespace from each field.

7. **Normalizing Column Names**:
   - Convert column names to lowercase and replace spaces with underscores for consistency.

8. **Handling Missing Values**:
   - Replace missing or empty fields with a consistent placeholder (e.g., empty string or `NULL`).

9. **Character Encoding Issues**:
   - Decode the input data as UTF-8, which is a common encoding for CSV files, and handle any decoding errors gracefully.

10. **Output Consistent CSV**:
    - Use a consistent delimiter (commas) and quote all fields to ensure they are correctly interpreted.

Here is the implementation of the `normalize_csv` function:

```python
import csv
import re
import io
from dateutil import parser as date_parser
import locale

locale.setlocale(locale.LC_NUMERIC, '')  # Use user's preferred locale settings

def normalize_csv(csv_data: str) -> str:
    # Prepare to read the CSV data
    # Split the input data into lines
    lines = csv_data.strip().splitlines()
    
    # Detect the delimiter by checking the first few lines
    possible_delimiters = [',', ';', '\t', '|']
    delimiter_counts = {d: 0 for d in possible_delimiters}

    for line in lines[:10]:  # Check the first few lines for delimiter detection
        for delim in possible_delimiters:
            delimiter_counts[delim] += line.count(delim)
    
    # Choose the most common delimiter
    delimiter = max(delimiter_counts, key=delimiter_counts.get)
    
    # Use csv.reader to handle the CSV parsing
    csv_reader = csv.reader(lines, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, skipinitialspace=True)
    data = list(csv_reader)
    
    # Normalize column names
    headers = data[0]
    normalized_headers = [re.sub(r'\s+', '_', h.strip().lower()) for h in headers]
    
    normalized_data = [normalized_headers]
    
    for row in data[1:]:
        if len(row) == 0 or all(not cell.strip() for cell in row):
            continue  # Skip empty rows
        
        normalized_row = []
        for cell in row:
            cell = cell.strip()
            
            # Handle missing values
            if cell == '' or cell.lower() in {'null', 'none'}:
                normalized_row.append('')
                continue
            
            # Handle numeric values
            try:
                # Remove currency symbols and thousand separators, replace commas with dots for decimals
                number = locale.atof(re.sub(r'[^\d,.-]', '', cell))
                normalized_row.append(f"{number:.2f}")
                continue
            except (ValueError, locale.Error):
                pass
            
            # Handle dates
            try:
                date = date_parser.parse(cell)
                normalized_row.append(date.strftime('%Y-%m-%d'))
                continue
            except (ValueError, OverflowError):
                pass
            
            # If no special handling needed, keep the original value
            normalized_row.append(cell)
        
        normalized_data.append(normalized_row)
    
    # Write normalized data back to CSV format
    output = io.StringIO()
    csv_writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_ALL)
    csv_writer.writerows(normalized_data)
    
    return output.getvalue()

```

### Explanation of the Code

- **Delimiter Detection**: We check the first few lines to determine the most common delimiter.
- **CSV Parsing**: We use Python's `csv.reader` to handle different quoting styles and delimiters.
- **Normalization**: We iterate through each row and field, applying transformations:
  - **Column Names**: Normalized to lowercase and underscores.
  - **Numbers**: Stripped of non-numeric characters and converted using `locale.atof`.
  - **Dates**: Parsed and formatted using `dateutil.parser`.
  - **Empty Values**: Consistently replaced.
- **Output**: The cleaned data is written back using `csv.writer` with all fields quoted for consistency.

This function is designed to be robust against a variety of edge cases and should handle most common inconsistencies in CSV files.