# Round 3 Response Comparison

## Original Prompt

```
# CSV Normalization and Cleaning Challenge

I want you to make me a sophisticated Python function called `normalize_csv` that takes messy, inconsistent CSV data as input and produces a clean, normalized version while preserving as much information as possible.

The function should have this signature:
```python
def normalize_csv(csv_data: str) -> str:
    """
    Clean and normalize messy CSV data.
    
    Args:
        csv_data: String containing messy CSV data
        
    Returns:
        String containing cleaned, normalized CSV data
    """
```

Your solution should handle the following common issues in CSV files:

1. **Inconsistent delimiters**: Some rows might use commas, others semicolons or tabs
2. **Mixed quote styles**: Some fields might use double quotes, others single quotes, or no quotes
3. **Inconsistent date formats**: Convert all dates to ISO format (YYYY-MM-DD)
4. **Inconsistent number formats**: Convert numbers with various formats (1,000.00 or 1.000,00) to standard format
5. **Empty rows**: Remove completely empty rows
6. **Extra whitespace**: Trim unnecessary whitespace from field values
7. **Inconsistent column names**: Normalize column names to lowercase with underscores
8. **Missing values**: Replace with appropriate NULL values or empty strings
9. **Character encoding issues**: Handle and fix common encoding problems

Here are some examples of messy CSV data that your function should be able to clean:

Example 1:
```
Name, Age, Birth Date, Salary 
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
```

Example 2:
```
Product Name|Price|In Stock|Last Updated
"Wireless Headphones"|"$129.99"|"Yes"|"2023-01-15"
'Smart Watch';€199,95;'no';'01/22/2023'
"USB-C Cable",$19.99,true,"February 3, 2023"
```

Example 3:
```
customer_id,first_name,last_name,email,purchase_amount,purchase_date
1001,John,Doe,john.doe@example.com,"1,240.50","01/15/2023"
1002,Jane,Smith,jane.smith@example.com,"985,75","2023-01-20"
1003,"David, Jr.",Johnson,"david.johnson@example.com","2.399,00","Jan 25, 2023"
```

Your solution should:
1. Detect and adapt to different delimiter styles automatically
2. Handle different types of fields (strings, numbers, dates, booleans) appropriately
3. Preserve the header row but normalize column names
4. Output a consistently formatted CSV with properly escaped fields
5. Be robust to unexpected edge cases
6. Use appropriate error handling

For full credit, provide a detailed explanation of your approach and any assumptions you made.
```

## Model Responses

### o3_mini

**Metrics:** 393 lines, 15.14 KB

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

class o3_miniRound3Solution:
    @staticmethod
    def solve(input_text):
        return o3_miniRound3Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.
        
        Issues addressed:
          - Encoding problems (including BOM)
          - Inconsistent line endings
          - Automatic detection of delimiter among comma, semicolon, tab, pipe
          - Mixed quoting styles: converts to double quotes when needed
          - Normalizes header names (to lowercase with underscores)
          - Adjusts row-lengths when rows have too few/many fields
          - Trims whitespace and strips extraneous quotes from fields
          - Normalizes dates (to ISO format YYYY-MM-DD)
          - Normalizes numbers (removing currency symbols and thousand separators)
          - Standardizes booleans and null-like values
          
        Args:
            csv_data: Input CSV data (string or bytes)
            
        Returns:
            Clean, normalized CSV data as a string
        """
        # 1. Fix encoding issues (if bytes) and remove BOM; normalize line endings.
        if isinstance(csv_data, bytes):
            detected = chardet.detect(csv_data)
            encoding = detected.get('encoding') or 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Split into nonempty lines.
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # 3. Detect the most likely delimiter.
        delimiter = o3_miniRound3Solution.detect_delimiter(lines)
        
        # 4. Convert all lines to a standardized format:
        #    • Replace any common delimiter by comma.
        #    • Normalize quotes to double quotes.
        standardized_lines = o3_miniRound3Solution.standardize_delimiters(lines, delimiter)
        
        # 5. Parse CSV rows using the csv module.
        rows = o3_miniRound3Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # 6. Normalize header names.
        header = [o3_miniRound3Solution.normalize_column_name(col) for col in rows[0]]
        
        # 7. Process and clean each subsequent row.
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip rows that are entirely empty.
            if not row or all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header columns.
            row = o3_miniRound3Solution.adjust_row_length(row, len(header))
            
            # Clean each field according to its content as well as column name hints.
            cleaned_row = []
            for i, field in enumerate(row):
                col_name = header[i] if i < len(header) else f"column_{i}"
                cleaned_field = o3_miniRound3Solution.clean_field(field, col_name)
                cleaned_row.append(cleaned_field)
            normalized_rows.append(cleaned_row)
        
        # 8. Write normalized rows into CSV using the csv writer.
        output = io.StringIO()
        writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines: List[str]) -> str:
        """
        Detect the most likely delimiter based on the first few lines.
        Counts occurrences of candidate delimiters outside quotes.
        Candidates: comma, semicolon, tab, pipe.
        
        Returns:
            The candidate delimiter with the best (most consistent) score.
        """
        candidates = [',', ';', '\t', '|']
        delim_counts: Dict[str, List[int]] = {d: [] for d in candidates}
        
        for line in lines[:min(10, len(lines))]:
            for d in candidates:
                count = o3_miniRound3Solution.count_delimiters_outside_quotes(line, d)
                if count > 0:
                    delim_counts[d].append(count)
        
        best_delim = ','
        best_score = 0
        for d, counts in delim_counts.items():
            if not counts:
                continue
            counter = Counter(counts)
            common_count, frequency = counter.most_common(1)[0]
            score = frequency * common_count  # weight consistency and frequency
            if score > best_score:
                best_score = score
                best_delim = d

        # Special-case: if no delimiter got found use multiple spaces.
        if best_score == 0:
            for line in lines[:min(5, len(lines))]:
                if re.search(r'\s{2,}', line):
                    return r'\s+'
        return best_delim

    @staticmethod
    def count_delimiters_outside_quotes(line: str, delimiter: str) -> int:
        """
        Count delimiter characters that occur outside quoted parts.
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
            elif char == delimiter and not in_quotes:
                count += 1
        return count

    @staticmethod
    def standardize_delimiters(lines: List[str], primary_delimiter: str) -> List[str]:
        """
        Convert all common delimiters in the lines to a standard comma.
        Also convert any mixed quotes to double quotes.
        
        Args:
            lines: List of CSV lines.
            primary_delimiter: Detected delimiter.
            
        Returns:
            A list of modified lines.
        """
        standardized = []
        
        for line in lines:
            # special handling for space-delimited if we detected multiple spaces
            if primary_delimiter == r'\s+' and not any(d in line for d in [',', ';', '\t', '|']):
                # Split by at least 2 spaces.
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
                    new_line += ','
                else:
                    new_line += char
            standardized.append(new_line)
        
        return standardized

    @staticmethod
    def parse_csv_rows(lines: List[str], detected_delimiter: str) -> List[List[str]]:
        """
        Parse CSV rows from standardized lines.
        Uses csv.reader to properly handle quotes.
        
        Args:
            lines: List of standardized CSV lines.
            detected_delimiter: Original detected delimiter (used if needed).
            
        Returns:
            List of rows with each row as a list of strings.
        """
        # If using space delimiter (r'\s+'), we already split fields.
        actual_delimiter = ',' if detected_delimiter == r'\s+' else detected_delimiter
        csv_text = "\n".join(lines)
        try:
            reader = csv.reader(io.StringIO(csv_text), delimiter=actual_delimiter)
            rows = [ [cell.strip() for cell in row] for row in reader ]
            return rows
        except Exception:
            # Fallback to manual splitting:
            rows = []
            for line in lines:
                row = line.split(actual_delimiter)
                rows.append([cell.strip() for cell in row])
            return rows

    @staticmethod
    def normalize_column_name(name: str) -> str:
        """
        Normalize header column name to lowercase with underscores.
        Also strips any surrounding quotes and extra whitespace.
        """
        name = name.strip()
        if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
            name = name[1:-1].strip()
        # Replace any non-alphanumeric character with underscore and collapse multiple underscores.
        name = re.sub(r'[^\w\s]', '_', name)
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.strip('_').lower() or "column"

    @staticmethod
    def adjust_row_length(row: List[str], expected_length: int) -> List[str]:
        """
        Adjust the row so that it has exactly expected_length fields.
          - If too few, pad with empty strings.
          - If too many, try to merge extra fields into the last cell.
        """
        if len(row) == expected_length:
            return row
        if len(row) < expected_length:
            return row + [""] * (expected_length - len(row))
        
        # If too many fields, attempt to combine extras in the last column.
        # This simple approach concatenates all extra fields.
        new_row = row[:expected_length-1]
        combined = " ".join(row[expected_length-1:])
        new_row.append(combined)
        return new_row

    @staticmethod
    def clean_field(field: str, column_name: str) -> str:
        """
        Clean and normalize a single field.
          • Remove extraneous whitespace and matching outer quotes.
          • Replace null-like values with empty string.
          • Convert boolean-like values to "true"/"false".
          • If the value looks like a date (or column name suggests it), convert to ISO (YYYY-MM-DD).
          • If numeric, normalize number format.
        
        Args:
            field: Field content.
            column_name: Name of the column (to help infer type).
            
        Returns:
            Cleaned field.
        """
        field = field.strip()
        if len(field) >= 2 and field[0] == field[-1] and field[0] in ['"', "'"]:
            field = field[1:-1].strip()
        
        # Handle missing/null values.
        if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
            return ""
        
        # Normalize booleans.
        low = field.lower()
        if low in ['true', 'yes', 'y', '1']:
            return "true"
        if low in ['false', 'no', 'n', '0']:
            return "false"
        
        # Try converting dates: if the field looks like a date
        if o3_miniRound3Solution.looks_like_date(field) or o3_miniRound3Solution.is_date_column(column_name):
            date_obj = o3_miniRound3Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        # Try converting numbers if field contains digits.
        if o3_miniRound3Solution.looks_like_numeric(field):
            num = o3_miniRound3Solution.format_number(field)
            if num is not None:
                return num
        
        return field

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """
        Heuristic to decide if a field value resembles a date.
        Checks for common date patterns.
        """
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',     # e.g., 04/25/1991 or 25-12-2023
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',         # e.g., 1991-04-25
            r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # e.g., May 3rd, 1992
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'     # e.g., 3rd May 1992
        ]
        return any(re.search(p, field) for p in date_patterns)

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """
        Check if the column name suggests it contains dates.
        """
        indicators = ['date', 'day', 'month', 'year', 'time', 'birth', 'updated', 'created']
        return any(ind in column_name.lower() for ind in indicators)

    @staticmethod
    def parse_date(date_str: str) -> Optional[date_parser]:
        """
        Attempt to parse a date from a string, removing ordinal suffixes if present.
        
        Returns:
            A datetime object if successful, otherwise None.
        """
        try:
            # Remove ordinal suffixes like "3rd" -> "3"
            cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
            dt = date_parser.parse(cleaned, fuzzy=True)
            return dt
        except Exception:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """
        Check if the field appears to represent a number.
        Looks for digits and characters like comma or period.
        """
        # Remove any currency symbols for the check
        cleaned = re.sub(r'[$€£]', '', field)
        return any(ch.isdigit() for ch in cleaned)

    @staticmethod
    def format_number(number_str: str) -> Optional[str]:
        """
        Normalize a number into a standard format.
        Removes currency symbols, thousands separators and converts decimals.
        
        Handles US style (e.g. 1,234.56) and European style (e.g. 1.234,56).
        
        Returns:
            A standardized number string (integer if no fractional part) or None on failure.
        """
        # Remove spaces and currency symbols.
        temp = re.sub(r'[\s$€£]', '', number_str)
        
        # If both comma and dot exist, decide which is the decimal separator.
        if ',' in temp and '.' in temp:
            if temp.rfind(',') > temp.rfind('.'):
                # Likely European style: thousands separator is dot; replace dot by empty and comma by dot.
                temp = temp.replace('.', '')
                temp = temp.replace(',', '.')
            else:
                # Likely US style: remove commas.
                temp = temp.replace(',', '')
        elif ',' in temp and '.' not in temp:
            # Ambiguous: if comma appears only once and last part is two digits, treat as decimal.
            parts = temp.split(',')
            if len(parts) == 2 and len(parts[1]) in {1,2}:
                temp = temp.replace(',', '.')
            else:
                temp = temp.replace(',', '')
        
        try:
            num = float(temp)
            # Return integer string if no fraction, else a normalized float (dropping unneeded trailing zeros)
            if num.is_integer():
                return str(int(num))
            else:
                s = f"{num:.6f}".rstrip('0').rstrip('.')
                return s
        except ValueError:
            return None
```

### gpt4o

**Metrics:** 217 lines, 7.87 KB

```python
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

class gpt4oRound3Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return gpt4oRound3Solution.normalize_csv(input_text)
    
    @staticmethod
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
        delimiter = gpt4oRound3Solution.detect_delimiter(lines)
        standardized_lines = gpt4oRound3Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV with detected delimiter
        rows = gpt4oRound3Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""
        
        # Normalize the header
        header = [gpt4oRound3Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            if not row or all(not cell.strip() for cell in row):
                continue
            adjusted_row = gpt4oRound3Solution.adjust_row_length(row, len(header))
            normalized_row = [gpt4oRound3Solution.clean_field(cell, header[i] if i < len(header) else f"column_{i}") for i, cell in enumerate(adjusted_row)]
            normalized_rows.append(normalized_row)
        
        # Write to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        
        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines: List[str]) -> str:
        common_delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: [] for d in common_delimiters}
        
        for line in lines[:min(10, len(lines))]:
            for delim in common_delimiters:
                count = gpt4oRound3Solution.count_delimiters_outside_quotes(line, delim)
                delimiter_counts[delim].append(count)
        
        best_delimiter = max(delimiter_counts, key=lambda d: Counter(delimiter_counts[d]).most_common(1)[0][1])
        return best_delimiter

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def parse_csv_rows(lines: List[str], detected_delimiter: str) -> List[List[str]]:
        csv_text = '\n'.join(lines)
        try:
            reader = csv.reader(io.StringIO(csv_text), delimiter=',')
            return [row for row in reader]
        except Exception:
            return []

    @staticmethod
    def normalize_column_name(column: str) -> str:
        column = column.strip().lower()
        column = re.sub(r'[^\w\s]', '_', column)
        column = re.sub(r'\s+', '_', column)
        return column.strip('_')

    @staticmethod
    def adjust_row_length(row: List[str], expected_length: int) -> List[str]:
        if len(row) < expected_length:
            return row + [""] * (expected_length - len(row))
        elif len(row) > expected_length:
            return row[:expected_length]
        return row

    @staticmethod
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
        
        if gpt4oRound3Solution.looks_like_date(field) or gpt4oRound3Solution.is_date_column(column_name):
            date_obj = gpt4oRound3Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        if gpt4oRound3Solution.looks_like_numeric(field):
            return gpt4oRound3Solution.format_number(field)
        
        return field

    @staticmethod
    def looks_like_date(field: str) -> bool:
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',
            r'\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{2,4}'
        ]
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def parse_date(date_str: str) -> Optional[date_parser.parser]:
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except (date_parser.ParserError, ValueError):
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        cleaned = re.sub(r'[^\d.,-]', '', field)
        return bool(re.match(r'^[+-]?\d+([.,]\d+)*$', cleaned))

    @staticmethod
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
```

### claude37

**Metrics:** 477 lines, 16.84 KB

```python
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Any

class claude37Round3Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return claude37Round3Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.
        
        Args:
            csv_data: String containing messy CSV data
            
        Returns:
            String containing cleaned, normalized CSV data
        """
        # --- 1. Handle encoding issues ---
        if isinstance(csv_data, bytes):
            detected = chardet.detect(csv_data)
            encoding = detected.get('encoding') or 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        
        # Remove BOM if present and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into non-empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # --- 2. Detect delimiter ---
        delimiter = claude37Round3Solution.detect_delimiter(lines)
        
        # --- 3. Parse CSV data ---
        rows = claude37Round3Solution.parse_csv_with_delimiter(lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # --- 4. Normalize header ---
        header = [claude37Round3Solution.normalize_column_name(col) for col in rows[0]]
        
        # --- 5. Process data rows ---
        cleaned_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header length
            row = claude37Round3Solution.adjust_row_length(row, len(header))
            
            # Clean each field based on content and column name
            cleaned_row = []
            for i, field in enumerate(row):
                column_name = header[i] if i < len(header) else f"column_{i}"
                cleaned_row.append(claude37Round3Solution.clean_field(field, column_name))
            
            cleaned_rows.append(cleaned_row)
        
        # --- 6. Write normalized data back to CSV ---
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(cleaned_rows)
        
        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines: List[str]) -> str:
        """
        Detect the most likely delimiter in CSV data by analyzing patterns outside quoted sections.
        
        Args:
            lines: List of CSV data lines
            
        Returns:
            The most likely delimiter character
        """
        common_delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: [] for d in common_delimiters}
        
        # Analyze first few lines
        for line in lines[:min(10, len(lines))]:
            for delimiter in common_delimiters:
                # Count delimiters outside quotes
                count = claude37Round3Solution.count_delimiters_outside_quotes(line, delimiter)
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

    @staticmethod
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

    @staticmethod
    def parse_csv_with_delimiter(lines: List[str], delimiter: str) -> List[List[str]]:
        """
        Parse CSV data using the detected delimiter.
        
        Args:
            lines: List of CSV data lines
            delimiter: The detected delimiter
            
        Returns:
            List of parsed rows
        """
        # Handle space delimiter specially
        if delimiter == r'\s+':
            rows = []
            for line in lines:
                # Split by multiple spaces while preserving quoted content
                fields = []
                current = ""
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
                        current += char
                    elif char.isspace() and not in_quotes and current:
                        if re.match(r'\s{2,}', char):
                            fields.append(current)
                            current = ""
                        else:
                            current += char
                    else:
                        current += char
                
                if current:
                    fields.append(current)
                rows.append(fields)
            return rows
        
        # For standard delimiters, use csv module
        try:
            # First standardize delimiters and quotes
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
            
            csv_text = '\n'.join(standardized_lines)
            reader = csv.reader(io.StringIO(csv_text))
            return list(reader)
        except Exception:
            # Fallback to manual parsing if csv module fails
            rows = []
            for line in lines:
                fields = []
                current = ""
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
                        else:
                            current += char
                    elif char == delimiter and not in_quotes:
                        fields.append(current)
                        current = ""
                    else:
                        current += char
                
                fields.append(current)
                rows.append(fields)
            
            return rows

    @staticmethod
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
        
        # Convert to lowercase and replace spaces with underscores
        name = re.sub(r'\s+', '_', name.lower())
        
        # Replace non-alphanumeric characters with underscores
        name = re.sub(r'[^\w]', '_', name)
        
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        return name if name else "column"  # Ensure we don't return empty string

    @staticmethod
    def adjust_row_length(row: List[str], expected_length: int) -> List[str]:
        """
        Adjust row to match expected length.
        
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
        
        # If row is too long, try to intelligently combine fields
        new_row = []
        i = 0
        
        while i < len(row):
            # If we've reached our target length minus 1, combine all remaining fields
            if len(new_row) == expected_length - 1:
                new_row.append(" ".join(row[i:]))
                break
            
            field = row[i]
            
            # Check if this field starts with a quote but doesn't end with one
            if ((field.startswith('"') and not field.endswith('"')) or 
                (field.startswith("'") and not field.endswith("'"))):
                # Find matching end quote
                combined = field
                j = i + 1
                quote_char = field[0]
                found_end = False
                
                while j < len(row):
                    combined += "," + row[j]  # Add back the delimiter
                    if row[j].endswith(quote_char):
                        found_end = True
                        break
                    j += 1
                
                if found_end:
                    new_row.append(combined)
                    i = j + 1
                else:
                    new_row.append(field)
                    i += 1
            else:
                new_row.append(field)
                i += 1
        
        # If we still have too many fields, truncate
        if len(new_row) > expected_length:
            return new_row[:expected_length]
        
        return new_row

    @staticmethod
    def clean_field(field: str, column_name: str = "") -> str:
        """
        Clean and normalize a field value.
        
        Args:
            field: Field value to clean
            column_name: Name of the column (for type inference)
            
        Returns:
            Cleaned field value
        """
        # Remove outer quotes and trim whitespace
        field = field.strip()
        if len(field) >= 2:
            if (field[0] == field[-1]) and field[0] in ['"', "'"]:
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
        if claude37Round3Solution.looks_like_date_column(column_name) or claude37Round3Solution.looks_like_date(field):
            try:
                date_obj = claude37Round3Solution.normalize_date(field)
                if date_obj:
                    return date_obj
            except Exception:
                pass  # Fall through to other normalizations if date parsing fails
        
        # Try to detect and normalize numeric fields
        if claude37Round3Solution.looks_like_numeric(field):
            try:
                return claude37Round3Solution.normalize_number(field)
            except Exception:
                pass  # Return original if number parsing fails
        
        return field

    @staticmethod
    def looks_like_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if field value looks like a date."""
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # Month DD, YYYY
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
        ]
        
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def normalize_date(date_str: str) -> str:
        """
        Normalize date to ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            Normalized date string
        """
        # Remove currency symbols and other non-date characters
        date_str = re.sub(r'[$€£]', '', date_str)
        
        # Handle ordinal suffixes (1st, 2nd, 3rd, 4th)
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        try:
            date_obj = date_parser.parse(date_str, fuzzy=True)
            return date_obj.strftime('%Y-%m-%d')
        except Exception:
            return date_str  # Return original if parsing fails

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if field looks like it contains a number."""
        # Strip currency symbols, spaces, and common separators
        cleaned = re.sub(r'[$€£\s]', '', field)
        return bool(re.match(r'^[+-]?[\d,.]+$', cleaned))

    @staticmethod
    def normalize_number(number_str: str) -> str:
        """
        Normalize numeric format.
        
        Args:
            number_str: Number string to normalize
            
        Returns:
            Normalized number string
        """
        # Remove currency symbols and whitespace
        number_str = re.sub(r'[$€£\s]', '', number_str)
        
        # Handle different number formats
        if ',' in number_str and '.' in number_str:
            # Determine which is the decimal separator
            if number_str.rindex(',') > number_str.rindex('.'):
                # European format: 1.234,56
                number_str = number_str.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.56
                number_str = number_str.replace(',', '')
        elif ',' in number_str:
            # Could be either 1,234 (US) or 1,23 (European)
            if len(number_str.split(',')[-1]) <= 2:
                # Likely European decimal comma
                number_str = number_str.replace(',', '.')
            else:
                # Likely US thousands separator
                number_str = number_str.replace(',', '')
        
        try:
            # Check if it's a valid number
            float(number_str)
            # Return it without trailing zeros after decimal
            if '.' in number_str:
                return str(float(number_str))
            return number_str
        except ValueError:
            # If conversion fails, return original
            return number_str
```

### mistral_large

**Metrics:** 477 lines, 16.3 KB

```python
import csv
import re
import io
import chardet
from dateutil import parser
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

class mistral_largeRound3Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return mistral_largeRound3Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.

        Args:
            csv_data: String containing messy CSV data

        Returns:
            String containing cleaned, normalized CSV data
        """
        # Handle character encoding issues
        if isinstance(csv_data, bytes):
            detected = chardet.detect(csv_data)
            csv_data = csv_data.decode(detected['encoding'] or 'utf-8', errors='replace')

        # Remove BOM if present and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')

        # Split into lines and remove empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""

        # Detect and standardize delimiters
        delimiter = mistral_largeRound3Solution.detect_delimiter(lines)
        standardized_lines = mistral_largeRound3Solution.standardize_delimiters(lines, delimiter)

        # Parse CSV with detected delimiter
        rows = mistral_largeRound3Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""

        # Normalize header row
        header = [mistral_largeRound3Solution.normalize_column_name(col) for col in rows[0]]

        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue

            # Adjust row length to match header
            adjusted_row = mistral_largeRound3Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue

            # Clean each field
            normalized_row = [
                mistral_largeRound3Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
                for i, field in enumerate(adjusted_row)
            ]

            normalized_rows.append(normalized_row)

        # Write normalized data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)

        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines: List[str]) -> str:
        """
        Detect the most likely delimiter in CSV data by analyzing patterns outside quoted sections.

        Args:
            lines: List of CSV data lines

        Returns:
            The most likely delimiter character
        """
        common_delimiters = [',', ';', '\t', '|']
        delimiter_counts: Dict[str, List[int]] = {d: [] for d in common_delimiters}

        for line in lines[:min(10, len(lines))]:  # Check first 10 lines or fewer
            for delimiter in common_delimiters:
                count = mistral_largeRound3Solution.count_delimiters_outside_quotes(line, delimiter)
                if count > 0:
                    delimiter_counts[delimiter].append(count)

        # Find most consistent non-zero delimiter
        best_delimiter = ','  # Default to comma
        best_score = 0

        for delimiter, counts in delimiter_counts.items():
            if not counts:
                continue

            # Calculate consistency (how many lines have the same count)
            count_frequencies = Counter(counts)
            most_common_count, frequency = count_frequencies.most_common(1)[0]
            consistency_score = frequency * most_common_count  # Weight by both frequency and count

            if consistency_score > best_score:
                best_delimiter = delimiter
                best_score = consistency_score

        # Special case: if no clear delimiter is found, check for multiple spaces
        if best_score == 0:
            for line in lines[:min(5, len(lines))]:
                if re.search(r'\s{2,}', line):
                    return r'\s+'

        return best_delimiter

    @staticmethod
    def count_delimiters_outside_quotes(line: str, delimiter: str) -> int:
        """Count delimiters that are outside of quoted sections."""
        count = 0
        in_quotes = False
        quote_char = None
        escaped = False

        for i, char in enumerate(line):
            # Handle escape sequences
            if escaped:
                escaped = False
                continue

            if char == '\\':
                escaped = True
                continue

            # Toggle quote state
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None

            # Count delimiter if outside quotes
            elif char == delimiter and not in_quotes:
                count += 1

        return count

    @staticmethod
    def standardize_delimiters(lines: List[str], primary_delimiter: str) -> List[str]:
        """
        Standardize all lines to use the primary delimiter.

        Args:
            lines: List of CSV data lines
            primary_delimiter: The delimiter to standardize to

        Returns:
            List of standardized CSV lines
        """
        standardized_lines = []

        for line in lines:
            # Handle space-delimited lines
            if primary_delimiter == r'\s+' and not any(d in line for d in [',', ';', '\t', '|']):
                fields = re.split(r'\s{2,}', line)
                standardized_lines.append(','.join(fields))
                continue

            # For other delimiters, process quotes properly
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
                        new_line += char  # Different quote inside quoted text
                elif char in [',', ';', '\t', '|'] and not in_quotes:
                    new_line += ','  # Standardize to comma
                else:
                    new_line += char

            standardized_lines.append(new_line)

        return standardized_lines

    @staticmethod
    def parse_csv_rows(lines: List[str], detected_delimiter: str) -> List[List[str]]:
        """
        Parse CSV lines into rows, handling mixed quote styles and other issues.

        Args:
            lines: List of standardized CSV lines
            detected_delimiter: The primary delimiter used in the data

        Returns:
            List of parsed CSV rows
        """
        # Join lines back into a single string
        csv_text = '\n'.join(lines)

        # Use the correct delimiter for parsing
        actual_delimiter = ',' if detected_delimiter == r'\s+' else detected_delimiter

        try:
            # Try parsing with csv module
            reader = csv.reader(io.StringIO(csv_text), delimiter=actual_delimiter)
            rows = list(reader)

            # Post-process to clean up quotes and whitespace
            clean_rows = []
            for row in rows:
                clean_row = []
                for field in row:
                    field = field.strip()
                    # Remove matching outer quotes if present
                    if (field.startswith('"') and field.endswith('"')) or \
                       (field.startswith("'") and field.endswith("'")):
                        field = field[1:-1].strip()
                    clean_row.append(field)
                clean_rows.append(clean_row)

            return clean_rows
        except Exception as e:
            # Fallback: manual parsing
            rows = []
            for line in lines:
                fields = []
                current_field = ""
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
                        else:
                            current_field += char
                    elif char == actual_delimiter and not in_quotes:
                        fields.append(current_field.strip())
                        current_field = ""
                    else:
                        current_field += char

                fields.append(current_field.strip())
                rows.append(fields)

            return rows

    @staticmethod
    def normalize_column_name(column: str) -> str:
        """
        Normalize column name to lowercase with underscores.

        Args:
            column: The column name to normalize

        Returns:
            Normalized column name
        """
        # Remove outer quotes if present
        column = column.strip()
        if (column.startswith('"') and column.endswith('"')) or \
           (column.startswith("'") and column.endswith("'")):
            column = column[1:-1].strip()

        # Replace non-alphanumeric with underscores
        normalized = re.sub(r'[^\w\s]', '_', column)
        # Replace whitespace with underscores and convert to lowercase
        normalized = re.sub(r'\s+', '_', normalized).lower()
        # Remove consecutive underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        return normalized or "column"  # Ensure we don't return empty string

    @staticmethod
    def adjust_row_length(row: List[str], expected_length: int) -> Optional[List[str]]:
        """
        Adjust row to match expected length.

        Args:
            row: The row to adjust
            expected_length: The expected number of fields

        Returns:
            Adjusted row or None if adjustment is not possible
        """
        if len(row) == expected_length:
            return row

        # If row is too short, pad with empty strings
        if len(row) < expected_length:
            return row + [""] * (expected_length - len(row))

        # If row is too long, try to combine fields that might have been incorrectly split
        new_row = []
        i = 0
        while i < len(row):
            field = row[i]

            # If we've already reached expected length, combine all remaining fields
            if len(new_row) == expected_length - 1:
                new_row.append(' '.join(row[i:]))
                break

            # Check if this field starts with a quote but doesn't end with one
            if (field.startswith('"') and not field.endswith('"')) or \
               (field.startswith("'") and not field.endswith("'")):
                # Find the matching end quote
                combined = field
                j = i + 1
                found_end = False
                quote_char = field[0]

                while j < len(row):
                    combined += "," + row[j]  # Add back the comma that was removed during parsing
                    if row[j].endswith(quote_char):
                        found_end = True
                        break
                    j += 1

                if found_end:
                    new_row.append(combined)
                    i = j + 1
                else:
                    new_row.append(field)
                    i += 1
            else:
                new_row.append(field)
                i += 1

        # If we still have too many fields, truncate
        if len(new_row) > expected_length:
            return new_row[:expected_length]

        return new_row

    @staticmethod
    def clean_field(field: str, column_name: str) -> str:
        """
        Clean and normalize field value based on content and column name.

        Args:
            field: The field value to clean
            column_name: The name of the column this field belongs to

        Returns:
            Cleaned field value
        """
        # Trim whitespace and remove outer quotes
        field = field.strip()
        if (field.startswith('"') and field.endswith('"')) or \
           (field.startswith("'") and field.endswith("'")):
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
        if mistral_largeRound3Solution.looks_like_date(field) or mistral_largeRound3Solution.is_date_column(column_name):
            try:
                date_obj = mistral_largeRound3Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass  # If date parsing fails, continue with other cleaning

        # Try to detect and normalize numeric fields
        if mistral_largeRound3Solution.looks_like_numeric(field):
            try:
                return mistral_largeRound3Solution.format_number(field)
            except Exception:
                pass  # If number parsing fails, return cleaned string

        # Return the cleaned string
        return field
    
    @staticmethod
    def looks_like_date(field: str) -> bool:
        """
        Check if field looks like a date.
        """
        # Check for common date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',     # YYYY/MM/DD
            r'\d{1,2}\s+[A-Za-z]{3,}\s+\d{2,4}',  # DD Month YYYY
        ]
        for pattern in date_patterns:
            if re.match(pattern, field):
                return True
        return False

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """
        Check if column name suggests it contains date values.
        """
        date_keywords = ['date', 'day', 'month', 'year', 'time', 'born', 'created', 'modified', 'updated']
        return any(keyword in column_name.lower() for keyword in date_keywords)

    @staticmethod
    def parse_date(field: str) -> Optional[datetime]:
        """
        Parse a date string into a datetime object.
        """
        try:
            return parser.parse(field)
        except:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """
        Check if field looks like a numeric value.
        """
        # Remove currency symbols and thousands separators
        cleaned = re.sub(r'[$€£¥]', '', field)
        cleaned = re.sub(r'[,\s]', '', cleaned)
        # Check if it's a valid number
        return bool(re.match(r'^-?\d+(\.\d+)?$', cleaned))

    @staticmethod
    def format_number(field: str) -> str:
        """
        Format numeric values consistently.
        """
        # Remove currency symbols and thousands separators
        cleaned = re.sub(r'[$€£¥]', '', field)
        cleaned = re.sub(r'[,\s]', '', cleaned)
        
        # Handle decimal numbers
        if '.' in cleaned:
            try:
                value = float(cleaned)
                # If it's a whole number, convert to int
                if value.is_integer():
                    return str(int(value))
                # Otherwise, return with consistent decimal format
                return str(value)
            except ValueError:
                return field
        
        # Handle integers
        try:
            return str(int(cleaned))
        except ValueError:
            return field
```

