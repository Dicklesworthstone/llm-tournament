#!/usr/bin/env python3
"""
LLM Tournament Test Suite

This script tests solutions from all rounds and models on a given input file,
and collects metrics on the results.
"""

import os
import time
import json
import inspect
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import io
import re
from dateutil.parser import parse

class o3_miniRound0Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return o3_miniRound0Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.
        
        Args:
            csv_data: String containing messy CSV data
            
        Returns:
            String containing cleaned, normalized CSV data
        """
        # List of known delimiters (other than comma) that we will standardize to comma.
        additional_delims = [';', '\t', '|']
        cleaned_lines = []
        
        # Split the CSV data into lines and drop those that are completely empty.
        for line in csv_data.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue   # skip blank lines
            
            # Replace any occurrence of our additional delimiters with comma.
            new_line = stripped_line
            for d in additional_delims:
                new_line = new_line.replace(d, ',')
            
            # Fallback: if after our replacements there is no comma but the line seems to contain multiple fields,
            # check for two (or more) whitespace separation and rejoin by comma.
            if ',' not in new_line:
                # If there are at least two chunks separated by two or more spaces
                fields = re.split(r'\s{2,}', new_line)
                if len(fields) > 1:
                    new_line = ','.join(fields)
            cleaned_lines.append(new_line)
        
        # Join back into a single normalized CSV text.
        unified_csv = "\n".join(cleaned_lines)
        
        # Now use csv.reader to parse the (hopefully) delimited rows.
        # We use io.StringIO to wrap our text.
        reader = csv.reader(io.StringIO(unified_csv), delimiter=',')
        rows = list(reader)
        if not rows:
            return ""
        
        # Normalize the header row: trim and lowercase names, and replace whitespace with underscores.
        header = []
        for col in rows[0]:
            c = col.strip().lower()
            # replace internal spaces with underscore and remove any stray non-alnum characters (except underscores)
            c = re.sub(r'\s+', '_', c)
            c = re.sub(r'[^\w]', '', c)
            header.append(c)
        
        normalized_rows = [header]
        
        # Now process each non-header row.
        for row in rows[1:]:
            # If the entire row is empty (or consisting solely of empty/whitespace values) skip it.
            if all(x.strip() == "" for x in row):
                continue
            normalized_row = [o3_miniRound0Solution.normalize_value(field) for field in row]
            normalized_rows.append(normalized_row)
        
        # Write normalized data using csv.writer so that fields are properly escaped.
        out_io = io.StringIO()
        writer = csv.writer(out_io, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in normalized_rows:
            writer.writerow(row)
        
        return out_io.getvalue()

    @staticmethod
    def remove_outer_quotes(s: str) -> str:
        """Strip leading and trailing whitespace and remove matching single or double quotes."""
        s = s.strip()
        if len(s) >= 2:
            # Check for matching double quotes or single quotes
            if (s[0] == s[-1]) and s[0] in ('"', "'"):
                return s[1:-1].strip()
        return s

    @staticmethod
    def try_normalize_date(s: str) -> str:
        """Try to parse a date and reformat to ISO YYYY-MM-DD.
           Returns the ISO string if successful, or None."""
        try:
            # Use fuzzy parsing to handle things like "May 3rd, 1992"
            dt = parse(s, fuzzy=True, dayfirst=False)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None

    @staticmethod
    def try_normalize_number(s: str) -> str:
        """Try to clean up a number string.
           It removes currency symbols and standardizes thousand and decimal separators.
           Returns the normalized number as a string if successful; otherwise None."""
        # Remove common currency symbols and spaces.
        cleaned = re.sub(r'[€$]', '', s).strip()
        # First attempt: Assume US/International style where comma is a thousand separator.
        try:
            # Remove commas then convert to float.
            num = float(cleaned.replace(',', ''))
            # We output in standard format.
            return f"{num}"
        except Exception:
            pass
        # Second attempt: Assume European format, e.g. "1.000,00"
        try:
            # Remove dots as thousand separators and replace comma with dot.
            cleaned2 = cleaned.replace('.', '').replace(',', '.')
            num = float(cleaned2)
            return f"{num}"
        except Exception:
            return None

    @staticmethod
    def normalize_value(val: str) -> str:
        v = o3_miniRound0Solution.remove_outer_quotes(val)
        v = v.strip()
        if v == "":
            return ""  # treat empty as missing
        
        lower_v = v.lower()

        # Handle booleans explicitly.
        if lower_v in ('yes', 'true'):
            return "true"
        if lower_v in ('no', 'false'):
            return "false"
        
        # Try to normalize date if possible.
        date_norm = o3_miniRound0Solution.try_normalize_date(v)
        if date_norm is not None:
            return date_norm
        
        # Try to normalize the number.
        # We only attempt this if v has digits in it.
        if re.search(r'\d', v):
            num_norm = o3_miniRound0Solution.try_normalize_number(v)
            if num_norm is not None:
                return num_norm
        
        # Otherwise, return the cleaned string.
        return v
import csv
import re
import io
from dateutil import parser as date_parser
import locale

class gpt4oRound0Solution:
    @staticmethod
    def solve(input_text):
        return gpt4oRound0Solution.normalize_csv(input_text)
    
    @staticmethod
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
        
        locale.setlocale(locale.LC_NUMERIC, '')  # Use user's preferred locale settings
        
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
import csv
import re
import io
import chardet
import dateutil.parser
from datetime import datetime
import unicodedata
import locale

class claude37Round0Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return claude37Round0Solution.normalize_csv(input_text)
    
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
        
        # Remove BOM if present
        csv_data = csv_data.lstrip('\ufeff')
        
        # Normalize line endings
        csv_data = csv_data.replace('\r\n', '\n').replace('\r', '\n')
        
        # Skip completely empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # Detect delimiter
        delimiter = claude37Round0Solution.detect_delimiter(lines)
        
        # Parse CSV with the detected delimiter
        reader = csv.reader(io.StringIO('\n'.join(lines)), delimiter=delimiter, quotechar='"')
        rows = list(reader)
        
        if not rows:
            return ""
        
        # Normalize header row
        header = [claude37Round0Solution.normalize_column_name(col.strip()) for col in rows[0]]
        
        # Process data rows
        cleaned_rows = [header]
        for row in rows[1:]:
            # Skip rows that don't match header length after attempting to fix
            adjusted_row = claude37Round0Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue
                
            # Clean each field
            cleaned_row = []
            for i, field in enumerate(adjusted_row):
                col_name = header[i] if i < len(header) else f"column_{i}"
                cleaned_field = claude37Round0Solution.clean_field(field, col_name)
                cleaned_row.append(cleaned_field)
            
            cleaned_rows.append(cleaned_row)
        
        # Write the cleaned data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(cleaned_rows)
        
        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines):
        """Detect the most likely delimiter in the CSV data."""
        common_delimiters = [',', ';', '\t', '|']
        
        # Count occurrences of each delimiter in each line
        delimiter_counts = {d: [] for d in common_delimiters}
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Count delimiters outside of quoted sections
            for delimiter in common_delimiters:
                count = claude37Round0Solution.count_delimiters_outside_quotes(line, delimiter)
                delimiter_counts[delimiter].append(count)
        
        # Find the delimiter that has the most consistent non-zero count
        best_delimiter = ','  # Default to comma
        best_consistency = 0
        
        for delimiter, counts in delimiter_counts.items():
            if not counts:
                continue
                
            # Filter out zeros
            non_zero_counts = [c for c in counts if c > 0]
            if not non_zero_counts:
                continue
                
            # Calculate consistency (how many lines have the same count)
            most_common_count = max(set(non_zero_counts), key=non_zero_counts.count)
            consistency = non_zero_counts.count(most_common_count) / len(non_zero_counts)
            
            if consistency > best_consistency:
                best_delimiter = delimiter
                best_consistency = consistency
        
        return best_delimiter

    @staticmethod
    def count_delimiters_outside_quotes(line, delimiter):
        """Count delimiters that are outside of quoted sections."""
        count = 0
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(line):
            # Toggle quote state
            if char in ['"', "'"]:
                # Check if this quote is escaped
                if i > 0 and line[i-1] == '\\':
                    continue
                    
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
    def normalize_column_name(name):
        """Normalize column name to lowercase with underscores."""
        # Replace non-alphanumeric chars with underscores
        name = re.sub(r'[^\w\s]', '_', name)
        # Replace spaces with underscores and convert to lowercase
        name = re.sub(r'\s+', '_', name).lower()
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name

    @staticmethod
    def adjust_row_length(row, expected_length):
        """Adjust row to match expected length."""
        if len(row) == expected_length:
            return row
        
        # Try to combine fields that might have been incorrectly split
        if len(row) > expected_length:
            # Look for quotes that might indicate fields that should be combined
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
                    while j < len(row):
                        combined += " " + row[j]
                        if (field.startswith('"') and row[j].endswith('"')) or \
                           (field.startswith("'") and row[j].endswith("'")):
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
                    
            if len(new_row) == expected_length:
                return new_row
        
        # If row is too short, pad with empty strings
        if len(row) < expected_length:
            return row + [""] * (expected_length - len(row))
        
        # If row is still too long, truncate
        if len(row) > expected_length:
            return row[:expected_length]
        
        return row

    @staticmethod
    def clean_field(field, column_name):
        """Clean and normalize a field value based on content and column name."""
        # Remove leading/trailing quotes and whitespace
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
        if claude37Round0Solution.looks_like_date_column(column_name):
            try:
                date_obj = claude37Round0Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except:
                pass  # If date parsing fails, continue with other cleaning
        
        # Try to detect and normalize numeric fields
        if claude37Round0Solution.looks_like_numeric(field):
            try:
                return claude37Round0Solution.format_number(field)
            except:
                pass  # If number parsing fails, return cleaned string
        
        return field

    @staticmethod
    def looks_like_date_column(column_name):
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def parse_date(date_str):
        """Parse a date string in various formats."""
        # Remove currency symbols and other non-date characters
        date_str = re.sub(r'[$€£]', '', date_str)
        
        # Handle special formats like "May 3rd, 1992"
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        try:
            return dateutil.parser.parse(date_str, fuzzy=True)
        except:
            return None

    @staticmethod
    def looks_like_numeric(field):
        """Check if field looks like it contains a number."""
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[$€£\s,]', '', field)
        # Replace European decimal separator
        cleaned = cleaned.replace(',', '.')
        
        # Check if it's a number
        return bool(re.match(r'^-?\d+(\.\d+)?$', cleaned))

    @staticmethod
    def format_number(number_str):
        """Format a number string to a standard format."""
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£\s]', '', number_str)
        
        # Handle European number format (1.234,56 -> 1234.56)
        if ',' in cleaned and '.' in cleaned:
            if cleaned.rindex(',') > cleaned.rindex('.'):
                # European format: 1.234,56
                cleaned = cleaned.replace('.', '')
                cleaned = cleaned.replace(',', '.')
            # else: US format: 1,234.56 - just remove commas
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned and '.' not in cleaned:
            # Could be European decimal or US thousand separator
            # Assume European if after last comma there are exactly 2 digits
            if re.search(r',\d{2}$', cleaned):
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        
        # Parse and format the number
        try:
            num = float(cleaned)
            # If it's an integer, return it as such
            if num.is_integer():
                return str(int(num))
            else:
                # Format with up to 6 decimal places, removing trailing zeros
                return str(round(num, 6)).rstrip('0').rstrip('.') if '.' in str(round(num, 6)) else str(round(num, 6))
        except ValueError:
            return number_str  # Return original if parsing fails
import csv
import re
from io import StringIO
from dateutil import parser
from collections import Counter

class mistral_largeRound0Solution:
    @staticmethod
    def detect_delimiter(csv_data):
        lines = csv_data.splitlines()
        delimiter_candidates = [',', ';', '\t', '|']
        delimiter_counts = Counter()
        for line in lines[:5]:  # Check the first 5 lines
            for delimiter in delimiter_candidates:
                delimiter_counts[delimiter] += line.count(delimiter)
        return delimiter_counts.most_common(1)[0][0]

    @staticmethod
    def normalize_quotes(field):
        if field.startswith(("'", '"')) and field.endswith(("'", '"')):
            return f'"{field[1:-1]}"'
        return f'"{field}"'

    @staticmethod
    def parse_date(date_str):
        try:
            return parser.parse(date_str).strftime('%Y-%m-%d')
        except (ValueError, OverflowError):
            return date_str

    @staticmethod
    def parse_number(number_str):
        number_str = re.sub(r'[^\d.,-]', '', number_str)
        if ',' in number_str and '.' in number_str:
            if number_str.rindex(',') > number_str.rindex('.'):
                number_str = number_str.replace('.', '').replace(',', '.')
            else:
                number_str = number_str.replace(',', '')
        return number_str

    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        # Detect delimiter
        delimiter = mistral_largeRound0Solution.detect_delimiter(csv_data)

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
                    field = mistral_largeRound0Solution.normalize_quotes(field)
                    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', field):
                        field = mistral_largeRound0Solution.parse_date(field)
                    elif re.match(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', field):
                        field = mistral_largeRound0Solution.parse_number(field)
                    normalized_row.append(field)
            normalized_rows.append(normalized_row)

        # Write normalized CSV data
        output = StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(normalized_rows)

        return output.getvalue()

    @staticmethod
    def solve(input_text):
        return mistral_largeRound0Solution.normalize_csv(input_text)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import io
from collections import Counter
from dateutil import parser as date_parser
import chardet

class o3_miniRound1Solution:
    @staticmethod
    def solve(input_text):
        return o3_miniRound1Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.
        
        The function addresses various common CSV issues:
          • Inconsistent (or mixed) delimiters by detecting the most common delimiter
          • Mixed quote styles – it removes outer single or double quotes
          • Inconsistent date formats – uses dateutil to parse and convert to ISO (YYYY-MM-DD)
          • Inconsistent numeric formats – converts numbers (US and European style) to standard numeric format
          • Trims extra whitespace; replaces missing/null values with empty strings
          • Normalizes header names to lowercase and underscores
          • Handles simple encoding issues and BOM

        Args:
            csv_data: String containing messy CSV data.
        
        Returns:
            A string with cleaned, normalized CSV data.
        """
        # --- 1. Handle encoding issues and BOM removal ---
        if isinstance(csv_data, bytes):
            detected = chardet.detect(csv_data)
            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        # Remove any BOM and normalize line endings to '\n'
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into nonempty lines (stripping leading/trailing whitespace)
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # --- 2. Detect the delimiter ---
        delimiter = o3_miniRound1Solution.detect_delimiter(lines)
        
        # --- 3. Parse CSV using the detected delimiter ---
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter, skipinitialspace=True)
        rows = list(reader)
        if not rows:
            return ""
        
        # --- 4. Normalize header names (assume first row is header) ---
        header = [o3_miniRound1Solution.normalize_column_name(col) for col in rows[0]]
        normalized_rows = [header]
        
        # --- 5. Process each data row ---
        for row in rows[1:]:
            # Skip rows that are completely empty (after stripping)
            if all(not cell.strip() for cell in row):
                continue
            
            # If row field count is not the same as header, adjust:
            row = o3_miniRound1Solution.adjust_row_length(row, len(header))
            
            new_row = []
            for cell in row:
                cleaned = o3_miniRound1Solution.clean_field(cell)
                norm_val = o3_miniRound1Solution.normalize_value(cleaned)
                new_row.append(norm_val)
            normalized_rows.append(new_row)
        
        # --- 6. Write the normalized output to CSV format (always use comma delimiter) ---
        out_io = io.StringIO()
        # Using QUOTE_MINIMAL so that only fields needing quotes are quoted.
        writer = csv.writer(out_io, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        return out_io.getvalue()

    @staticmethod
    def detect_delimiter(lines):
        """
        Detect the most likely delimiter (from a set of common delimiters)
        by counting occurrences outside of quotes. 
        """
        common_delimiters = [',', ';', '\t', '|']
        # For each candidate, count occurrences outside of quoted parts in each line.
        delimiter_counts = {d: [] for d in common_delimiters}
        for line in lines:
            for delim in common_delimiters:
                count = o3_miniRound1Solution.count_delimiters_outside_quotes(line, delim)
                delimiter_counts[delim].append(count)
        # Prefer the delimiter with a consistent nonzero count across lines
        best_delim = ','
        best_score = 0
        for delim, counts in delimiter_counts.items():
            # Only consider lines that actually have nonzero counts
            non_zero = [c for c in counts if c > 0]
            if not non_zero:
                continue
            # Use consistency: most common count
            most_common = Counter(non_zero).most_common(1)[0][1]
            if most_common > best_score:
                best_score = most_common
                best_delim = delim
        return best_delim

    @staticmethod
    def count_delimiters_outside_quotes(line, delimiter):
        """
        Count instances of the delimiter outside of any quoted regions.
        Quotes may be single or double.
        """
        count = 0
        in_quotes = False
        quote_char = None
        for char in line:
            # Toggle the in_quotes flag if we encounter an unescaped quote
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
    def normalize_column_name(name: str) -> str:
        """Normalize a column name to lowercase, trim it, 
           replace spaces with underscores and remove extra non-alphanumeric characters."""
        name = name.strip().lower()
        name = re.sub(r'\s+', '_', name)
        name = re.sub(r'[^\w]', '', name)
        return name

    @staticmethod
    def adjust_row_length(row, expected_length):
        """
        Adjust row length to match expected header length.
        If row has too few values, pad with empty strings.
        If too many, try to combine extra splits if they are likely due to erroneous splits
        or else truncate.
        """
        if len(row) == expected_length:
            return row
        elif len(row) < expected_length:
            return row + [""] * (expected_length - len(row))
        else:
            # Attempt to combine fields if extra fields might be the result of splitting on a delimiter inside quoted text
            new_row = []
            i = 0
            while i < len(row) and len(new_row) < expected_length - 1:
                field = row[i]
                # If the field starts with a quote but does not end with one, combine with next fields.
                if (field.startswith('"') and not field.endswith('"')) or (field.startswith("'") and not field.endswith("'")):
                    combined = field
                    i += 1
                    while i < len(row) and not row[i].endswith(field[0]):
                        combined += " " + row[i]
                        i += 1
                    if i < len(row):
                        combined += " " + row[i]
                    new_row.append(combined)
                    i += 1
                else:
                    new_row.append(field)
                    i += 1
            # Append any remaining fields to the last column (if any)
            if i < len(row):
                new_row.append(" ".join(row[i:]))
            # Ensure correct length
            if len(new_row) < expected_length:
                new_row += [""] * (expected_length - len(new_row))
            elif len(new_row) > expected_length:
                new_row = new_row[:expected_length]
            return new_row

    @staticmethod
    def clean_field(field: str) -> str:
        """
        Remove extraneous whitespace and remove matching outer quotes 
        (both single and double), then return the cleaned field.
        """
        field = field.strip()
        if len(field) >= 2:
            if (field[0] == field[-1]) and field[0] in ('"', "'"):
                field = field[1:-1].strip()
        return field

    @staticmethod
    def normalize_value(value: str) -> str:
        """
        Normalize an individual field value:
          1. Replace missing/null-like values with empty string.
          2. Normalize booleans ("yes"/"true" => "true", "no"/"false" => "false").
          3. If it looks like a date, try to parse and convert to ISO YYYY-MM-DD.
          4. If it looks like a number, clean up currency and thousand separators.
          5. Otherwise, return the cleaned string.
        """
        # Handle missing/null values.
        if value == "" or value.lower() in {'null', 'none', 'na', 'n/a', '-'}:
            return ""
        
        # Normalize booleans
        low_val = value.lower()
        if low_val in {'yes', 'true', 'y', '1'}:
            return "true"
        if low_val in {'no', 'false', 'n', '0'}:
            return "false"
        
        # Attempt date normalization if there is any alphabetic character or common date separator.
        if re.search(r'[a-zA-Z]', value) or re.search(r'[\-/]', value):
            date_normalized = o3_miniRound1Solution.try_normalize_date(value)
            if date_normalized is not None:
                return date_normalized

        # Attempt to normalize numbers if the text contains digits.
        if re.search(r'\d', value):
            number_normalized = o3_miniRound1Solution.try_normalize_number(value)
            if number_normalized is not None:
                return number_normalized

        return value

    @staticmethod
    def try_normalize_date(text: str) -> str or None:
        """
        Try to parse a string as a date using dateutil's parser (with fuzzy matching)
        and if successful, return the date in ISO format (YYYY-MM-DD). Otherwise, return None.
        """
        try:
            # Remove ordinal suffixes like "3rd", "2nd", etc.
            cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)
            dt = date_parser.parse(cleaned, fuzzy=True)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None

    @staticmethod
    def try_normalize_number(text: str) -> str or None:
        """
        Try to convert a numeric string that may include various currency symbols,
        thousand separators, or European-style decimals into a standard numeric format.
        Returns the numeric string if successful, else None.
        """
        # Remove common currency symbols and extra whitespace.
        cleaned = re.sub(r'[\$€£]', '', text).strip()
        # First, try assuming commas are thousand separators (US style).
        try:
            num = float(cleaned.replace(',', ''))
            # Return as simple float string (or integer if exact)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except Exception:
            pass

        # Second, try European convention: dots as thousand separators and comma as decimal separator.
        try:
            cleaned2 = cleaned.replace('.', '').replace(',', '.')
            num = float(cleaned2)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except Exception:
            return None
import csv
import re
import io
from dateutil import parser as date_parser
from collections import Counter

class gpt4oRound1Solution:
    @staticmethod
    def solve(input_text):
        return gpt4oRound1Solution.normalize_csv(input_text)
    
    @staticmethod
    def normalize_csv(csv_data: str) -> str:
        """
        Clean and normalize messy CSV data.
        
        Args:
            csv_data: String containing messy CSV data
            
        Returns:
            String containing cleaned, normalized CSV data
        """
        # Detect and decode character encoding issues
        if isinstance(csv_data, bytes):
            csv_data = csv_data.decode('utf-8', errors='replace')
            
        # Remove BOM if present
        csv_data = csv_data.lstrip('\ufeff')
        
        # Normalize line endings
        csv_data = csv_data.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove completely empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # Detect delimiter
        delimiter = gpt4oRound1Solution.detect_delimiter(lines)
        
        # Parse CSV with the detected delimiter
        reader = csv.reader(io.StringIO('\n'.join(lines)), delimiter=delimiter, quotechar='"')
        rows = list(reader)
        if not rows:
            return ""
        
        # Normalize header row
        headers = [gpt4oRound1Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        cleaned_rows = [headers]
        for row in rows[1:]:
            # Skip rows that are completely empty
            if all(not cell.strip() for cell in row):
                continue
            
            # Ensure row length matches header length
            row = gpt4oRound1Solution.adjust_row_length(row, len(headers))
            
            cleaned_row = [gpt4oRound1Solution.clean_field(cell, headers[i]) for i, cell in enumerate(row)]
            cleaned_rows.append(cleaned_row)
        
        # Write the cleaned data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(cleaned_rows)
        
        return output.getvalue()

    @staticmethod
    def detect_delimiter(lines):
        """Detect the most likely delimiter in the CSV data."""
        delimiter_candidates = [',', ';', '\t', '|']
        delimiter_counts = Counter()

        for line in lines[:5]:  # Analyze the first few lines
            for delimiter in delimiter_candidates:
                delimiter_counts[delimiter] += line.count(delimiter)
        
        # Return the most common delimiter
        return delimiter_counts.most_common(1)[0][0]

    @staticmethod
    def normalize_column_name(name):
        """Normalize column name to lowercase with underscores."""
        name = re.sub(r'[^\w\s]', '_', name)
        name = re.sub(r'\s+', '_', name).lower()
        return name.strip('_')

    @staticmethod
    def adjust_row_length(row, expected_length):
        """Adjust row to match expected length."""
        if len(row) < expected_length:
            return row + [''] * (expected_length - len(row))
        elif len(row) > expected_length:
            return row[:expected_length]
        return row

    @staticmethod
    def clean_field(field, column_name):
        """Clean and normalize a field value based on content and column name."""
        field = field.strip().strip('"').strip("'")
        
        # Handle empty or null-like values
        if not field or field.lower() in ['null', 'none', 'na', 'n/a', '-']:
            return ""
        
        # Normalize boolean values
        if field.lower() in ['true', 'yes', 'y', '1']:
            return "true"
        if field.lower() in ['false', 'no', 'n', '0']:
            return "false"
        
        # Try to detect and normalize date fields
        if gpt4oRound1Solution.looks_like_date_column(column_name):
            try:
                date_obj = date_parser.parse(field, fuzzy=True)
                return date_obj.strftime('%Y-%m-%d')
            except:
                pass

        # Try to detect and normalize numeric fields
        if gpt4oRound1Solution.looks_like_numeric(field):
            return gpt4oRound1Solution.parse_number(field)
        
        return field

    @staticmethod
    def looks_like_date_column(column_name):
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def looks_like_numeric(field):
        """Check if field looks like it contains a number."""
        cleaned = re.sub(r'[^\d.,-]', '', field)
        return bool(re.match(r'^[+-]?\d+([.,]\d+)*$', cleaned))

    @staticmethod
    def parse_number(number_str):
        """Normalize number format."""
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
import csv
import re
import io
import chardet
from dateutil import parser
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any

class claude37Round1Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return claude37Round1Solution.normalize_csv(input_text)
    
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
        delimiter = claude37Round1Solution.detect_delimiter(lines)
        standardized_lines = claude37Round1Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV with detected delimiter
        rows = claude37Round1Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""
        
        # Normalize header row
        header = [claude37Round1Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header
            adjusted_row = claude37Round1Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue
            
            # Clean each field
            normalized_row = [
                claude37Round1Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
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
                count = claude37Round1Solution.count_delimiters_outside_quotes(line, delimiter)
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
        if claude37Round1Solution.looks_like_date(field) or claude37Round1Solution.is_date_column(column_name):
            try:
                date_obj = claude37Round1Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass  # If date parsing fails, continue with other cleaning
        
        # Try to detect and normalize numeric fields
        if claude37Round1Solution.looks_like_numeric(field):
            try:
                return claude37Round1Solution.format_number(field)
            except Exception:
                pass  # If number parsing fails, return cleaned string
        
        return field

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if field value looks like a date."""
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}',  # Month DD, YYYY
            r'\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
        ]
        
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def parse_date(date_str: str) -> Optional[Any]:
        """
        Parse a date string in various formats.
        
        Args:
            date_str: The date string to parse
            
        Returns:
            Parsed date object or None if parsing fails
        """
        # Remove currency symbols and other non-date characters
        date_str = re.sub(r'[$€£]', '', date_str)
        
        # Handle special formats like "May 3rd, 1992"
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        try:
            return parser.parse(date_str, fuzzy=True)
        except Exception:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """
        Check if field looks like it contains a number.
        
        Args:
            field: The field to check
            
        Returns:
            True if field looks like a number, False otherwise
        """
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[$€£¥]', '', field)
        cleaned = re.sub(r'[,\s]', '', cleaned)
        
        # Check for numeric patterns
        return bool(re.match(r'^-?\d*\.?\d+%?$', cleaned))

    @staticmethod
    def format_number(field: str) -> str:
        """
        Format a numeric field consistently.
        
        Args:
            field: The field to format
            
        Returns:
            Formatted number as string
        """
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[$€£¥]', '', field)
        cleaned = re.sub(r'[,\s]', '', cleaned)
        
        # Handle percentages
        is_percentage = cleaned.endswith('%')
        if is_percentage:
            cleaned = cleaned[:-1]
        
        # Parse as float
        try:
            num = float(cleaned)
            
            # Convert back to percentage if needed
            if is_percentage:
                return f"{num:.2f}%"
                
            # Handle integers vs decimals
            if num.is_integer():
                return str(int(num))
            else:
                # Format with up to 6 decimal places, removing trailing zeros
                formatted = f"{num:.6f}".rstrip('0').rstrip('.')
                return formatted
        except ValueError:
            return field
import csv
import re
import io
import chardet
from dateutil import parser
from collections import Counter

class mistral_largeRound1Solution:
    @staticmethod
    def detect_delimiter(csv_data):
        lines = csv_data.splitlines()
        delimiter_candidates = [',', ';', '\t', '|']
        delimiter_counts = Counter()
        for line in lines[:5]:  # Check the first 5 lines
            for delimiter in delimiter_candidates:
                delimiter_counts[delimiter] += line.count(delimiter)
        return delimiter_counts.most_common(1)[0][0]

    @staticmethod
    def normalize_quotes(field):
        if field.startswith(("'", '"')) and field.endswith(("'", '"')):
            return f'"{field[1:-1]}"'
        return f'"{field}"'

    @staticmethod
    def parse_date(date_str):
        try:
            return parser.parse(date_str).strftime('%Y-%m-%d')
        except (ValueError, OverflowError):
            return date_str

    @staticmethod
    def parse_number(number_str):
        number_str = re.sub(r'[^\d.,-]', '', number_str)
        if ',' in number_str and '.' in number_str:
            if number_str.rindex(',') > number_str.rindex('.'):
                number_str = number_str.replace('.', '').replace(',', '.')
            else:
                number_str = number_str.replace(',', '')
        return number_str

    @staticmethod
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
        delimiter = mistral_largeRound1Solution.detect_delimiter(csv_data)

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
                    field = mistral_largeRound1Solution.normalize_quotes(field)
                    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', field):
                        field = mistral_largeRound1Solution.parse_date(field)
                    elif re.match(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', field):
                        field = mistral_largeRound1Solution.parse_number(field)
                    normalized_row.append(field)
            normalized_rows.append(normalized_row)

        # Write normalized CSV data
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(normalized_rows)

        return output.getvalue()

    @staticmethod
    def solve(input_text):
        return mistral_largeRound1Solution.normalize_csv(input_text)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import io
import chardet
from collections import Counter
from dateutil import parser as date_parser

class o3_miniRound2Solution:
    @staticmethod
    def solve(input_text):
        return o3_miniRound2Solution.normalize_csv(input_text)
    
    @staticmethod
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
              • Converting dates (using dateutil's fuzzy parser into ISO YYYY-MM-DD)
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
        delimiter = o3_miniRound2Solution.detect_delimiter(lines)
        
        # --- 4. Parse CSV using the detected delimiter ---
        # Use csv.reader to allow proper handling of quotes (our cleaning will remove extraneous quotes later)
        reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter, skipinitialspace=True)
        rows = list(reader)
        if not rows:
            return ""
        
        # --- 5. Normalize header row ---
        header = [o3_miniRound2Solution.normalize_column_name(col) for col in rows[0]]
        
        # --- 6. Process each row (skip fully empty rows, adjust row length) ---
        cleaned_rows = [header]  # first row is header
        for row in rows[1:]:
            # Skip a row if all fields are blank after stripping
            if all(not cell.strip() for cell in row):
                continue
            
            row = o3_miniRound2Solution.adjust_row_length(row, len(header))
            
            # Clean each field based on its content (and if needed, by its column name)
            new_row = []
            for idx, field in enumerate(row):
                new_field = o3_miniRound2Solution.clean_field(field)
                new_field = o3_miniRound2Solution.normalize_value(new_field, header[idx] if idx < len(header) else f'column_{idx}')
                new_row.append(new_field)
            cleaned_rows.append(new_row)
        
        # --- 7. Write normalized rows back to CSV (use comma as delimiter) ---
        output = io.StringIO()
        writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(cleaned_rows)
        
        return output.getvalue()


    # ---------------- Helper Functions ----------------

    @staticmethod
    def detect_delimiter(lines):
        """
        Detect the most likely delimiter by counting occurrences outside quoted regions.
        Considers common delimiters: comma, semicolon, tab, and pipe.
        """
        candidates = [',', ';', '\t', '|']
        delim_counts = {d: 0 for d in candidates}
        for line in lines[:5]:  # analyze first few lines
            for d in candidates:
                delim_counts[d] += o3_miniRound2Solution.count_outside_quotes(line, d)
        # Pick the delimiter with the highest total count
        return max(delim_counts, key=delim_counts.get) if max(delim_counts.values()) > 0 else ','

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def clean_field(field: str) -> str:
        """
        Clean extraneous whitespace and remove matching outer quotes (both single and double).
        """
        field = field.strip()
        if len(field) >= 2 and field[0] == field[-1] and field[0] in ('"', "'"):
            field = field[1:-1].strip()
        return field

    @staticmethod
    def normalize_value(value: str, column_name: str) -> str:
        """
        Normalize an individual cell's value.
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
        if (re.search(r'[A-Za-z]', value) or re.search(r'[\-/]', value)) or o3_miniRound2Solution.is_date_column(column_name):
            date_norm = o3_miniRound2Solution.try_normalize_date(value)
            if date_norm is not None:
                return date_norm
        
        # If the field contains digits, attempt number normalization.
        if re.search(r'\d', value):
            num_norm = o3_miniRound2Solution.try_normalize_number(value)
            if num_norm is not None:
                return num_norm
        
        # Otherwise, return the value as-is.
        return value

    @staticmethod
    def is_date_column(col: str) -> bool:
        """Heuristic to see if a column name suggests date values."""
        return any(ind in col.lower() for ind in ['date', 'day', 'month', 'year', 'time', 'birth', 'updated', 'created'])

    @staticmethod
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

    @staticmethod
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional

class gpt4oRound2Solution:
    @staticmethod
    def solve(input_text):
        return gpt4oRound2Solution.normalize_csv(input_text)

    @staticmethod
    def normalize_csv(csv_data: str) -> str:
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
        
        # Detect delimiter and normalize lines
        delimiter = gpt4oRound2Solution.detect_delimiter(lines)
        standardized_lines = gpt4oRound2Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV with detected delimiter
        rows = gpt4oRound2Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""
        
        # Normalize header
        header = [gpt4oRound2Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            if not row or all(not cell.strip() for cell in row):
                continue
            adjusted_row = gpt4oRound2Solution.adjust_row_length(row, len(header))
            normalized_row = [gpt4oRound2Solution.clean_field(cell, header[i] if i < len(header) else f"column_{i}") for i, cell in enumerate(adjusted_row)]
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
        
        for line in lines[:10]:
            for delim in common_delimiters:
                count = gpt4oRound2Solution.count_delimiters_outside_quotes(line, delim)
                delimiter_counts[delim].append(count)
        
        best_delimiter = max(delimiter_counts, key=lambda d: sum(delimiter_counts[d]))
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
        
        if gpt4oRound2Solution.looks_like_date(field) or gpt4oRound2Solution.is_date_column(column_name):
            date_obj = gpt4oRound2Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        if gpt4oRound2Solution.looks_like_numeric(field):
            return gpt4oRound2Solution.format_number(field)
        
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
    def parse_date(date_str: str) -> Optional[date_parser.ParserError]:
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Any

class claude37Round2Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return claude37Round2Solution.normalize_csv(input_text)
    
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
            encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        
        # Remove BOM if present and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into non-empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # --- 2. Detect delimiter ---
        delimiter = claude37Round2Solution.detect_delimiter(lines)
        
        # --- 3. Parse CSV data ---
        rows = claude37Round2Solution.parse_csv_with_delimiter(lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # --- 4. Normalize header ---
        header = [claude37Round2Solution.normalize_column_name(col) for col in rows[0]]
        
        # --- 5. Process data rows ---
        cleaned_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header length
            row = claude37Round2Solution.adjust_row_length(row, len(header))
            
            # Clean each field based on content and column name
            cleaned_row = []
            for i, field in enumerate(row):
                column_name = header[i] if i < len(header) else ""
                cleaned_row.append(claude37Round2Solution.clean_field(field, column_name))
            
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
                count = claude37Round2Solution.count_delimiters_outside_quotes(line, delimiter)
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
            csv_text = '\n'.join(lines)
            reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
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
        if claude37Round2Solution.looks_like_date_column(column_name) or claude37Round2Solution.looks_like_date(field):
            try:
                date_obj = claude37Round2Solution.normalize_date(field)
                if date_obj:
                    return date_obj
            except Exception:
                pass  # Fall through to other normalizations if date parsing fails
        
        # Try to detect and normalize numeric fields
        if claude37Round2Solution.looks_like_numeric(field):
            try:
                return claude37Round2Solution.normalize_number(field)
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
        # Remove currency symbols, commas, dots, spaces
        cleaned = re.sub(r'[$€£\s,.]', '', field)
        # Check if the remaining string is a number
        return bool(re.match(r'^[+-]?\d+$', cleaned))

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
            parts = number_str.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely European decimal separator
                number_str = number_str.replace(',', '.')
            else:
                # Likely US thousands separator
                number_str = number_str.replace(',', '')
        
        try:
            # Convert to float and format appropriately
            num = float(number_str)
            if num.is_integer():
                return str(int(num))
            else:
                # Return with appropriate decimal places
                return str(num)
        except ValueError:
            return number_str  # Return original if conversion fails
import csv
import re
import io
import chardet
from dateutil import parser
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

class mistral_largeRound2Solution:
    @staticmethod
    def solve(input_text):
        return mistral_largeRound2Solution.normalize_csv(input_text)
        
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
        delimiter = mistral_largeRound2Solution.detect_delimiter(lines)
        standardized_lines = mistral_largeRound2Solution.standardize_delimiters(lines, delimiter)

        # Parse CSV with detected delimiter
        rows = mistral_largeRound2Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""

        # Normalize header row
        header = [mistral_largeRound2Solution.normalize_column_name(col) for col in rows[0]]

        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue

            # Adjust row length to match header
            adjusted_row = mistral_largeRound2Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue

            # Clean each field
            normalized_row = [
                mistral_largeRound2Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
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
                count = mistral_largeRound2Solution.count_delimiters_outside_quotes(line, delimiter)
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
    def looks_like_date(field: str) -> bool:
        """Check if a field looks like a date."""
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY, etc.
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # YYYY-MM-DD, YYYY/MM/DD
            r'\w+\s+\d{1,2},?\s+\d{4}',        # Month DD, YYYY
            r'\d{1,2}\s+\w+\s+\d{4}'           # DD Month YYYY
        ]
        
        return any(re.match(pattern, field) for pattern in date_patterns)

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if a column name indicates it contains date values."""
        date_indicators = ['date', 'time', 'day', 'month', 'year', 'born', 'created', 'modified', 'updated']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def parse_date(date_str: str):
        """Try to parse a date string into a date object."""
        try:
            return parser.parse(date_str, fuzzy=True)
        except:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if a field looks like a numeric value."""
        # Remove thousand separators and other formatting
        cleaned = re.sub(r'[,$%]', '', field.strip())
        # Match common numeric patterns
        return re.match(r'^-?\d+(\.\d+)?$', cleaned) is not None

    @staticmethod
    def format_number(field: str) -> str:
        """Format a numeric field consistently."""
        # Remove any non-numeric characters except decimal points and negative signs
        cleaned = re.sub(r'[^-0-9.]', '', field.strip())
        # Handle integer vs. float
        if '.' in cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return cleaned
        else:
            try:
                return str(int(cleaned))
            except ValueError:
                return cleaned

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
        if mistral_largeRound2Solution.looks_like_date(field) or mistral_largeRound2Solution.is_date_column(column_name):
            try:
                date_obj = mistral_largeRound2Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass  # If date parsing fails, continue with other cleaning

        # Try to detect and normalize numeric fields
        if mistral_largeRound2Solution.looks_like_numeric(field):
            try:
                return mistral_largeRound2Solution.format_number(field)
            except Exception:
                pass  # If number parsing fails, return cleaned string

        return field
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

class gpt4oRound4Solution:
    @staticmethod
    def solve(input_text: str) -> str:
        return gpt4oRound4Solution.normalize_csv(input_text)
    
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
        delimiter = gpt4oRound4Solution.detect_delimiter(lines)
        standardized_lines = gpt4oRound4Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV with detected delimiter
        rows = gpt4oRound4Solution.parse_csv_rows(standardized_lines)
        if not rows:
            return ""
        
        # Normalize the header
        header = [gpt4oRound4Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            if not row or all(not cell.strip() for cell in row):
                continue
            adjusted_row = gpt4oRound4Solution.adjust_row_length(row, len(header))
            normalized_row = [gpt4oRound4Solution.clean_field(cell, header[i] if i < len(header) else f"column_{i}") for i, cell in enumerate(adjusted_row)]
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
                count = gpt4oRound4Solution.count_delimiters_outside_quotes(line, delim)
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
    def parse_csv_rows(lines: List[str]) -> List[List[str]]:
        csv_text = '\n'.join(lines)
        reader = csv.reader(io.StringIO(csv_text), delimiter=',')
        return [row for row in reader]

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
        
        if gpt4oRound4Solution.looks_like_date(field) or gpt4oRound4Solution.is_date_column(column_name):
            date_obj = gpt4oRound4Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        if gpt4oRound4Solution.looks_like_numeric(field):
            return gpt4oRound4Solution.format_number(field)
        
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Any, Union

class claude37Round4Solution:
    @staticmethod
    def solve(input_text):
        return claude37Round4Solution.normalize_csv(input_text)

    @staticmethod
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
            encoding = detected.get('encoding') or 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        
        # Remove BOM and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into non-empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # Detect delimiter and standardize lines
        delimiter = claude37Round4Solution.detect_delimiter(lines)
        standardized_lines = claude37Round4Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV rows
        rows = claude37Round4Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # Normalize header row
        header = [claude37Round4Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header
            adjusted_row = claude37Round4Solution.adjust_row_length(row, len(header))
            
            # Clean each field based on content and column name
            cleaned_row = []
            for i, field in enumerate(adjusted_row):
                column_name = header[i] if i < len(header) else f"column_{i}"
                cleaned_row.append(claude37Round4Solution.clean_field(field, column_name))
            
            normalized_rows.append(cleaned_row)
        
        # Write normalized data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        
        return output.getvalue()

    @staticmethod
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
                count = claude37Round4Solution.count_delimiters_outside_quotes(line, delimiter)
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

    @staticmethod
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
                standardized_lines.append(','.join(f'"{f}"' if ',' in f else f for f in fields))
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
        Parse CSV lines into rows, handling quotes properly.
        
        Args:
            lines: List of standardized CSV lines
            detected_delimiter: The detected delimiter
            
        Returns:
            List of parsed CSV rows
        """
        csv_text = '\n'.join(lines)
        
        try:
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
                    elif char == ',' and not in_quotes:
                        fields.append(current)
                        current = ""
                    else:
                        current += char
                
                if current or len(fields) > 0:
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
        
        # Convert to lowercase and replace non-alphanumeric with underscores
        name = re.sub(r'[^\w\s]', '_', name.lower())
        name = re.sub(r'\s+', '_', name)
        
        # Remove consecutive and trailing underscores
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name if name else "column"  # Ensure we don't return empty string

    @staticmethod
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
    def clean_field(field: str, column_name: str) -> str:
        """
        Clean and normalize a field value based on content and column name.
        
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
        if claude37Round4Solution.is_date_column(column_name) or claude37Round4Solution.looks_like_date(field):
            date_obj = claude37Round4Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        # Try to detect and normalize numeric fields
        if claude37Round4Solution.looks_like_numeric(field):
            normalized_number = claude37Round4Solution.format_number(field)
            if normalized_number is not None:
                return normalized_number
        
        return field

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if field value looks like a date."""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # Month DD, YYYY
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
        ]
        
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def parse_date(date_str: str) -> Optional[Any]:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Datetime object or None if parsing fails
        """
        try:
            # Remove ordinal suffixes (1st, 2nd, 3rd, etc.)
            cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
            return date_parser.parse(cleaned, fuzzy=True)
        except Exception:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if field looks like it contains a number."""
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£\s]', '', field)
        # Check if it has digits and optionally decimal/thousand separators
        return bool(re.match(r'^[+-]?[\d,.]+$', cleaned))

    @staticmethod
    def format_number(number_str: str) -> Optional[str]:
        """
        Normalize number format by handling different separators.
        
        Args:
            number_str: Number string to normalize
            
        Returns:
            Normalized number string or None if parsing fails
        """
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£\s]', '', number_str)
        
        # Handle different number formats
        if ',' in cleaned and '.' in cleaned:
            # Determine which is the decimal separator
            if cleaned.rindex(',') > cleaned.rindex('.'):
                # European format (1.234,56)
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format (1,234.56)
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be either 1,234 (US) or 1,23 (European)
            # If comma is followed by exactly 2 digits, likely European decimal
            if re.search(r',\d{2}$', cleaned):
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        
        try:
            num = float(cleaned)
            # Format integers without decimal point, floats with up to 6 digits
            if num.is_integer():
                return str(int(num))
            else:
                # Remove trailing zeros
                return str(num).rstrip('0').rstrip('.') if '.' in str(num) else str(num)
        except ValueError:
            return None
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict

class mistral_largeRound4Solution:
    @staticmethod
    def solve(input_text):
        return mistral_largeRound4Solution.normalize_csv(input_text)
    
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
        delimiter = mistral_largeRound4Solution.detect_delimiter(lines)
        standardized_lines = mistral_largeRound4Solution.standardize_delimiters(lines, delimiter)

        # Parse CSV with detected delimiter
        rows = mistral_largeRound4Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""

        # Normalize header row
        header = [mistral_largeRound4Solution.normalize_column_name(col) for col in rows[0]]

        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue

            # Adjust row length to match header
            adjusted_row = mistral_largeRound4Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue

            # Clean each field
            normalized_row = [
                mistral_largeRound4Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
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
                count = mistral_largeRound4Solution.count_delimiters_outside_quotes(line, delimiter)
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
        if mistral_largeRound4Solution.looks_like_date(field) or mistral_largeRound4Solution.is_date_column(column_name):
            try:
                date_obj = mistral_largeRound4Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass  # If date parsing fails, continue with other cleaning

        # Try to detect and normalize numeric fields
        if mistral_largeRound4Solution.looks_like_numeric(field):
            try:
                return mistral_largeRound4Solution.format_number(field)
            except Exception:
                pass  # If number parsing fails, return cleaned string

        return field
        
    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if a field looks like a date."""
        # Common date patterns
        date_patterns = [
            r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}',    # YYYY/MM/DD
            r'\d{1,2}[ -][A-Za-z]{3,9}[ -]\d{2,4}',  # DD Month YYYY
            r'[A-Za-z]{3,9}[ -]\d{1,2}[ -,]\d{2,4}'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, field):
                return True
        return False

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'birthday', 'dob', 'created', 'updated']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def parse_date(date_str: str):
        """Parse date string using dateutil parser."""
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if a field looks like a number."""
        # Strip currency symbols and other common number formatting
        cleaned = re.sub(r'[,$%€£¥\s]', '', field)
        # Check if it's a valid number representation
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    @staticmethod
    def format_number(field: str) -> str:
        """Format numeric field consistently."""
        # Strip currency symbols and other formatting
        cleaned = re.sub(r'[,$%€£¥\s]', '', field)
        
        # Convert to float and format
        value = float(cleaned)
        if value.is_integer():
            return str(int(value))
        else:
            # Use a reasonable precision
            return str(round(value, 6)).rstrip('0').rstrip('.') if '.' in str(round(value, 6)) else str(int(value))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union, Any

class o3_miniRound5Solution:
    @staticmethod
    def solve(input_text):
        return o3_miniRound5Solution.normalize_csv(input_text)
    
    @staticmethod
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
        delimiter = o3_miniRound5Solution.detect_delimiter(lines)
        
        # 4. Standardize the delimiters and quote characters.
        standardized_lines = o3_miniRound5Solution.standardize_delimiters(lines, delimiter)
        
        # 5. Parse CSV rows.
        rows = o3_miniRound5Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # 6. Normalize header row.
        header = [o3_miniRound5Solution.normalize_column_name(col) for col in rows[0]]
        
        # 7. Process and clean each row.
        normalized_rows = [header]
        for row in rows[1:]:
            if not row or all(not cell.strip() for cell in row):
                continue  # skip completely blank rows
            
            adjusted = o3_miniRound5Solution.adjust_row_length(row, len(header))
            if not adjusted:
                continue
            
            cleaned = [o3_miniRound5Solution.clean_field(adjusted[i], header[i] if i < len(header) else f"column_{i}") 
                          for i in range(len(adjusted))]
            normalized_rows.append(cleaned)
        
        # 8. Write back to CSV string.
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        
        return output.getvalue()

    @staticmethod
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
                cnt = o3_miniRound5Solution.count_delimiters_outside_quotes(line, delim)
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        if o3_miniRound5Solution.is_date_column(column_name) or o3_miniRound5Solution.looks_like_date(field):
            dt = o3_miniRound5Solution.parse_date(field)
            if dt:
                return dt.strftime('%Y-%m-%d')
        
        # If the field looks numeric, try to format it.
        if o3_miniRound5Solution.looks_like_numeric(field):
            num = o3_miniRound5Solution.format_number(field)
            if num is not None:
                return num
        
        return field

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union

class gpt4oRound5Solution:
    @staticmethod
    def solve(input_text):
        return gpt4oRound5Solution.normalize_csv(input_text)
        
    @staticmethod
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
        delimiter = gpt4oRound5Solution.detect_delimiter(lines)
        standardized_lines = gpt4oRound5Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV rows
        rows = gpt4oRound5Solution.parse_csv_rows(standardized_lines)
        if not rows or len(rows) < 1:
            return ""
        
        # Normalize header row
        header = [gpt4oRound5Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header
            adjusted_row = gpt4oRound5Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue
            
            # Clean each field based on content and column name
            cleaned_row = [
                gpt4oRound5Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
                for i, field in enumerate(adjusted_row)
            ]
            
            normalized_rows.append(cleaned_row)
        
        # Write normalized data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        
        return output.getvalue()

    @staticmethod
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
                count = gpt4oRound5Solution.count_delimiters_outside_quotes(line, delimiter)
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

    @staticmethod
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
        
        # Convert to lowercase and replace non-alphanumeric with underscores
        name = re.sub(r'[^\w\s]', '_', name.lower())
        name = re.sub(r'\s+', '_', name)
        
        # Remove consecutive and trailing underscores
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name if name else "column"  # Ensure we don't return empty string

    @staticmethod
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

    @staticmethod
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
        if gpt4oRound5Solution.is_date_column(column_name) or gpt4oRound5Solution.looks_like_date(field):
            date_obj = gpt4oRound5Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        # Try to detect and normalize numeric fields
        if gpt4oRound5Solution.looks_like_numeric(field):
            normalized_number = gpt4oRound5Solution.format_number(field)
            if normalized_number is not None:
                return normalized_number
        
        return field

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if field value looks like a date."""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # Month DD, YYYY
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
        ]
        
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def parse_date(date_str: str) -> Optional[date_parser.parser]:
        """Attempt to parse a date string to a datetime object."""
        try:
            # Remove ordinal suffixes (e.g., 1st, 2nd, 3rd)
            cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
            return date_parser.parse(cleaned, fuzzy=True)
        except (date_parser.ParserError, ValueError):
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if field appears to be numeric."""
        cleaned = re.sub(r'[$€£\s]', '', field)
        return bool(re.search(r'\d', cleaned))

    @staticmethod
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
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union, Any

class claude37Round5Solution:
    @staticmethod
    def solve(input_text: Union[str, bytes]) -> str:
        return claude37Round5Solution.normalize_csv(input_text)
    
    @staticmethod
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
            encoding = detected.get('encoding') or 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')
        
        # Remove BOM and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into non-empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""
        
        # Detect delimiter and standardize lines
        delimiter = claude37Round5Solution.detect_delimiter(lines)
        standardized_lines = claude37Round5Solution.standardize_delimiters(lines, delimiter)
        
        # Parse CSV rows
        rows = claude37Round5Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows or len(rows) < 1:
            return ""
        
        # Normalize header row
        header = [claude37Round5Solution.normalize_column_name(col) for col in rows[0]]
        
        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue
            
            # Adjust row length to match header
            adjusted_row = claude37Round5Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue
            
            # Clean each field based on content and column name
            cleaned_row = []
            for i, field in enumerate(adjusted_row):
                column_name = header[i] if i < len(header) else f"column_{i}"
                cleaned_row.append(claude37Round5Solution.clean_field(field, column_name))
            
            normalized_rows.append(cleaned_row)
        
        # Write normalized data back to CSV
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(normalized_rows)
        
        return output.getvalue()

    @staticmethod
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
                count = claude37Round5Solution.count_delimiters_outside_quotes(line, delimiter)
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

    @staticmethod
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
                standardized_lines.append(','.join(f'"{f}"' if ',' in f else f for f in fields))
                continue
                
            new_line = ""
            in_quotes = False
            quote_char = None
            escaped = False
            
            for char in line:
                if escaped:
                    escaped = False
                    new_line += char
                    continue
                    
                if char == '\\':
                    escaped = True
                    new_line += char
                    continue
                    
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
        Parse CSV lines into rows, handling quotes properly.
        
        Args:
            lines: List of standardized CSV lines
            detected_delimiter: The detected delimiter
            
        Returns:
            List of parsed CSV rows
        """
        csv_text = '\n'.join(lines)
        
        try:
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
                escaped = False
                
                for char in line:
                    if escaped:
                        escaped = False
                        current += char
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
                        else:
                            current += char
                    elif char == ',' and not in_quotes:
                        fields.append(current)
                        current = ""
                    else:
                        current += char
                
                if current or len(fields) > 0:
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
        
        # Convert to lowercase and replace non-alphanumeric with underscores
        name = re.sub(r'[^\w\s]', '_', name.lower())
        name = re.sub(r'\s+', '_', name)
        
        # Remove consecutive and trailing underscores
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        
        return name if name else "column"  # Ensure we don't return empty string

    @staticmethod
    def adjust_row_length(row: List[str], expected_length: int) -> Optional[List[str]]:
        """
        Adjust row to match expected length by padding or combining fields.
        
        Args:
            row: Row to adjust
            expected_length: Expected number of fields
            
        Returns:
            Adjusted row or None if adjustment is not possible
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
    def clean_field(field: str, column_name: str) -> str:
        """
        Clean and normalize a field value based on content and column name.
        
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
        if claude37Round5Solution.is_date_column(column_name) or claude37Round5Solution.looks_like_date(field):
            date_obj = claude37Round5Solution.parse_date(field)
            if date_obj:
                return date_obj.strftime('%Y-%m-%d')
        
        # Try to detect and normalize numeric fields
        if claude37Round5Solution.looks_like_numeric(field):
            normalized_number = claude37Round5Solution.format_number(field)
            if normalized_number is not None:
                return normalized_number
        
        return field

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if column name suggests it contains dates."""
        date_indicators = ['date', 'day', 'month', 'year', 'time', 'updated', 'created', 'birth']
        return any(indicator in column_name.lower() for indicator in date_indicators)

    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if field value looks like a date."""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}',  # Month DD, YYYY
            r'\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{2,4}'   # DD Month YYYY
        ]
        
        return any(re.search(pattern, field) for pattern in date_patterns)

    @staticmethod
    def parse_date(date_str: str) -> Optional[Any]:
        """
        Parse date string to datetime object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Datetime object or None if parsing fails
        """
        try:
            # Remove ordinal suffixes (1st, 2nd, 3rd, etc.)
            cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
            return date_parser.parse(cleaned, fuzzy=True)
        except Exception:
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if field looks like it contains a number."""
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£\s]', '', field)
        # Check if it has digits and optionally decimal/thousand separators
        return bool(re.match(r'^[+-]?[\d,.]+$', cleaned))

    @staticmethod
    def format_number(number_str: str) -> Optional[str]:
        """
        Normalize number format by handling different separators.
        
        Args:
            number_str: Number string to normalize
            
        Returns:
            Normalized number string or None if parsing fails
        """
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£\s]', '', number_str)
        
        try:
            # Handle different number formats
            if ',' in cleaned and '.' in cleaned:
                # Determine which is the decimal separator
                if cleaned.rindex(',') > cleaned.rindex('.'):
                    # European format (1.234,56)
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    # US format (1,234.56)
                    cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # Could be either 1,234 (US) or 1,23 (European)
                # If comma is followed by exactly 2 digits at the end, likely European decimal
                if re.search(r',\d{2}$', cleaned):
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
            
            # Try to convert to float to standardize format
            float_value = float(cleaned)
            
            # Format integers without decimals, otherwise use exactly 2 decimal places
            if float_value.is_integer():
                return str(int(float_value))
            else:
                return str(float_value)
        except Exception:
            return number_str
import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List, Optional, Dict, Union
from datetime import datetime

class mistral_largeRound5Solution:
    @staticmethod
    def solve(input_text):
        return mistral_largeRound5Solution.normalize_csv(input_text)
        
    @staticmethod
    def normalize_csv(csv_data: Union[str, bytes]) -> str:
        """
        Clean and normalize messy CSV data.

        Args:
            csv_data: String or bytes containing messy CSV data

        Returns:
            String containing cleaned, normalized CSV data
        """
        # Handle character encoding issues
        if isinstance(csv_data, bytes):
            detected = chardet.detect(csv_data)
            encoding = detected.get('encoding') or 'utf-8'
            csv_data = csv_data.decode(encoding, errors='replace')

        # Remove BOM if present and normalize line endings
        csv_data = csv_data.lstrip('\ufeff').replace('\r\n', '\n').replace('\r', '\n')

        # Split into lines and remove empty lines
        lines = [line for line in csv_data.split('\n') if line.strip()]
        if not lines:
            return ""

        # Detect and standardize delimiters
        delimiter = mistral_largeRound5Solution.detect_delimiter(lines)
        standardized_lines = mistral_largeRound5Solution.standardize_delimiters(lines, delimiter)

        # Parse CSV with detected delimiter
        rows = mistral_largeRound5Solution.parse_csv_rows(standardized_lines, delimiter)
        if not rows:
            return ""

        # Normalize header row
        header = [mistral_largeRound5Solution.normalize_column_name(col) for col in rows[0]]

        # Process data rows
        normalized_rows = [header]
        for row in rows[1:]:
            # Skip completely empty rows
            if not row or all(not cell.strip() for cell in row):
                continue

            # Adjust row length to match header
            adjusted_row = mistral_largeRound5Solution.adjust_row_length(row, len(header))
            if not adjusted_row:
                continue

            # Clean each field
            normalized_row = [
                mistral_largeRound5Solution.clean_field(field, header[i] if i < len(header) else f"column_{i}")
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
                count = mistral_largeRound5Solution.count_delimiters_outside_quotes(line, delimiter)
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
        if mistral_largeRound5Solution.looks_like_date(field) or mistral_largeRound5Solution.is_date_column(column_name):
            try:
                date_obj = mistral_largeRound5Solution.parse_date(field)
                if date_obj:
                    return date_obj.strftime('%Y-%m-%d')
            except Exception:
                pass  # If date parsing fails, continue with other cleaning

        # Try to detect and normalize numeric fields
        if mistral_largeRound5Solution.looks_like_numeric(field):
            try:
                return mistral_largeRound5Solution.format_number(field)
            except Exception:
                pass  # If number parsing fails, return original field

        return field
    
    @staticmethod
    def looks_like_date(field: str) -> bool:
        """Check if a field value looks like a date."""
        # Date patterns: YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, etc.
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',  # MM-DD-YY or DD-MM-YY
            r'[A-Za-z]{3,} \d{1,2},? \d{4}',  # Month DD, YYYY
            r'\d{1,2} [A-Za-z]{3,},? \d{4}'   # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, field):
                return True
        return False

    @staticmethod
    def is_date_column(column_name: str) -> bool:
        """Check if a column name suggests it contains dates."""
        date_keywords = ['date', 'day', 'month', 'year', 'dob', 'created', 'updated', 
                         'timestamp', 'time', 'birthday', 'birth']
        return any(keyword in column_name.lower() for keyword in date_keywords)

    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        try:
            return date_parser.parse(date_str, fuzzy=True)
        except (ValueError, OverflowError):
            return None

    @staticmethod
    def looks_like_numeric(field: str) -> bool:
        """Check if a field value looks like a number."""
        # Remove common number formatting characters
        cleaned = re.sub(r'[,$%()]', '', field).strip()
        
        # Check if it's a decimal number
        if re.fullmatch(r'-?\d+(\.\d+)?', cleaned):
            return True
            
        # Check for scientific notation
        if re.fullmatch(r'-?\d+(\.\d+)?[eE][+-]?\d+', cleaned):
            return True
            
        return False

    @staticmethod
    def format_number(field: str) -> str:
        """Format numeric values consistently."""
        # Remove currency symbols, commas, parentheses (negative numbers)
        cleaned = re.sub(r'[,$%]', '', field)
        cleaned = cleaned.replace('(', '-').replace(')', '')
        
        # Convert to float and back to string to normalize
        try:
            num = float(cleaned)
            # Format integers without decimal point
            if num.is_integer():
                return str(int(num))
            else:
                # Remove trailing zeros but keep significant decimals
                return str(num).rstrip('0').rstrip('.') if '.' in str(num) else str(num)
        except ValueError:
            return field

def main():
    """Test all solutions and collect metrics"""
    parser = argparse.ArgumentParser(description="Test LLM tournament solutions")
    parser.add_argument("--input", type=str, required=True, help="Input file to test on")
    parser.add_argument("--output-dir", type=str, default="output_results_for_each_round_and_model",
                    help="Directory for results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read the input file
    with open(args.input, "r", encoding="utf-8") as f:
        input_text = f.read()
    
    def count_lines(text: str) -> int:
        """Count the number of lines in a text"""
        return len(text.splitlines())

    def count_chars(text: str) -> int:
        """Count the number of characters in a text"""
        return len(text)        
    
    # Get input file metrics
    input_lines = count_lines(input_text)
    input_chars = count_chars(input_text)
    print(f"Input file: {args.input}")
    print(f"Input lines: {input_lines}")
    print(f"Input chars: {input_chars}")
    
    # List of all solution classes
    solution_classes = [
        o3_miniRound0Solution, gpt4oRound0Solution, claude37Round0Solution, mistral_largeRound0Solution, o3_miniRound1Solution, gpt4oRound1Solution, claude37Round1Solution, mistral_largeRound1Solution, o3_miniRound2Solution, gpt4oRound2Solution, claude37Round2Solution, mistral_largeRound2Solution, o3_miniRound3Solution, gpt4oRound3Solution, claude37Round3Solution, mistral_largeRound3Solution, o3_miniRound4Solution, gpt4oRound4Solution, claude37Round4Solution, mistral_largeRound4Solution, o3_miniRound5Solution, gpt4oRound5Solution, claude37Round5Solution, mistral_largeRound5Solution
    ]
    
    # Test each solution
    metrics = []
    
    for solution_class in solution_classes:
        class_name = solution_class.__name__
        print(f"\nTesting {class_name}...")

        # Extract model name and round number
        parts = class_name.split("Round")
        model_name = parts[0].replace("_", "-").lower()
        round_num = parts[1].split("Solution")[0]
        
        try:
            # Apply the solution
            start_time = time.time()
            result = solution_class.solve(input_text)
            execution_time = time.time() - start_time
            
            # Save the result
            output_filename = f"sample_file_output__{model_name}_round_{round_num}.md"
            output_path = output_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            # Calculate metrics
            output_lines = count_lines(result)
            output_chars = count_chars(result)
            output_size_kb = len(result) / 1024
            
            # Store metrics
            solution_metrics = {
                "model": model_name,
                "round": round_num,
                "execution_time": round(execution_time, 2),
                "output_lines": output_lines,
                "output_chars": output_chars,
                "output_size_kb": round(output_size_kb, 2),
                "lines_ratio": round(output_lines / input_lines, 2),
                "chars_ratio": round(output_chars / input_chars, 2)
            }
            
            metrics.append(solution_metrics)
            
            # Print metrics
            print(f"  Execution time: {solution_metrics['execution_time']}s")
            print(f"  Output lines: {output_lines}")
            print(f"  Output size: {solution_metrics['output_size_kb']} KB")
            print(f"  Output saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error testing {class_name}: {str(e)}")

    # Save metrics
    metrics_path = output_dir.parent / "metrics" / "test_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nMetrics saved to: {metrics_path}")
    
if __name__ == "__main__":
    main()
