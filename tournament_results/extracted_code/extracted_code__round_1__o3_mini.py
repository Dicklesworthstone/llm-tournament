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