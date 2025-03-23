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