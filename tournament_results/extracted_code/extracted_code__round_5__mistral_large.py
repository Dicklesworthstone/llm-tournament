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