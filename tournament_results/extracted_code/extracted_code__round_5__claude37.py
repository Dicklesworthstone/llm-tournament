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