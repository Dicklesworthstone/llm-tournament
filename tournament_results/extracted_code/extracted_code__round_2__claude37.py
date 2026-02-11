import csv
import re
import io
import chardet
from dateutil import parser as date_parser
from collections import Counter
from typing import List

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