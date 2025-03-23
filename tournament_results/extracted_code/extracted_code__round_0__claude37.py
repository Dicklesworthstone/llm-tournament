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