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