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