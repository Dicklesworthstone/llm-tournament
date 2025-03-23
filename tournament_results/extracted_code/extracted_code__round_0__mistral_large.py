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