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