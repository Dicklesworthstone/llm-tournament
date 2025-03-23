Below is one complete solution. In our solution we take the following steps:

1. Pre‐process the input text by splitting it into lines and “unifying” delimiters. We scan each input line and replace any semicolons, tabs, pipes, etc. with a comma. (In the one “whitespace‐delimited” row we fall back to splitting on two-or-more spaces.) This “unification” step lets us later parse everything using the csv module with a single delimiter.

2. Use the csv.reader (with delimiter set to a comma) to parse rows. We then tidy up each field by:
  a. Removing extra whitespace and outer quotes (be they single or double quotes).
  b. Attempting to parse dates using dateutil’s parser (with fuzzy matching) and reformat to ISO (YYYY‐MM‐DD);
  c. Trying to “normalize” numbers that may be expressed either in US style (e.g. “1,000.00”) or European style (e.g. “1.000,00”). We remove common currency symbols and try first a “simple” cleaning and then a European‐style cleaning if needed.
  d. Converting common boolean expressions such as “yes”/“true” and “no”/“false” into a normalized lowercase form.
  e. Leaving missing fields as empty strings.

3. We also normalize the column names from the header row so that they are lowercase with underscores (removing leading/trailing spaces).

4. Finally, we write out the “normalized” rows using csv.writer so that output fields are properly quoted if needed.

Throughout the function we try to be robust by catching exceptions (when dates or numbers cannot be parsed) and “fuzzily” cleaning messy data.

Below is the complete code with detailed comments:

------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import io
import re
from dateutil.parser import parse

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
    
    # Helper functions:

    def remove_outer_quotes(s: str) -> str:
        """Strip leading and trailing whitespace and remove matching single or double quotes."""
        s = s.strip()
        if len(s) >= 2:
            # Check for matching double quotes or single quotes
            if (s[0] == s[-1]) and s[0] in ('"', "'"):
                return s[1:-1].strip()
        return s

    def try_normalize_date(s: str) -> str:
        """Try to parse a date and reformat to ISO YYYY-MM-DD.
           Returns the ISO string if successful, or None."""
        try:
            # Use fuzzy parsing to handle things like "May 3rd, 1992"
            dt = parse(s, fuzzy=True, dayfirst=False)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return None

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

    def normalize_value(val: str) -> str:
        v = remove_outer_quotes(val)
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
        date_norm = try_normalize_date(v)
        if date_norm is not None:
            return date_norm
        
        # Try to normalize the number.
        # We only attempt this if v has digits in it.
        if re.search(r'\d', v):
            num_norm = try_normalize_number(v)
            if num_norm is not None:
                return num_norm
        
        # Otherwise, return the cleaned string.
        return v

    # Now process each non-header row.
    for row in rows[1:]:
        # If the entire row is empty (or consisting solely of empty/whitespace values) skip it.
        if all(x.strip() == "" for x in row):
            continue
        normalized_row = [normalize_value(field) for field in row]
        normalized_rows.append(normalized_row)
    
    # Write normalized data using csv.writer so that fields are properly escaped.
    out_io = io.StringIO()
    writer = csv.writer(out_io, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for row in normalized_rows:
        writer.writerow(row)
    
    return out_io.getvalue()

# Example usage

if __name__ == "__main__":
    # Example 1
    csv_example1 = '''
Name, Age, Birth Date, Salary 
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
'''
    print("Normalized Example 1:")
    print(normalize_csv(csv_example1))
    print("-------------")

    # Example 2
    csv_example2 = '''
Product Name|Price|In Stock|Last Updated
"Wireless Headphones"|"$129.99"|"Yes"|"2023-01-15"
'Smart Watch';€199,95;'no';'01/22/2023'
"USB-C Cable",$19.99,true,"February 3, 2023"
'''
    print("Normalized Example 2:")
    print(normalize_csv(csv_example2))
    print("-------------")

    # Example 3
    csv_example3 = '''
customer_id,first_name,last_name,email,purchase_amount,purchase_date
1001,John,Doe,john.doe@example.com,"1,240.50","01/15/2023"
1002,Jane,Smith,jane.smith@example.com,"985,75","2023-01-20"
1003,"David, Jr.",Johnson,"david.johnson@example.com","2.399,00","Jan 25, 2023"
'''
    print("Normalized Example 3:")
    print(normalize_csv(csv_example3))
    
"""
Explanation of key design decisions and assumptions:

1. Delimiter handling: Instead of attempting to use csv.Sniffer (which expects a consistent dialect), we pre‐replace known alternative delimiters
   (semicolon, tab, pipe) with commas. For rows that do not contain any of these delimiters, we fall back to splitting on multiple spaces.
2. Quote stripping: We manually remove matching outer quotes (whether single or double) so that later normalization (of dates or numbers)
   can occur on clean strings.
3. Date normalization: We use dateutil.parser.parse with fuzzy matching to handle various date formats and then output in ISO format.
4. Number normalization: We first remove any currency symbols. Then we try a two‐step attempt: first treating the string as using commas
   as thousand separators (removing commas), and if that fails, we assume a European style (removing dots and replacing comma with a decimal point).
5. Booleans and missing values are handled specifically, and the header row is normalized by lowercasing and replacing spaces with underscores.
6. Finally, we write out the cleaned data using csv.writer so that fields with commas or special characters are correctly escaped.

This solution is intended to be robust for a variety of common CSV “mess” issues, although extremely pathological cases may require further refinement.
