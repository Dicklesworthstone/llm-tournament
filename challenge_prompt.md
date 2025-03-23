# CSV Normalization and Cleaning Challenge

I want you to make me a sophisticated Python function called `normalize_csv` that takes messy, inconsistent CSV data as input and produces a clean, normalized version while preserving as much information as possible.

The function should have this signature:
```python
def normalize_csv(csv_data: str) -> str:
    """
    Clean and normalize messy CSV data.
    
    Args:
        csv_data: String containing messy CSV data
        
    Returns:
        String containing cleaned, normalized CSV data
    """
```

Your solution should handle the following common issues in CSV files:

1. **Inconsistent delimiters**: Some rows might use commas, others semicolons or tabs
2. **Mixed quote styles**: Some fields might use double quotes, others single quotes, or no quotes
3. **Inconsistent date formats**: Convert all dates to ISO format (YYYY-MM-DD)
4. **Inconsistent number formats**: Convert numbers with various formats (1,000.00 or 1.000,00) to standard format
5. **Empty rows**: Remove completely empty rows
6. **Extra whitespace**: Trim unnecessary whitespace from field values
7. **Inconsistent column names**: Normalize column names to lowercase with underscores
8. **Missing values**: Replace with appropriate NULL values or empty strings
9. **Character encoding issues**: Handle and fix common encoding problems

Here are some examples of messy CSV data that your function should be able to clean:

Example 1:
```
Name, Age, Birth Date, Salary 
"John Smith", 32, "04/25/1991", "$75,000.00"
'Jane Doe';31;'May 3rd, 1992';'€65.000,00'
Robert Johnson  45  "Jan 12 1978"  "92,500"
```

Example 2:
```
Product Name|Price|In Stock|Last Updated
"Wireless Headphones"|"$129.99"|"Yes"|"2023-01-15"
'Smart Watch';€199,95;'no';'01/22/2023'
"USB-C Cable",$19.99,true,"February 3, 2023"
```

Example 3:
```
customer_id,first_name,last_name,email,purchase_amount,purchase_date
1001,John,Doe,john.doe@example.com,"1,240.50","01/15/2023"
1002,Jane,Smith,jane.smith@example.com,"985,75","2023-01-20"
1003,"David, Jr.",Johnson,"david.johnson@example.com","2.399,00","Jan 25, 2023"
```

Your solution should:
1. Detect and adapt to different delimiter styles automatically
2. Handle different types of fields (strings, numbers, dates, booleans) appropriately
3. Preserve the header row but normalize column names
4. Output a consistently formatted CSV with properly escaped fields
5. Be robust to unexpected edge cases
6. Use appropriate error handling

For full credit, provide a detailed explanation of your approach and any assumptions you made.