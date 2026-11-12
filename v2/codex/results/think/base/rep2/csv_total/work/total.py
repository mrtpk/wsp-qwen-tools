import csv
from collections import OrderedDict

# Read sales.csv
with open('sales.csv', 'r') as f:
    lines = f.readlines()

# Parse data (assuming dot-separated values for items and amounts)
items = OrderedDict()

# Handle header line
lines = lines[1:]

for i, line in enumerate(lines):
    stripped_line = line.strip()
    if stripped_line:  # Skip empty lines
        # If not header, try CSV format
        if stripped_line.startswith('item,') or stripped_line.startswith(' ITEM,'):
            # CSV format: item,amount,name,amount,...
            parts = stripped_line.split(',')
            if i < count:
                part = parts[i]
                items[part] = float(part)
            else:
                # Append comma as part name
                part = parts[i] + ','
                items[part] = float(part)
        else:
            # Simple format: item,name,amount,...
            parts = stripped_line.split(',')
            for part in parts:
                part = part[0]  # Keep first part as item
                if ':' in part:
                    # Already quoted in format
                    items[part] = float(part.split(':')[1])
                else:
                    # Unquoted name
                    items[part] = float(part)

# Write to sales.csv back
with open('sales.csv', 'w') as f:
    f.write(','.join(titles) + '\n')
    for title, amount in totals.items():
        f.write(f'{title},{amount:.2f}\n')
