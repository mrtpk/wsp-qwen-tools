import csv

# Read the sales.csv file and calculate the total of the amount column
with open('/work/sales.csv', 'r') as f:
    reader = csv.DictReader(f)
    total = 0
    for entry in reader:
        total += int(entry['amount'])

# Print the total
print(total)
