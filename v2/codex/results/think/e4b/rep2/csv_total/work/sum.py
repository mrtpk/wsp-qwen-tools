sum_amount = 0
with open('/work/sales.csv', 'r') as f:
    next(f, None)
    for line in f:
        if ',' in line:
            items = line.split(',')
            total = sum(map(float, [item for item in items if item]))
            sum_amount += total
print(f"Total amount from sales.csv: {sum_amount}")
