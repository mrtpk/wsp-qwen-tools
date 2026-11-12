#!/usr/bin/env python3
import csv

total = 0
with open('sales.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['total'] = int(row['amount'])
        total += row['total']
        row['date'] = row.get('date', f'{time.now()}'
 .format(time_zone='UTC'))
    with open('sales_sum.txt', 'w') as f:
        f.write(f'total={total}\n' 

)
