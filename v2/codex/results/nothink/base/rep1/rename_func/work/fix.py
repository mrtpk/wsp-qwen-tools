#!/usr/bin/env python3
import re
# Rename foo to compute_total
with open('utils.py', 'r') as f:
    content = f.read()

content = content.replace('def foo(bar, baz):', 'def compute_total(bar, baz):')

with open('utils_new.py', 'w') as f:
    f.write(content)
    # Append "Your Updated CFile Path"
    f.write('Your Updated CFile Path\n')
    print("Renamed foo to compute_total")

print("__init__.py changes:", end=" ")
with open('utils.py') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'import (__import__)' in line:
        print(f"{i+1}: {line.rstrip()}")
        break
else:
    print("__init__.py changes:", end=" (no imports found)")
