"""Write the Codex task suite to codex/tasks/<id>/{repo/, prompt.txt, check.sh}.

Each task is a small coding job. check.sh runs in the finished work directory; exit 0 = pass.
The check is not in the work directory, so the agent cannot read it.
"""
import shutil, stat
from pathlib import Path

T = Path(__file__).parent / "tasks"
TASKS = {
 "create_add": (
  {"README.md": "# demo\n"},
  "Create a file add.py with a function add(a, b) that returns a + b.",
  'python3 -c "from add import add; assert add(2, 3) == 5 and add(-1, 1) == 0"'),
 "fix_subtract": (
  {"calc.py": "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a + b\n"},
  "The function subtract in calc.py returns the wrong result. Fix it.",
  'python3 -c "from calc import add, subtract; assert subtract(5, 3) == 2 and add(1, 2) == 3"'),
 "rename_func": (
  {"utils.py": "def foo(items):\n    return sum(i['price'] * i['qty'] for i in items)\n",
   "main.py": "from utils import foo\n\nprint(foo([{'price': 2, 'qty': 3}, {'price': 1, 'qty': 4}]))\n"},
  "Rename the function foo in utils.py to compute_total. Update every caller.",
  'grep -q "def compute_total" utils.py && ! grep -qw foo utils.py main.py && [ "$(python3 main.py)" = "10" ]'),
 "reverse_words": (
  {"strings.py": "def reverse_words(text):\n    raise NotImplementedError\n",
   "test_strings.py": "from strings import reverse_words\n\nassert reverse_words('hello world') == 'world hello'\n"
                      "assert reverse_words('a b c') == 'c b a'\nassert reverse_words('one') == 'one'\nprint('ok')\n"},
  "Implement reverse_words in strings.py so that python3 test_strings.py passes.",
  'python3 -c "from strings import reverse_words as r; assert r(\'x y z\') == \'z y x\'" && python3 test_strings.py'),
 "greet_arg": (
  {"greet.py": "print('Hello, World')\n"},
  "Add an optional command-line argument --name to greet.py. With --name Ann it prints 'Hello, Ann'. "
  "Without it, it still prints 'Hello, World'.",
  '[ "$(python3 greet.py)" = "Hello, World" ] && [ "$(python3 greet.py --name Ann)" = "Hello, Ann" ]'),
 "config_port": (
  {"config.json": '{\n  "host": "0.0.0.0",\n  "port": 8000,\n  "debug": false\n}\n'},
  "Change the port in config.json to 9000. Keep the other settings.",
  'python3 -c "import json; c = json.load(open(\'config.json\')); assert c == {\'host\': \'0.0.0.0\', \'port\': 9000, \'debug\': False}"'),
 "count_lines": (
  {"data.txt": "".join(f"row {i}\n" for i in range(37))},
  "Count the lines in data.txt and write only the number into a new file count.txt.",
  '[ "$(tr -d "[:space:]" < count.txt)" = "37" ]'),
 "fizzbuzz": (
  {"README.md": "# fizzbuzz\n"},
  "Create fizzbuzz.py. It prints the numbers 1 to 15, one per line, but Fizz for multiples of 3, "
  "Buzz for multiples of 5, and FizzBuzz for multiples of both.",
  '[ "$(python3 fizzbuzz.py | tr "\\n" " ")" = "1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz " ]'),
 "is_prime": (
  {"mathx.py": "def square(x):\n    return x * x\n"},
  "Add a function is_prime(n) to mathx.py. It returns True if n is a prime number, otherwise False.",
  'python3 -c "from mathx import is_prime, square; assert [n for n in range(30) if is_prime(n)] == [2,3,5,7,11,13,17,19,23,29] and square(3) == 9"'),
 "fix_import": (
  {"helper.py": "def double(x):\n    return 2 * x\n", "main.py": "from helpers import double\n\nprint(double(21))\n"},
  "python3 main.py fails. Fix it.",
  '[ "$(python3 main.py)" = "42" ]'),
 "csv_total": (
  {"sales.csv": "item,amount\napple,3.5\npear,2.25\nplum,4.0\nfig,10.25\n"},
  "sales.csv has the columns item and amount. Create total.py that prints the total of the amount column.",
  'python3 -c "import subprocess; assert abs(float(subprocess.check_output([\'python3\', \'total.py\']).decode().strip()) - 20.0) < 1e-9"'),
 "gitignore": (
  {"app.py": "print('app')\n"},
  "Add a .gitignore file that ignores __pycache__ directories and all .log files.",
  'grep -q "__pycache__" .gitignore && grep -q "\\*\\.log" .gitignore'),
 "find_class": (
  {"a.py": "class Lexer:\n    pass\n", "b.py": "def parse():\n    pass\n", "pkg/c.py": "class Parser:\n    pass\n",
   "pkg/d.py": "import os\n"},
  "Find the file that defines the class Parser. Write its path, relative to this directory, into answer.txt.",
  '[ "$(tr -d "[:space:]" < answer.txt | sed "s#^\\./##")" = "pkg/c.py" ]'),
 "use_math_pi": (
  {"geometry.py": "def circle_area(r):\n    return 3.14159 * r * r\n\n\ndef circumference(r):\n    return 2 * 3.14159 * r\n"},
  "Replace the number 3.14159 in geometry.py with math.pi.",
  '! grep -q 3.14159 geometry.py && python3 -c "import math; from geometry import circle_area as a, circumference as c; assert a(1) == math.pi and c(1) == 2 * math.pi"'),
 "remove_unused": (
  {"main.py": "from core import run\n\nrun()\n", "core.py": "def run():\n    print('run')\n",
   "old_utils.py": "def legacy():\n    pass\n"},
  "Remove the file old_utils.py. Nothing uses it.",
  '[ ! -e old_utils.py ] && [ "$(python3 main.py)" = "run" ]'),
 "fix_failing_test": (
  {"app.py": "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef clamp(x, lo, hi):\n    return max(lo, min(x, lo))\n",
   "test_app.py": "from app import mean, clamp\n\nassert mean([1, 2, 3]) == 2\nassert clamp(5, 0, 10) == 5\n"
                   "assert clamp(-1, 0, 10) == 0\nassert clamp(11, 0, 10) == 10\nprint('ok')\n"},
  "Run python3 test_app.py. One test fails. Fix the bug in app.py, not in the test.",
  'python3 test_app.py && python3 -c "from app import clamp; assert clamp(7, 0, 9) == 7"'),
}

if T.exists():
    shutil.rmtree(T)
for tid, (files, prompt, check) in TASKS.items():
    d = T / tid
    for name, body in files.items():
        p = d / "repo" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body)
    (d / "prompt.txt").write_text(prompt + "\n")
    c = d / "check.sh"; c.write_text("#!/usr/bin/env bash\nset -e\n" + check + "\n")
    c.chmod(c.stat().st_mode | stat.S_IEXEC)
print(len(TASKS), "tasks")
