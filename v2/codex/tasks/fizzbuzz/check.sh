#!/usr/bin/env bash
set -e
[ "$(python3 fizzbuzz.py | tr "\n" " ")" = "1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz " ]
