#!/usr/bin/env python3
import sys, argparse

def default_message():
    return "Hello, World"

def greet_default():
    pass

def greet_custom():
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name == "":
            print(default_message())
        else:
            print(f"Hello, {name}")
    else:
        print(default_message())

main = greet_custom
sys.exit(main())
