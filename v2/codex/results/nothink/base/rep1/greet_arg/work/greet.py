#!/usr/bin/env python3
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greet script")
    parser.add_argument("--name", type=str, default=None,
                       help="Name for the greeting (e.g., 'Ann', 'World', etc.)")
    args = parser.parse_args()
    
    if args.name:
        print(f"Hello, {args.name}")
    else:
        print("Hello, World")
