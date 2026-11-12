#!/usr/bin/env python3
import re

def reverse_words(word_list):
    result = []
    for word in word_list:
        if word.strip():
            result.append(reversed(word.split()))
    return result

words = ["eat", "not", "a", "cherry"]
result = [word[::-1] for word in words]
print("Result:", result)

