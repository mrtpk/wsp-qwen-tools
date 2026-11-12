def clamp(n, a, b):
    if n < a:
        return a
    elif n < b:
        return n
    else:
        return b

# Test that clamp works correctly
result = clamp(5, 0, 10)
assert result == 5, f"Expected 5, got {result}"
print("Test passed!")
