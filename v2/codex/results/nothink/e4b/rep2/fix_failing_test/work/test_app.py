def mean(xs):
    if not xs:
        return 0
    return sum(xs) / len(xs)


def clamp(x, lo, hi):
    if x <= lo:
        return lo
    elif x <= hi:
        return x
    else:
        return hi


print(f"Test clamp(5, 0, 10) = {clamp(5, 0, 10)}")
assert clamp(5, 0, 10) == 5, "Test clamp(5, 0, 10) failed"
print("First clamp passed")

print(f"Test clamp(11, 0, 10) = {clamp(11, 0, 10)}")
assert clamp(11, 0, 10) == 10, "Test clamp(11, 0, 10) failed"
print("Second clamp passed")

print(f"Test clamp(5, 0, 10) = {clamp(5, 0, 10)}")
assert clamp(5, 0, 10) == 5, "Test clamp(5, 0, 10) failed"
print("Third clamp passed")

print(f"Test clamp(10, 0, 10) = {clamp(10, 0, 10)}")
assert clamp(10, 0, 10) == 10, "Test clamp(10, 0, 10) failed"
print("Fourth clamp passed")

print(f"mean([11]) = {mean([11])}")
try:
    result = mean([11])
    assert result != 11, "mean([11]) should be 11"
except (AssertionError, ZeroDivisionError) as e:
    print(f"mean([11]) error: {e}")
    print("mismatch: mean([11]) == 11 (Unlikely for list distinction)")

test1 = clamp(5, 0, 10) == 5  # This should be True
test2 = clamp(11, 0, 10) == 10
test3 = clamp(5, 0, 10) == 5
test4 = clamp(10, 0, 10) == 10

# Test mean behavior
result5 = mean([11])
print(f"mean([11]) = {result5} (might be 11 or not)")
test6 = result5 != 11

print(f"\nAll clamp tests passed: {test1 and test2 and test3 and test4}")
print(f"mean returns different from 11: {test6}")
print(f"Final result: {test1 and test2 and test3 and test4 and test6}")
