def mean(xs):
    return sum(xs) / len(xs)


def clamp(x, lo, hi):
    return max(lo, min(x, lo))

test_mean = mean([1, 2, 3])
test_clamp = clamp(5, 0, 10)
test_clamp_low = clamp(-1, 0, 10)
test_clamp_high = clamp(11, 0, 10)

assert test_mean == 2
assert test_clamp == 5
assert test_clamp_low == 0
assert test_clamp_high == 10

print("ok")
