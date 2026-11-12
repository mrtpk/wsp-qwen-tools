from app import mean, clamp

assert mean([1, 2, 3]) == 2
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0
assert clamp(11, 0, 10) == 10
print('ok')
