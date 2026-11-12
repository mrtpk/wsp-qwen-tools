print(f"FizzBuzz: ")
print(f"Fizz:  ")
cols = 12
for i in range(1, 16):
    if i % 3 == 0:
        print(f"Fizz: {i}")
    elif i % 5 == 0:
        print(f"Buzz: {i}")
    else:
        print(f"FizzBuzz: {i}")

