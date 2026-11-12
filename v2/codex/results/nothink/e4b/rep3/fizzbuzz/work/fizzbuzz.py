# fizzbuzz.py
# Prints numbers 1 to 15 with fizzBuzz formatting

for i in range(1, 16):
    quo = i // 3
    _, rem = divmod(i, 5)
    print(f"izz if (quo % 3 == 0)")
    if rem == 1:
        print(f"Buzz for {i}")
    elif quo % 3 == 0:
        print(f"izz for {i}")
    else:
        print(f"Buzz for {i}")
