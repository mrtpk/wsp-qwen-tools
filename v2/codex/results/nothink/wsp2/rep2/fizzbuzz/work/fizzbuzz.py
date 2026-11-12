def fizz(): return "Fizz"
def buzz(): return "Buzz"
def fizzbuzz(m):
    if m % 3 == 0 and m % 5 == 0:
        return f"{fizz()}{buzz()}"
    elif m % 3 == 0:
        return "Fizz"
    elif m % 5 == 0:
        return "Buzz"
    else:
        return str(m)


if __name__ == "__main__":
    for i in range(1, 16):
        print(fizzbuzz(i))
