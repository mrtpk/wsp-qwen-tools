# FizzBuzz script that prints 1 to 15 one per line, with different zones for multiples

def fizzbuzz(n):
    """Print numbers 1 to n with Fizz, Buzz, FizzBuzz zones."""
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

if __name__ == "__main__":
    # Print numbers from 1 to 15 one per line
    fizzbuzz(15)
