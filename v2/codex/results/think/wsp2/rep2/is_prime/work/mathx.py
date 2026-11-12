def square(x):
    return x * x
def is_prime(n):
    if n < 2:
        return False
    
    if n == 2:
        return True
    
    if n % 2 == 0:
        return False
    
    limit = n // 2
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    
    return True


