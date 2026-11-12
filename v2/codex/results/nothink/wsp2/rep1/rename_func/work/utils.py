def foo(items):
    return sum(i['price'] * i['qty'] for i in items)
