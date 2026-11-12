def compute_total(items):
    return sum(i['price'] * i['qty'] for i in items)
