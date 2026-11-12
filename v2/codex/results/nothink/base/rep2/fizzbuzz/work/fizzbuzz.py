def fizzBuzz(*singleMultipliers):
    for number in range(1, 16):
        if isinstance(number, int):
            result = number
        else:
            result = str(number)
        
        for multiplier in singleMultipliers:
            if multiplier is not None and result % (multiplier * 10) == 0:
                result = str(multiplier)
        
        if result:
            print(result)
