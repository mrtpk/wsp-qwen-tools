def print_greeting(name='Hello, World'):
    '''Print a greeting to the user.'''
    print(f'Hello, {name}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print_greeting()
    else:
        print_greeting(sys.argv[1])
