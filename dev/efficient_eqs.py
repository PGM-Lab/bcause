
def decimal_to_binary(n, nbits):
    binary_digits = []
    while len(binary_digits) < nbits:
        remainder = n % 2
        binary_digits.append(str(remainder))
        n = n // 2

    # Reverse the list to get the correct binary representation
    binary_digits.reverse()

    return ''.join(binary_digits)




def f(u,x):
    return int(decimal_to_binary(u,2)[x])
def is_compatible(u,x,y):
    return f(u,x) == y



[f(u,0) for u in range(0,4)]
[f(u,1) for u in range(0,4)]

is_compatible(u=0,x=0,y=0)

is_compatible(u=0,x=0,y=1)










