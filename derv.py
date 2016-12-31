EPSILON = 10**(-6)

def est_derv(x):
    val = (f(x + EPSILON) - f(x)) / EPSILON
    return val

def derv(x):
    return (2 * x)

def f(x):
    return x**2

def main(x):
    a = f(x)
    b = derv(x)
    c = est_derv(x)
    print(">f(x)={}, f'(x)={}, f^'(x)={}".format(a, b, c))



xs = range(30)

for x in xs:
    main(x)
