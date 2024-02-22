INFINITY = (1 << 20)
ZERO = 0.00000001
EPSILON = 0.0000000001

def EQUAL_ZERO(value):
    return (value < ZERO) & (value > -ZERO)

def EQUAL_EPSILON(value):
    return (value < EPSILON) & (value > -EPSILON)