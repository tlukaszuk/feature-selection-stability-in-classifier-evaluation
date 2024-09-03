INFINITY = (1 << 20)
ZERO = 0.00000001
EPSILON = 0.0000000001

def EQUAL_ZERO(value):
    return abs(value) < ZERO

def EQUALS_EPSILON(value1, value2):
    return abs(value1 - value2) < EPSILON