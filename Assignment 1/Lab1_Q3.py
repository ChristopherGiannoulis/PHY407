import numpy as np
from sympy import *
import matplotlib.pyplot as plt
# import our functions from my functions.py (that i make) please make sure
# it's in the same environment
from Functions import *
import timeit


# calculate integral using sympy after defining our function

def f(x):
    return 4 / (1 + x ** 2)


a = 0
b = 1
xs = Symbol('xs', real=True)  # the variable of integration
integralvalue = integrate(f(xs), (xs, a, b))
print(integralvalue, np.pi)


# set up function call inorder to find runtime of integrals
def run_code():

    N = 32
    h = (b - a) / N

    # I will now set up the for loops for the integral in for both the
    # trapezoidal rule and simpsons rule
    trapsum = 0
    simpsumodd = 0
    simpsumeven = 0

    for i in range(1, N):
        trapsum += f(a + i * h)

    for i in range(1, N, 2):
        simpsumodd += f(a + i * h)

    for i in range(2, N, 2):
        simpsumeven += f(a + i * h)

    # now use the equations to calculate each integral
    trapezoidalintegral = h * (0.5 * f(a) + 0.5 * f(b) + trapsum)
    simpsonsintegral = (h * (1. / 3.)) * (
            f(a) + f(b) + 4. * simpsumodd + 2. * simpsumeven)
    print(trapezoidalintegral, simpsonsintegral, 'Left is the result using '
                                                 'trapezoidal integration rule,'
                                                 'and right is using simpson\'s')

    print((trapezoidalintegral - float(integralvalue)) / float(integralvalue),
          (simpsonsintegral - float(integralvalue)) / float(integralvalue),
          'relative error of the trapezoidal rule on the left, and simpsons '
          'on the right')


execution_time = timeit.timeit(run_code,
                               number=1)  # call run_code and obtain its
# runtime
print(execution_time, ' run time')

# prepare for part d by taking trapezoidal rule out of the function run_time

N = 16
h = (b - a) / N
h1 = (b - a) / (2 * N)

trapsum = 0
trapsum1 = 0

for i in range(1, N):
    trapsum += f(a + i * h)

for i in range(1, 2 * N):
    trapsum1 += f(a + i * h1)

trapezoidalintegral = h * (0.5 * f(a) + 0.5 * f(b) + trapsum)
trapezoidalintegral1 = h1 * (0.5 * f(a) + 0.5 * f(b) + trapsum1)

print(trapezoidalintegral1)
print((1. / 3.) * (trapezoidalintegral1 - trapezoidalintegral), 'error of the '
                                                                'estimation '
                                                                'using the '
                                                                'trapezoidal '
                                                                'rule of N = 32')
# simpsumodd = 0
# simpsumodd1 = 0
# simpsumeven = 0
# simpsumeven1 = 0
# for i in range(1, N, 2):
# simpsumodd += f(a + i * h)

# for i in range(2, N, 2):
# simpsumeven += f(a + i * h)

# for i in range(1, 2*N, 2):
# simpsumodd1 += f(a + i * h1)

# for i in range(2, 2*N, 2):
# simpsumeven1 += f(a + i * h1)

# simpsonsintegral = (h * (1. / 3.)) * (
#      f(a) + f(b) + 4. * simpsumodd + 2. * simpsumeven)
# simpsonsintegral1 = (h1 * (1. / 3.)) * (
#  f(a) + f(b) + 4. * simpsumodd1 + 2. * simpsumeven1)

# print((1./15.)*(simpsonsintegral1 - simpsonsintegral))
