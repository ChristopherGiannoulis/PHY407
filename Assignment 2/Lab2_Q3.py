import numpy as np
import matplotlib.pyplot as plt
from sympy import *


# this py file looks at central and forward differences methods of
# differentiating a function and compares the relative error of the two methods
# to the analytical derivative of the function

# define our function

def f(x):
    return np.exp(-x ** 2)


# now we do the central difference method
start = -16
end = 0

# Generate the points
h_values = np.logspace(start, end, num=17)
# print(h_values)

# Calculate the derivative using central difference at x = 1/2
central_differences = np.zeros(17)
for i in range(len(h_values)):
    h = h_values[i]
    central_differences[i] = (f(0.5 + h / 2.) - f(0.5 - h / 2.)) / h
for i in range(len(h_values)):
    print(f"For h = {h_values[i]:.2e}, the derivative using central "
          f"difference is {central_differences[i]:.10e}")

# now let's calculate it analytically, on paper the derivative of f(x) = e^(
# -x^2) is -2xe^(-x^2) so at x = 1/2 the derivative is -e^(-1/4) but lets
# calculate it using sympy for practice
x1 = Symbol('x')
g = exp(-x1 ** 2)
f_prime = g.diff(x1)
f_prime = f_prime.subs(x1, 1. / 2.)
print('The analytical derivative is:', f_prime)

# now we calculate the relative error of our values to the 'truth'
relative_error = np.abs((central_differences - f_prime) / f_prime)
for i in range(len(h_values)):
    print(f"For h = {h_values[i]:.2e}, "
          f"the relative error is {relative_error[i]:.10e}")

# now we repeat this whole process using forward difference
forward_differences = np.zeros(17)
for i in range(len(h_values)):
    h = h_values[i]
    forward_differences[i] = (f(0.5 + h) - f(0.5)) / h

for i in range(len(h_values)):
    print(f"For h = {h_values[i]:.2e}, the derivative using forward "
          f"difference is {forward_differences[i]:.10e}")

# now we calculate the relative error of our values to the 'truth'
relative_error_forward = np.abs((forward_differences - f_prime) / f_prime)
for i in range(len(h_values)):
    print(f"For h = {h_values[i]:.2e}, "
          f"the relative error "
          f"using forward difference is {relative_error_forward[i]:.10e}")

# now to plot the relative error vs h
plt.figure()
plt.scatter(h_values, relative_error, label='Central Difference')
plt.scatter(h_values, relative_error_forward, label='Forward Difference')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('h')
plt.ylabel('Relative Error')
plt.title('Relative Error vs h')
plt.legend()
plt.grid()
plt.show()


# now we define a function for exp(2x) and use central difference to calculate
# the derivatives at x = 0 and compare to the analytical derivative we will do
# this for 5 derivatives using recursion
def f2(x):
    return np.exp(2 * x)


deltax = np.float64(10 ** -6)  # initiating our h value


# this below function is used to calculate the nth derivative of our function
# using the deltax we initiated above

def derivative_calculator(function, x, n, h=deltax):
    if n == 0:
        return function(x)
    elif n > 1:
        return (derivative_calculator(function, x + h / 2.,
                                      n - 1.) - derivative_calculator(function,
                                                                      x - h / 2.,
                                                                      n - 1.)) / h

    else:
        return (function(x + h / 2.) - function(x - h / 2.)) / h


amount_of_derivatives = 5  # the amount of derivatives we want to calculate

for i in range(amount_of_derivatives + 1):
    print(f"The {i}th derivative of f2(x) at x = 0 is:",
          derivative_calculator(f2, 0, i))

