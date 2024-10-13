import numpy as np
from sympy import *
import matplotlib.pyplot as plt
# import our functions from my functions.py (that i make) please make sure
# it's in the same environment
from Functions import *


# First define our functions
def p(u):
    return (1 - u) ** 8


def q(u):
    return 1 - 8 * u + 28 * u ** 2 - 56 * u ** 3 + 70 * u ** 4 - 56 * u ** 5 + 28 * u ** 6 - 8 * u ** 7 + u ** 8


# initiate our u values between 0.98 and 1.02
random_floats = np.linspace(0.98, 1.02, 500)

# plot them
plt.scatter(random_floats, [p(i) for i in random_floats], label='P(u)',
            linewidth=0.05)
plt.scatter(random_floats, [q(i) for i in random_floats], label='Q(u)',
            linewidth=0.05)
plt.title('P(u) and Q(u) vs 0.98 < u < 1.02')
plt.xlabel('U')
plt.legend()
plt.show()

# plotting p(u) - q(u) and their histogram

difference = np.array([p(i) for i in random_floats]) - np.array(
    [q(i) for i in random_floats])

plt.scatter(random_floats, difference, label='P(u) - Q(u)',
            linewidth=0.05)
plt.title('P(u) - Q(u) vs 0.98 < u < 1.02')
plt.xlabel('U')
plt.ylabel('P(u) - Q(u)')
plt.legend()
plt.show()

plt.hist(difference, bins=30, )
plt.title('P(u) - Q(u) vs 0.98 < u < 1.02, Histogram')
plt.xlabel('P(u) - Q(u)')
plt.ylabel('Counts')
plt.show()

# Calculate std of the difference, and then compare to equation 3 (made in
# functions.py)
# here we calculate N to be
truestd = np.std(difference)
N = 10  # we take N equal to ten since to calculate a single point in
# difference we have to sum 9 terms in Q(u) and 1 in P(u), 10 terms in total
# to calculate this value
# the mean in equation 3 is calculated taking np.mean() of each term in the
# summation squared, so it is calculated for each point in differences
eqn3std = Equation3_sigma(random_floats, N)

print(truestd, eqn3std[0], 'On the left we have np.std on the differences, '
                           'and on the right we wave the error according to '
                           'equation 3 on a single element in differences')
# now we start part c where we define new array of us and another array of
# u1 u values that slowly increase from 0.98 to 0.984

u1 = np.linspace(0.980, 1., 500)

u2 = np.linspace(0.980, 0.99, 1000)

# printing array of numerical errors of each point in the difference when
# 0.98 < u < 1
#print(Equation4_sigma(u1, N))

difference2 = np.array([p(i) for i in u2]) - np.array(
    [q(i) for i in u2])

# plotting to visualize errors
plt.plot(u2, (np.abs(difference2)) / np.abs([p(i) for i in u2]))
plt.xlabel('U')
plt.ylabel('(abs(P(u)-Q(u)))/abs(P(u))')
plt.title('(abs(P(u)-Q(u)))/nabs(P(u)) for 0.980 < u < 0.990')
plt.show()


# Pat d start we have random_floats from before for 0.98 u< 1.02 so we dont
# need to redo that define new product function
def f(u):
    return u ** 8 / ((u ** 4) * (u ** 4))


# find std
print(np.std([f(i) for i in random_floats], ddof=1),
      ' std of f(u) for  0.98 < u < 1.02')

# Now we can define C in order to use equation 3.5 in the textbook
C = 10 ** (-16)
print([f(i) * C for i in random_floats], 'Numerical error of each element of '
                                         'the array of values of f(u)')
# plotting f(u) -1 to visualize errors
plt.plot(random_floats, [f(i) - 1. for i in random_floats], label='F(u)',
         linewidth=1)
plt.xlabel('u')
plt.ylabel('f(u) - 1')
plt.title('f(u) - 1 vs u')
plt.show()
