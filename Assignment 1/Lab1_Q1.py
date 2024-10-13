import numpy as np
from sympy import *
import matplotlib.pyplot as plt
# import our functions from my functions.py (that i make) please make sure
# it's in the same environment
from Functions import *

# In this next sequence of code we will calculate the STD of
# our data using np.std and comparing it to the std calculated using eqn 1

data = np.loadtxt('cdata.txt')  # load cdata

truestd = np.std(data, ddof=1)
print(str(truestd) + ' \"True\" Std')  # checking our code works at this point

# next calculate x_average of our data according to eqn 1
x_average = average_1d(data)

# now use this to calculate std according to equation 1 (hidden in functions)
eqn1std = Equation1_std_1d(x_average, data)
print(str(eqn1std) + ' Eqn1 std')

# now calculate the relative error from eqn 1 std to numpy std
print(str((eqn1std - truestd) / truestd) + ' relative error with equation 1')

# now repeat the same process but with equation 2 instead
eqn2std = Equation2_std_1d(x_average, data)
print(str(eqn2std) + ' Eqn2 std')
print(str((eqn2std - truestd) / truestd) + ' relative error with equation 2')

# Part(c) start

# pre set values for normal distributions
mean1, sigma1, n1 = (0., 1., 2000)

mean2, sigma2, n2 = (1.e7, 1., 200)

# generate normal distributions arrays
values1 = np.random.normal(mean1, sigma1, n1)

values2 = np.random.normal(mean2, sigma2, n2)

# print our 'true values'
print(np.std(values1, ddof=1), np.std(values2, ddof=1), 'Numpy std of first '
                                                        'array on the left ('
                                                        'smaller mean)'
                                                        'and second array on '
                                                        'the right ( larger '
                                                        'mean)')

# calculate the arithmetic mean of our arrays and then use our equations of
# standard deviation and print them
values1average = average_1d(values1)
values2average = average_1d(values2)
print(values1average, values2average, 'Arithmetic mean of normal values with '
                                      'smaller mean on the left larger on the'
                                      ' right')

values1std1 = Equation1_std_1d(values1average, values1)
values1std2 = Equation2_std_1d(values1average, values1)
values2std1 = Equation1_std_1d(values2average, values2)
values2std2 = Equation2_std_1d(values2average, values2)

# now print the relative errors
print(str((values1std1 - np.std(values1, ddof=1)) / np.std(values1, ddof=1)) + ' relative error with equation 1 on array of smaller mean')
print(str((values2std1 - np.std(values2, ddof=1)) / np.std(values2, ddof=1)) + ' relative error with equation 1 on array of larger mean')
print(str((values1std2 - np.std(values1, ddof=1)) / np.std(values1, ddof=1)) + ' relative error with equation 2 on array of smaller mean')
print(str((values2std2 - np.std(values2, ddof=1)) / np.std(values2, ddof=1)) + ' relative error with equation 2 on array of larger mean')
