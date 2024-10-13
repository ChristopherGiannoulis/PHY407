import numpy as np
from sympy import *
from warnings import *


def average_1d(data):  # calculate the arithmetic mean of 1d data set
    n = float(len(data))
    sum = 0.
    for i in range(int(n)):
        sum += data[i]
    return sum / n


def Equation1_std_1d(avg,
                     data):  # take in avg of 1d dataset and calculate std
    # according to eqn 1
    n = float(len(data))
    sum = 0.
    for i in range(int(n)):
        sum += (data[
                    i] - avg) ** 2  # doing the sum of equation 1 using an
        # inputted average of the 1d data set
    return np.sqrt(1 / (n - 1) * sum)  # returning equation 1 std


def Equation2_std_1d(avg,
                     data):  # take in avg of 1d dataset and calculate std
    # according to eqn 2
    n = float(len(data))
    sum = 0.
    point = data[0]
    for i in range(int(n)):
        sum += float(data[i] - point) ** 2  # summing according to equation 2
        # and my correction
    if (1 / (n - 1) * sum) < 0:
        warn('Negative value encountered in calculation')
        return np.sqrt(
            np.abs(1. / (n - 1) * (float(sum) - n * float64(avg) ** 2)))
    return np.sqrt(1. / (n - 1) * (float(sum) - n * float(
        avg - point) ** 2))  # return equation 2 based on our sum and input
    # avg of the data passed in


def Equation3_sigma(U,N):  # uses equation 3 for in lab 1 Q2 for the difference
    # between our polynomials p and q
    C = float(10 ** (-16))
    array = np.zeros(len(U))
    for i in range(len(U)):
        u = U[i]
        calculatedterms = np.array(
            [(1 - u) ** (8 * 2), 1, (8 * u) ** 2, (28 * u ** 2) ** 2,
             (56 * u ** 3) ** 2, (70 * u ** 4) ** 2,
             (56 * u ** 5) ** 2, (28 * u ** 6) ** 2,
             (8 * u ** 7) ** 2, (u ** 8) ** 2])  # this is putting together a
        # array of each term squared in the difference
        value = np.mean(calculatedterms)  # this calculates the mean squared
        # of our sum
        array[i] = C * np.sqrt(N) * np.sqrt(np.abs(value))  # equation 3
    return array


def Equation4_sigma(U, N):  # uses equation 4 for in lab 1 Q2 for the
    # difference between our polynomials p and q
    C = float(10 ** (-16))
    array = np.zeros(len(U))
    for i in range(len(U)):
        u = U[i]
        calculatedterms1 = np.array(
            [(1 - u) ** 8, -1, (8 * u), -1 * (28 * u ** 2), (56 * u ** 3),
             -1 * (70 * u ** 4), (56 * u ** 5), -1 * (28 * u ** 6),
             (8 * u ** 7), -1 * (u ** 8)])  # this is putting together an array
        # of each term in the difference
        calculatedterms2 = np.array(
            [(1 - u) ** (8 * 2), 1, (8 * u) ** 2, (28 * u ** 2) ** 2,
             (56 * u ** 3) ** 2, (70 * u ** 4) ** 2,
             (56 * u ** 5) ** 2, (28 * u ** 6) ** 2,
             (8 * u ** 7) ** 2, (u ** 8) ** 2])  # this is putting together a
        # array of each term squared in the difference
        value2 = np.mean(calculatedterms2)  # these lines calculate the mean
        # squared
        value1 = np.mean(calculatedterms1)
        array[i] = (C / np.sqrt(N) * np.sqrt(np.abs(value2)) / np.abs(value1))
        # equation 4
    return array
