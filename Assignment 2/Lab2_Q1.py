import numpy as np
import matplotlib.pyplot as plt


# this py file calculates the potential energy of the quantum harmonic
# oscillator and also visualises the wave function for n = 0 to 5
# importing libraries needed

# defining the Hermite polynomial function
def H(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2. * x
    else:
        return 2. * x * H(n - 1., x) - 2. * (n - 1.) * H(n - 2., x)


# defining the wave function
def psi(n, x):
    return 1. / np.sqrt(
        2. ** n * np.math.factorial(n) * np.sqrt(np.pi)) * np.exp(
        -x ** 2. / 2.) * H(n, x)


# defining the function to visualise the wave function
def psi_visualiser(n):
    x = np.linspace(-4, 4, 1000)
    for i in range(n + 1):
        plt.plot(x, psi(i, x), label='n = ' + str(i), color='C' + str(i))
    plt.xlabel('x')
    plt.grid()
    plt.ylabel('psi')
    plt.title('Wave function for n = 0 to ' + str(n))
    plt.legend()
    plt.savefig('C:/Users/beema/PHY407/PHY407/Images/wave_function.png')
    plt.show()


# visualising the wave function for n = 0 to 5
psi_visualiser(3)


# importing gaussian quadrature function
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = np.linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)

    return x, w


def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


# implementing psi with a change of variables to evaluate the integral
# where x = z/(1-z)**2 and dx = (1+z)**2/(1-z**2)**2 dz

def Hz(n, z):
    x = z / (1 - z ** 2)  # changing the variable x to z/(1-z)**2
    if n == 0:
        return 1
    elif n == 1:
        return 2. * x
    else:
        return 2. * x * Hz(n - 1., z) - 2. * (n - 1.) * Hz(n - 2., z)


def psiz(n, z):
    x = z / (1 - z ** 2)  # changing the variable x to z/(1-z)**2
    dx = (1 + z) ** 2 / (1 - z ** 2) ** 2  # dx = (1+z)**2/(1-z**2)**2 dz
    hermite = Hz(n, z)
    wave_function = 1. / np.sqrt(
        2. ** n * np.math.factorial(n) * np.sqrt(np.pi)) * np.exp(
        -x ** 2. / 2.) * hermite
    return x ** 2 * (np.abs(wave_function)) ** 2 * dx


# defining the function to calculate the potential energy with gaussian quad
def potential_energy(x, n):
    points = x
    z1, w = gaussxw(points)
    return 0.5 * sum((w * psiz(n, z1)))  # multiply by 0.5 since we integrated
    # and calculated the measure of twice the potential energy


N = 10  # nth oscillator state
gausspoints = 100  # number of points for the Gaussian quadrature
for i in range(N + 1):
    print('The potential energy for n =', i, 'is',
          potential_energy(gausspoints, i))
