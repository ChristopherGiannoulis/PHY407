import numpy as np
import matplotlib.pyplot as plt
from scipy import \
    constants as con  # importing constants which includes speed of light


# this py file calculates the period of a relativistic particle in a spring
# and using gaussian quadrature to integrate for the period of the particle
# we then plot this and analyse the results
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


# initiate our vairables
k_constant = 12  # constant k
mass = 1  # mass of the particle
x_o = 0.01  # cm to m conversion


# now we have to define dx/dt and then a function for period that we will
# integrate

def velocity_of_relativistic_particle(x, k, m, x_0):
    return con.c * np.sqrt((k * (x_0 ** 2 - x ** 2) * (
            2 * m * con.c ** 2 + k * (x_0 ** 2 - x ** 2) / 2)) / (2 * (
            m * con.c ** 2 + k * (x_0 ** 2 - x ** 2) / 2) ** 2))


def period_of_relativistic_particle_integral(x, k, m, x_0):  # defining the
    # integral
    return 4 / velocity_of_relativistic_particle(x, k, m, x_0)


# now we set up the gaussian quadrature

N1 = 8
N2 = 16

x1, w1 = gaussxwab(N1, 0, x_o)
x2, w2 = gaussxwab(N2, 0, x_o)

# now we will calculate the period of the particle
period1 = sum(w1 * period_of_relativistic_particle_integral(x1, k_constant, mass
                                                            , x_o))
period2 = sum(w2 * period_of_relativistic_particle_integral(x2, k_constant, mass
                                                            , x_o))
print('The period of the particle using N = 8 is:', period1)
print('The period of the particle using N = 16 is:', period2)
print('The difference between the two periods ( and therefore fractional '
      'error of the integral using N1) is:', period2 - period1)

# now we will plot the integrand values vs x1 or x2 as well as the weights *
# integrand
plt.plot(x1,
         period_of_relativistic_particle_integral(x1, k_constant, mass, x_o))
plt.plot(x1, w1 * period_of_relativistic_particle_integral(x1, k_constant, mass,
                                                           x_o))
plt.plot(x2,
         period_of_relativistic_particle_integral(x2, k_constant, mass, x_o))
plt.plot(x2, w2 * period_of_relativistic_particle_integral(x2, k_constant, mass,
                                                           x_o))
plt.xlabel('x')
plt.ylabel('Integrand values')
plt.title('Integrand values and weights * integrand values vs x ')
plt.legend(
    ['Integrand values for N = 8', 'Weights * Integrand values for N = 8',
     'Integrand values for N = 16', 'Weights * Integrand values for N = 16'])
plt.grid()
plt.show()

# now we wil plot t as a function of x_0 between 1m <x< 10 x_c, but first we
# must calculate x_c for this spring constant and T classical
x_c = con.c * np.sqrt(mass / k_constant)
print('The critical distance x_c is:', x_c)

Tclassical = 2 * np.pi * np.sqrt(mass / k_constant)

# now we calculate periods and plot!
x_0s = np.linspace(0.01, 10 * x_c, 1000)  # initiate x_0 array
periodarray = np.zeros(len(x_0s))
for i in range(len(x_0s)):
    xpoints, weight = gaussxwab(N2, 0, x_0s[i])
    periodarray[i] = sum(weight * period_of_relativistic_particle_integral(
        xpoints, k_constant, mass,
        x_0s[i]))  # calculating the period at each x_0

plt.scatter(x_0s, periodarray, label='Period of the particle', color='b',
            s=0.5)  # plotting the period of the particle
plt.axhline(y=Tclassical, color='r', linestyle='--',
            label='Classical period: 2pi*sqrt(m/k)')  # plotting the
# classical period
plt.plot(x_0s, 4 * x_0s / con.c, linestyle='--', color='g',
         label='Relativistic period: 4x/c')  # plotting the relativistic
# period limit
plt.xlabel('x_0')
plt.ylabel('Period')
plt.title('Period of the particle vs x')
plt.legend()
plt.grid()
plt.show()
