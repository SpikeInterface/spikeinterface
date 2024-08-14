import numpy as np
import matplotlib.pyplot as plt

lambda_ = 100

c = 0.1
n_sd = 1
n_draws = (n_sd**2 * lambda_ / c**2)

t = n_draws  # * lambda_  # TODO: check this

print(n_draws)

MC = 10000
estimates = np.zeros(MC)
for i in range(MC):
    exp = np.random.poisson(lambda_, int(n_draws))
    estimates[i] = np.mean(exp)

plt.hist(estimates)
plt.show()

print(np.std(estimates))
print(np.sum(np.logical_and(estimates >= lambda_ - c,
                            estimates <= lambda_ + c) / estimates.size))


if False:
    # Q: how many spikes do we need to get a good estimate of l < std 1

    # Damn this works for small
    c = 5 * 0.01
    n_sd = 1

    n = n_sd**2 / (c**2 * rate**2)
    t = n/rate

    print("c", c)

    print("std:", 1 / (rate * np.sqrt(n)))

    print("n", n)
    print("t", t)

    MC = 10000
    estimates = np.zeros(MC)
    for i in range(MC):

        exp = np.random.exponential(1/rate, int(n))
        estimates[i] = np.mean(exp)

    plt.hist(estimates)
    plt.show()

    print(np.std(estimates))
    print(np.sum(np.logical_and(estimates >= (1/rate) - c, estimates <= (1/rate) + c) / estimates.size))
