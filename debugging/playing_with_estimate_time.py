import numpy as np
import matplotlib.pyplot as plt

l_pois =4
l_exp = 1 / l_pois


# Q: how many spikes do we need to get a good estimate of l < std 1

perc = 0.1
s = (l_exp * perc) / 2 # 99% of values (this is very conservative)
n = 1 / ( s**2 / l_exp**2)
t = n/l_pois

print(t)

MC = 10000
estimates = np.zeros(MC)
for i in range(MC):

    exp = np.random.exponential(l_exp, int(n))
    estimates[i] = np.mean(exp)

plt.hist(estimates)
plt.show()
print(np.sum(np.logical_and(estimates >= l_exp - l_exp * perc, estimates <= l_exp + l_exp * perc) / estimates.size))
