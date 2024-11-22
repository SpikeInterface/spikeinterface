import numpy as np

lambda_hat_s = 2
range_percent = 0.1

confidence_z = 1.645  # TODO: check, based on 90% confidence

e = lambda_hat_s * range_percent

n = lambda_hat_s / (e / confidence_z)**2


MC = 10000

sim_data = np.empty(MC)

for i in range(MC):

    # Dont do this, model a poisson process
    draws = np.random.exponential(1/lambda_hat_s, size=10000) # way too many, calculate properly

    in_time_range = np.cumsum(draws) < n
    assert not np.all(in_time_range) / n, "need to increase size"

    count = np.sum(in_time_range) / n

    sim_data[i] = count

in_range = np.logical_or(sim_data < lambda_hat_s - e, sim_data > lambda_hat_s + e)

print(f"confidence : {1 - np.mean(in_range)}")  # this is compeltely wrong
