import matplotlib.pyplot as plt
import numpy as np
import random

SAMPLES = 20

gauss = np.random.normal(loc=5, size=100)
expo = np.random.exponential(10, size=10000)

expo = expo + gauss[-1]

s = np.concatenate((gauss, expo))

# arr_num = [random.randrange(100, step=100/20) for i in range(20)]
histogram_index = list(range(1, 100, 5))
histogram_index.sort()

s_index = [random.randrange(len(s)) for i in range(20)]
s_index.sort()

arrivals_histogram = np.zeros(100)

for i in range(len(histogram_index)):
    arrivals_histogram[histogram_index.pop()] = s[s_index.pop()]

arrivals_histogram /= sum(arrivals_histogram)

print(sum(arrivals_histogram))

count, bins, ignored = plt.hist(arrivals_histogram)

plt.show()

for i in arrivals_histogram:
    print(f"- {i}")
