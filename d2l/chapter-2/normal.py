import math
import matplotlib.pyplot as plt 
import numpy as np
import random
import torch

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

x = np.arange(-7, 7, 0.01)

params = [(0, 1), (0, 2), (3, 1)]

plt.figure(figsize=(4.5, 2.5))
plt.plot(x, normal(x, mu=0, sigma=1), color='gray', label='mean 0, std 1')
plt.plot(x, normal(x, mu=0, sigma=2), color='orange', label='mean 0, std 2')
plt.plot(x, normal(x, mu=3, sigma=1), color='green', label='mean 3, std 1')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Normal Distribution')
plt.legend()
plt.show()
