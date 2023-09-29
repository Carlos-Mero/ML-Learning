import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns
sns.set(style="ticks", color_codes=True)

import pandas as pd

pd.set_option('display.precision', 2)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 100)

import sklearn
from sklearn.datasets import load_iris
iris = load_iris()
