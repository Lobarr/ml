from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np


mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)

dataset = pd.DataFrame(mnist['data'])
dataset['label'] = mnist['target']
print(dataset.describe())
print(dataset.info())