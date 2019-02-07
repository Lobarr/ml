from sklearn.datasets import fetch_openml
import pandas as pd


mnist = fetch_openml('mnist_784', version=1, cache=True)
# print(mnist['target'])

dataset = pd.DataFrame(mnist['data'])
dataset['label'] = mnist['label']
dataset.describe()
dataset.info()