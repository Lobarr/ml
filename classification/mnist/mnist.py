from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)

dataset = pd.DataFrame(mnist['data'])
dataset['label'] = mnist['target']
# print(dataset.describe())
# print(dataset.info())

# dependent and independent variables
X = dataset.iloc[:, :-1].values
y = dataset['label'].values

#train and test sets
train_set = None
test_set = None

# training and test data split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_index, test_index in split.split(X, y):
  train_set = dataset.loc[train_index]
  test_set = dataset.loc[test_index]

# print(train_set.describe())
# print(test_set.describe())

# binary classifier for number 3
train_set.loc[train_set['label'] == 5, 'label_bool'] = True
train_set.loc[train_set['label'] != 5, 'label_bool'] = False
test_set.loc[test_set['label'] == 5, 'label_bool'] = True
test_set.loc[test_set['label'] != 5, 'label_bool'] = False

train_set_y_5 = train_set['label_bool'].values
test_set_y_5 = test_set['label_bool'].values

sgd_classifier = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
sgd_classifier.fit(train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5)

# test = train_set.drop(['label', 'label_bool'], axis=1).iloc[11, :].values
# test_label = train_set.iloc[11, 784:]
# print(test_label)
# print(sgd_classifier.predict([test]))

pred = cross_val_predict(sgd_classifier, train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5)
conf_matrix = confusion_matrix(train_set_y_5, pred)
precision = precision_score(train_set_y_5, pred)
recall = recall_score(train_set_y_5, pred)
f1 = f1_score(test_set_y_5, pred) 

# print('conf-matrix:', conf_matrix)
# print('precision:', precision)
# print('recall:', recall)
# print('f1 score:', f1)