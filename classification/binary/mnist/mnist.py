from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


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
# train_set.loc[train_set['label'] == 5, 'label_bool'] = True
# train_set.loc[train_set['label'] != 5, 'label_bool'] = False
# test_set.loc[test_set['label'] == 5, 'label_bool'] = True
# test_set.loc[test_set['label'] != 5, 'label_bool'] = False

train_set_y_5 = train_set['label_bool'].values
test_set_y_5 = test_set['label_bool'].values
# train_set_y = train_set['label'].values
# test_set_y = test_set['label'].values

sgd_classifier = SGDClassifier(max_iter=8, tol=np.infty, random_state=42)
sgd_classifier.fit(train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5)

# knn_classifier = KNeighborsClassifier(n_neighbors=3)
# knn_classifier.fit(train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5)

# test = train_set.drop(['label', 'label_bool'], axis=1).iloc[11, :].values
# test_label = train_set.iloc[11, 784:]
# print(test_label)
# print(sgd_classifier.predict([test]))

pred_sgd = cross_val_predict(sgd_classifier, train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5, cv=3, method="decision_function")
# conf_matrix_sgd = confusion_matrix(train_set_y_5, pred_sgd)
# precision_sgd = precision_score(train_set_y_5, pred_sgd)
# recall_sgd = recall_score(train_set_y_5, pred_sgd)
# f1_sgd= f1_score(train_set_y_5, pred_sgd) 
# roc_auc_sgd = roc_auc_score(train_set_y_5, pred_sgd)

# # # print(pred)
# print('--- SGD Classifier ---')
# print('conf-matrix:', conf_matrix_sgd)
# print('precision:', precision_sgd)
# print('recall:', recall_sgd)
# print('f1 score:', f1_sgd)
# print('ROC AUC:', roc_auc_sgd)

# pred_knn = cross_val_predict(knn_classifier, train_set.drop(['label', 'label_bool'], axis=1).values, train_set_y_5, cv=3)
# conf_matrix_knn = confusion_matrix(train_set_y_5, pred_knn)
# precision_knn = precision_score(train_set_y_5, pred_knn)
# recall_knn = recall_score(train_set_y_5, pred_knn)
# f1_knn = f1_score(train_set_y_5, pred_knn) 

# # print(pred)
# print('--- KNN Classifier ---')
# print('conf-matrix:', conf_matrix_knn)
# print('precision:', precision_knn)
# print('recall:', recall_knn)
# print('f1 score:', f1_knn)

precisions, recalls, thresholds = precision_recall_curve(train_set_y_5, pred_sgd)
# precision recall curve graph
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall") 
# plt.xlabel("Threshold")
# plt.legend(loc="upper left")
# plt.ylim([0, 1])
# plt.xlim([-700000, 700000])

# recall vs precision graph
# plt.figure(figsize=(8, 6))
# plt.plot(recalls, precisions, "b-", linewidth=2)
# plt.xlabel("Recall", fontsize=16)
# plt.ylabel("Precision", fontsize=16)
# plt.axis([0, 1, 0, 1])


fpr, tpr, thresholds = roc_curve(train_set_y_5, pred_sgd)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2) 
plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()