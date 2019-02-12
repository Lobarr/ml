import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

df = pd.read_csv('./sonar.csv', header=None)

#dependent and independent variables
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

train_set = None
test_set = None

#training and test set split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_index, test_index in split.split(X, y):
  train_set = df.loc[train_index]
  test_set = df.loc[test_index]

# pipeline for dimentionality reduction preserving 90% variance
dim_rec_pipeline = Pipeline([
  ('dim-rec', PCA(n_components=0.9, random_state=42))
])

le = LabelEncoder()
le.fit(['M', 'R'])

train_set_prepared = dim_rec_pipeline.fit_transform(train_set.iloc[:, :-1])
train_set_labels = le.transform(train_set.iloc[:, -1])
test_set_prepared = dim_rec_pipeline.fit_transform(test_set.iloc[:, :-1])
test_set_labels = le.transform(test_set.iloc[:, -1])

svc_clf = SVC(max_iter=9, tol=np.infty, random_state=42)
svc_clf.fit(train_set_prepared, train_set_labels)

sgd_clf = SGDClassifier(max_iter=8, tol=np.infty, random_state=42)
sgd_clf.fit(train_set_prepared, train_set_labels)

pred_svc = cross_val_predict(svc_clf, train_set_prepared, train_set_labels, cv=3)
pred_sgd = cross_val_predict(sgd_clf, train_set_prepared, train_set_labels, cv=3)

conf_matrix_svc = confusion_matrix(train_set_labels, pred_svc)
precision_svc = precision_score(train_set_labels, pred_svc)
recall_svc = recall_score(train_set_labels, pred_svc)
f1_svc = f1_score(train_set_labels, pred_svc) 
roc_auc_svc = roc_auc_score(train_set_labels, pred_svc)

print('--- SVC Classifier ---')
print('conf-matrix:', conf_matrix_svc)
print('precision:', precision_svc)
print('recall:', recall_svc)
print('f1 score:', f1_svc)
print('ROC AUC:', roc_auc_svc)

conf_matrix_sgd = confusion_matrix(train_set_labels, pred_sgd)
precision_sgd = precision_score(train_set_labels, pred_sgd)
recall_sgd = recall_score(train_set_labels, pred_sgd)
f1_sgd= f1_score(train_set_labels, pred_sgd) 
roc_auc_sgd = roc_auc_score(train_set_labels, pred_sgd)


print('--- SGD Classifier ---')
print('conf-matrix:', conf_matrix_sgd)
print('precision:', precision_sgd)
print('recall:', recall_sgd)
print('f1 score:', f1_sgd)
print('ROC AUC:', roc_auc_sgd)


precisions_sgd, recalls_sgd, thresholds_sgd = precision_recall_curve(train_set_labels, pred_sgd)
plt.figure(figsize=(8, 6))
plt.plot(recalls_sgd, precisions_sgd, "b-", linewidth=2)
plt.title("SGDClassifier")
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.axis([0, 1, 0, 1])

fpr_sgd, tpr_sgd, _ = roc_curve(train_set_labels, pred_sgd)
plt.figure(figsize=(8, 6))
plt.plot(fpr_sgd, tpr_sgd, linewidth=2) 
plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.title("SGDClassifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

precisions_svc, recalls_svc, thresholds_svc = precision_recall_curve(train_set_labels, pred_svc)
plt.figure(figsize=(8, 6))
plt.plot(recalls_svc, precisions_svc, "b-", linewidth=2)
plt.title("SVClassifier")
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.axis([0, 1, 0, 1])

fpr_svc, tpr_svc, _ = roc_curve(train_set_labels, pred_sgd)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svc, tpr_svc, linewidth=2) 
plt.plot([0, 1], [0, 1], 'k--') 
plt.axis([0, 1, 0, 1])
plt.title("SVClassifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
# print(pred_sgd)