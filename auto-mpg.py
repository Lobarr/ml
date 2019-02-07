'''
--- Overfit Model ---
--Training data--
RMSE     [4.11058074 4.94920092 5.42635058 5.85945691 4.82955358]
Mean RMSE        5.035028544340014
Standard deviation       0.5894073569550409
--Testing data--
RMSE     16.204414355980596 



--- Non-Overfit Model ---
--Training data--
RMSE     [4.34244354 4.23194711 5.46693671 4.95464134 4.38020936]
Mean RMSE        4.675235614680463
Standard deviation       0.46886116094275226
--Testing data--
RMSE     6.466802527011628
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
# from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix

dataset = pd.read_csv('./auto-mpg.csv').drop(['car_name'], axis=1).fillna(0) # drop car_name column

# dependent and independent variables
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#train and test sets
train_set = None
test_set = None

# training and test data split
split = ShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_index, test_index in split.split(X, y):
  train_set = dataset.loc[train_index]
  test_set = dataset.loc[test_index]

train_set_copy = train_set.copy()

# attributes = ['mpg','horsepower', 'weight', 'displacement']
# scatter_matrix(train_set_copy[attributes], figsize=(12, 8))
# print(train_set_copy.info())
# print(train_set_copy.describe())
# train_set_copy.plot(kind='scatter', x='horsepower', y='mpg')
# train_set_copy.plot(kind='scatter', x='weight', y='mpg')
# plt.show()

clean_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='mean', missing_values=0, fill_value=0)),
  ('std_scaler', StandardScaler()),
  # ('polynomial', PolynomialFeatures(degree=2))
])

# train_set_prepared = clean_pipeline.fit_transform(train_set['horsepower'])
train_set_prepared = clean_pipeline.fit_transform(train_set.drop(['mpg', 'cylinders', 'acceleration', 'displacement', 'weight', 'model_year', 'origin'], axis=1))
train_set_labels = train_set['mpg'].copy()
# test_set_prepared = clean_pipeline.fit_transform(test_set['horsepower'])
test_set_prepared = clean_pipeline.fit_transform(test_set.drop(['mpg', 'cylinders', 'acceleration', 'displacement', 'weight', 'model_year', 'origin'], axis=1))
test_set_labels = test_set['mpg'].copy()

train_set_overfit_less = [] # holds horsepowers <= 0 (after normalization)
train_set_overfit_less_labels = [] # holds mpg of horsepowers <= 0 (after normalization)
train_set_overfit_more = [] # holds horsepowers > 0 (after normalization)
train_set_overfit_more_labels = [] # holds mpg of horsepowers >= 0 (after normalization)

#split training set into various pools
for i in range(len(train_set_prepared)):
  if train_set_prepared[i][0] <= 0:
    train_set_overfit_less.append(train_set_prepared[i])
    train_set_overfit_less_labels.append(train_set_labels.iloc[i])
  else:
    train_set_overfit_more.append(train_set_prepared[i])
    train_set_overfit_more_labels.append(train_set_labels.iloc[i])

train_set_overfit_less = np.reshape(train_set_overfit_less, [len(train_set_overfit_less), 1])
train_set_overfit_more = np.reshape(train_set_overfit_more, [len(train_set_overfit_more), 1])

# print(train_set_overfit_less)
# print(train_set_overfit_more)

# linear regression model
lin_reg_overfit = DecisionTreeRegressor()
lin_reg_overfit.fit(train_set_overfit_less, train_set_overfit_less_labels) # trains model with horsepower <= 0 (after normalization)

lin_reg = DecisionTreeRegressor()
lin_reg.fit(train_set_prepared, train_set_labels)

fig, (overfit, non_overfit) = plt.subplots(2, 1, sharey=True)
overfit.scatter(train_set_overfit_less, train_set_overfit_less_labels,color='b')
overfit.set_title('Overfitted')
overfit.set_ylabel('MPG')
overfit.set_xlabel('Horsepower (normalized)')
# overfit.plot(train_set_overfit_less, lin_reg_overfit.predict(train_set_overfit_less),color='k')
overfit.grid(True)

non_overfit.scatter(train_set_prepared, train_set_labels, color='b')
non_overfit.set_title('Non Overfit')
non_overfit.set_ylabel('MPG')
non_overfit.set_xlabel('Horsepower (normalized)')
# non_overfit.plot(train_set_prepared, lin_reg.predict(train_set_prepared),color='k')
non_overfit.grid(True)

fig.subplots_adjust(hspace=.5)
plt.savefig('auto.jpeg')
# plt.show()

print('--- Overfit Model ---')
print('--Training data--')
scores_overfit = cross_val_score(lin_reg_overfit, train_set_overfit_less, train_set_overfit_less_labels, scoring='neg_mean_squared_error', cv=5)
rmse_scores_overfit = np.sqrt(-scores_overfit)
print('RMSE\t',rmse_scores_overfit)
print('Mean RMSE\t', np.mean(rmse_scores_overfit))
print('Standard deviation\t', np.std(rmse_scores_overfit))

print('--Testing data--')
predictions_overfit = lin_reg_overfit.predict(train_set_overfit_more) # test model with horsepower >= 0 (after normalization)
lin_rmse_overfit = np.sqrt(mean_squared_error(train_set_overfit_more, train_set_overfit_more_labels))
print('RMSE\t',lin_rmse_overfit, '\n\n\n')
# print('Predictions:\t', predictions_overfit)
# print('Labels:\t\t', list(train_set_labels))


print('--- Non-Overfit Model ---')
print('--Training data--')
scores = cross_val_score(lin_reg, train_set_prepared, train_set_labels, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-scores)
print('RMSE\t',rmse_scores)
print('Mean RMSE\t', np.mean(rmse_scores))
print('Standard deviation\t', np.std(rmse_scores))

print('--Testing data--')
predictions = lin_reg.predict(test_set_prepared)
lin_rmse = np.sqrt(mean_squared_error(test_set_labels, predictions))
print('RMSE\t',lin_rmse)
# print('Predictions:\t', predictions)
# print('Labels:\t\t', list(train_set_labels))