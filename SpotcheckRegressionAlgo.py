# Load Library

from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# load dataset
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
         'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv("housingdata.csv", names=names)
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

kfold = KFold(n_splits=10, random_state=7)
scoring = 'neg_mean_squared_error'

# ALL LINEAR REGRESSION MODELS

# Linear Regression Model
model1 = LinearRegression()
results1 = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

# Ridge Regression Model
model2 = Ridge()
results2 = cross_val_score(model2, X, y, cv=kfold, scoring=scoring)
print(results2.mean())

# LASSO Regression Model
model3 = Lasso()
results3 = cross_val_score(model3, X, y, scoring=scoring, cv=kfold)
print(results3.mean())

# ElasticNet Regression Model
model4 = ElasticNet()
results4 = cross_val_score(model4, X, y, scoring=scoring, cv=kfold)
print(results4.mean())

# ALL NON-LINEAR REGRESSION MODELS

# K-Nearest NeighborsRegressor
model5 = KNeighborsRegressor()
results5 = cross_val_score(model5, X, y, scoring=scoring, cv=kfold)
print(results5.mean())


# Classification and Regression Trees
model6 = DecisionTreeRegressor()
results6 = cross_val_score(model6, X, y, scoring=scoring, cv=kfold)
print(results6.mean())


# Support Vector Machines
model7 = SVR()
results7 = cross_val_score(model7, X, y, scoring=scoring, cv=kfold)
print(results7.mean())


