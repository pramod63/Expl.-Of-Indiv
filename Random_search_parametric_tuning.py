# Load libraries
import numpy
from sklearn.model_selection import RandomizedSearchCV
from pandas import read_csv
from sklearn.linear_model import Ridge
from scipy.stats import uniform

# load dataset
dataset=read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

param_grid = {'alpha' : uniform()}

model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100,
                             random_state=7)
rsearch.fit(X, y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)