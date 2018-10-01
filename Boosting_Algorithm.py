# Load Libraries
from pandas import read_csv
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load dataset
dataset = read_csv("diabetes.csv")
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

seed = 7
num_trees = 30
kfold = KFold(n_splits=10, random_state=seed)
model1 = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results1 = cross_val_score(model1, X, y, cv=kfold)
print(results1.mean())

# Stochastic Gradient Boosting
model2 = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results2 = cross_val_score(model2, X, y, cv=kfold)
print(results2.mean())
