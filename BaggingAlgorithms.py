# Load libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Load dataset
dataset = read_csv("diabetes.csv")
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Random Forest

seed = 7
kfold1 = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model1 = BaggingClassifier(base_estimator=cart,
                          n_estimators=num_trees,
                          random_state=seed)
results1 = cross_val_score(model1, X, y, cv=kfold1)
print(results1.mean())

# Bagged Decision Tree
max_features = 3
kfold2 = KFold(n_splits=10, random_state=7)
model2 = RandomForestClassifier(n_estimators=num_trees,
                                max_features=max_features)
results2 = cross_val_score(model2, X, y, cv=kfold2)
print(results2.mean())

# ExtraTreeClassifier
max_features = 7
kfold3 = KFold(n_splits=10, random_state=7)
model3 = ExtraTreesClassifier(n_estimators=num_trees,
                                max_features=max_features)
results3 = cross_val_score(model3, X, y, cv=kfold2)
print(results3.mean())