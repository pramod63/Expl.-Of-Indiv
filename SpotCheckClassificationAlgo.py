# Load Libraries

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Load dataset

database = read_csv("diabetes.csv")
X = database.iloc[:, 0:8].values
y = database.iloc[:, 8].values

num_folds = 10
kfold = KFold(n_splits=10, random_state=7)

# Logistic Regression
model1 = LogisticRegression()
results1 = cross_val_score(model1, X, y, cv=kfold)
print(results1.mean())

# LinearDiscriminantAnalysis
model2 = LinearDiscriminantAnalysis()
results2 = cross_val_score(model2, X, y, cv=kfold)
print(results2.mean())

# K-Nearrest Neighbours
model3 = KNeighborsClassifier()
results3 = cross_val_score(model3, X, y, cv=kfold)
print(results3.mean())

# Naive Baves
model4 = GaussianNB()
results4 = cross_val_score(model4, X, y, cv=kfold)
print(results4.mean())

# DecisionTreeClassifier
model5 = DecisionTreeClassifier()
results5 = cross_val_score(model5, X, y, cv=kfold)
print(results5.mean())

# DecisionTree
model6 = SVC()
results6 = cross_val_score(model6,X, y, cv=kfold)
print(results6.mean())