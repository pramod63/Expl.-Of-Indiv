# Load Libraries

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Load Data
dataset = read_csv('diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Create Feature Union
features = []
features.append(('PCA', PCA(n_components=3)))
features.append(('Select_Best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# Create Pipeline
estimators = []
estimators.append(('feature_union', feature_union)) # important
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)

# Evaluate Pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
