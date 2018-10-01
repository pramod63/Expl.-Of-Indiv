# Load libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


# Load dataset
dataset = read_csv("diabetes.csv")
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Create feature union
features = []
features.append(('PCA', PCA(n_components=3)))
features.append(('Select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

# Create Pipeline
estimators = []
estimators.append(('Standardize', StandardScaler()))
estimators.append(('LDA', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

# Evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
