import pandas as pd
import numpy as np
from pandas import set_option

# importing the dataset


filename = pd.read_csv('diabetes.csv')
names = [ ' preg ' , ' plas ' , ' pres ' , ' skin ' , ' test ' , ' mass ' , ' pedi ' , ' age ' , ' class ' ]
dataset = pd.read_csv('diabetes.csv', names=names)
peek = dataset.head(20)
shape = dataset.shape
types = dataset.dtypes
data = filename.describe().transpose()
filename.groupby('Outcome').size()
corr = filename.corr(method='pearson')
skew = filename.skew()

from matplotlib import pyplot
filename.hist()
pyplot.show()
filename.plot(kind= ' density ' , subplots=True, layout=(3,3), sharex=False)
pyplot.show()
filename.plot(kind= ' box ' , subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

data = read_csv(filename, names=names)
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


from pandas.tools.plotting import scatter_matrix
scatter_matrix(filename)
pyplot.show()


# Rescale data (between 0 and 1)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
filename = 'diabetes.csv'
names = [ ' preg ' , ' plas ' , ' pres ' , ' skin ' , ' test ' , ' mass ' , ' pedi ' , ' age ' , ' class ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
filename1 = ' diabetes.csv '
names = [ ' preg ' , ' plas ' , ' pres ' , ' skin ' , ' test ' , ' mass ' , ' pedi ' , ' age ' , ' class ' ]
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

array.dtype