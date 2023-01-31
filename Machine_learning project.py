#boston price prediction machinelearning project:

#liberaries:

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from regressionML import*
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from feature import*
from BayesianML import*
from randomforest import*
#----------------------------------------
                    #load boaston dataset:

df = pd.read_csv('BostonHousing.csv')

print(df)
#----------------------------------------
                     #statistics analysis:

#matrix covariance:
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
print('dataset description\n')
print(df.describe())
plt.show()
#----------------------------------------
                        #preprocessing:

# copy the data
df_max_scaled = df.copy()
  
# apply normalization techniques
for column in df_max_scaled.columns:
    df_max_scaled[column] = df_max_scaled[column]/df_max_scaled[column].abs().max()
      
# view normalized data
#print(df_max_scaled)
df_max_scaled.plot(kind = 'bar')
df = df_max_scaled
#sns.pairplot(df, size=2.5)
#feature selcection:
#feature_selection(df_max_scaled)  #off/on
#plt.tight_layout()
#plt.show()
#----------------------------------------
#dataset shuffle:
df = shuffle(df)
#X = df.iloc[:,0:13]
X = df[[ 'lstat','rm','tax']]  #feature selected
Y = df.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
#------------------------------------------
                         #modelling:

#                           Linear Regression
'''
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")'''

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print("--------------------------------------\n")
print("prediction by Linear regression machinelearning model\n")
#print("The model performance for testing set")
print("--------------------------------------")
#print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#                           Bayesian Regression:
print("--------------------------------------\n")
print("prediction by Bayesian regression machinelearning model\n")
print("--------------------------------------")
bayes(X,Y)
print("--------------------------------------")
print("prediction by RandomForest regression machinelearning model\n")
print("--------------------------------------")
rf(X,Y)
print("--------------------------------------")
