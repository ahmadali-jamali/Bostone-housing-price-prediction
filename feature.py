import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns
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

def feature_selection(df):

    X = df.iloc[:,0:13] #independent columns
    y = df.iloc[:,-1]
    df.rename(columns={'medv':'Price'}, inplace=True)
    y = np.round(df['Price'])
    #Apply SelectKBest class to extract top 5 best features
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score'] #naming the dataframe columns
    print(featureScores)
    print("best 5 features of data base based on chi :")
    print(featureScores.nlargest(5,'Score')) #print 5 best features
''' model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    # Plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()'''
