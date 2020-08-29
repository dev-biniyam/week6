import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE 
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class Data:
    
    def fetch_dataset(self):
        data = pd.read_csv("datasource/bank-additional-full.csv", sep = ";")
        data.drop("duration", axis = 1, inplace=True)
        return data

    def drop_correlated(self, data):
        data.drop("emp.var.rate", axis = 1, inplace = True)
        data.drop("euribor3m", axis = 1, inplace = True)

    def remove_outliers(self, data):
        #apply the IQR method to remove outliers
        data = self.IQR(data, data.age)
        data = self.IQR(data, data.campaign)
        data = self.IQR(data, data["cons.conf.idx"])

    def IQR(self, df, col): 

        ''' returns the dataframe with the outliers removed by calculating the IQR for the feature and 
        filtering the dataframe to be within that range'''

        Q1 = col.quantile(0.25)       #get the 1st quartile
        Q3 = col.quantile(0.75)       #get the 3rd quartile
        IQR = Q3 - Q1                 #get the inter quartile range
        upper_limit = Q3 + 1.5*IQR    #set the upper limit
        lower_limit = Q1 - 1.5*IQR    #set the lower limit
        df = df[(col>lower_limit) & (col<upper_limit)]  #filter the dataframe to be in the IQR
        
        return df

    def encode_features(self, data):
        #encode categorical features using one-hot encoding
        encoded = pd.get_dummies(data, drop_first=True)
        return encoded


    def train_split(self, data):
        train, test = train_test_split(data,  test_size=0.1, train_size=0.9, random_state=1)
        return train, test

    def upsample_minority(self, train):
        #get the classes
        yes = train[train.y_yes==1]
        no = train[train.y_yes==0]

        # upsample minority
        upsampled = resample(yes,
                                replace=True, # sample with replacement
                                n_samples=len(no), # match number in majority class
                                random_state=27) # reproducible results

        # combine majority and upsampled minority
        train = pd.concat([no, upsampled])
        return train


    def target_split(self, train, test):
        #take all but the last feature as a predictor and the last feature as a target training set
        X_train = train.iloc[:,:-1]
        y_train = train.iloc[:,-1]

        #take all but the last feature as a predictor and the last feature as a target for the test set
        X_test = test.iloc[:,:-1]
        y_test = test.iloc[:,-1]

        return X_train, y_train, X_test, y_test

    def scale(self, X_train, X_test):
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

        return X_train, X_test

    def reduce_dimension(self, X_train, X_test):
        X_train = TSNE(n_components=3, n_iter=300).fit_transform(X_train)
        X_test = TSNE(n_components=3, n_iter=300).fit_transform(X_test)

        return X_train, X_test


    def to_df(self, data):
        return pd.DataFrame(data=data)


