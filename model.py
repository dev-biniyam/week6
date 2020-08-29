
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import xgboost as xgb
from sklearn.neural_network import MLPClassifier


def train_logistic_classifier(X_train, y_train):
    #initialize logistic regression from sklearn
    log_regression = LogisticRegression()

    #fit the training data into the model and get accuracy scores
    log_regression = train_model(log_regression, X_train, y_train)
    return log_regression


def train_xgboost(X_train, y_train):
    #initialize XGBoost from sklearn
    xg_boost = xgb.XGBClassifier()

    #fit the data and get accuracy scores
    xg_boost = train_model(xg_boost, X_train, y_train)
    return xg_boost


def train_perceptrons(X_train, y_train):
    #initialize Multilayer perceptron classifier from sklearn
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

    #fit the data and get accuracy scores
    clf = train_model(clf, X_train, y_train)
    return clf



def train_model(classifier, X_train, y_train):
    accuracy=[]  #initialize accuracy array
    skf = StratifiedKFold(n_splits=10, random_state=None) #initialize the StratifiedKFold with 10 splits
    skf.get_n_splits(X_train, y_train) #get split indexes
    
    for train_index, test_index in skf.split(X_train,y_train): #get the data for each split index
        print("TRAIN:", train_index, "TEST:", test_index)
        X1_train, X1_test = X_train.iloc[train_index], X_train.iloc[test_index] 
        y1_train, y1_test = y_train.iloc[train_index], y_train.iloc[test_index] 

        classifier.fit(X1_train,y1_train) #fit the data into the model
        prediction=classifier.predict(X1_test) #get predictions
        score = metrics.accuracy_score(prediction,y1_test) #get accuracy of the prediction
        accuracy.append(score) #add the accuracy to the list of accuracies
    print("\nAccuracies: ",accuracy)
    print("\nAverage Accuracy =", round(np.array(accuracy).mean()*100,2),"%")
    print("\nMinimum Accuracy =", round(np.array(accuracy).min()*100,2),"%")     
    print("\nMaximum Accuracy =", round(np.array(accuracy).max()*100,2),"%")
    return classifier 


def get_accuracy(model, X_test, y_test):
    prediction = model.predict(X_test) #get predictions
    score = metrics.accuracy_score(prediction,y_test) #get accuracy of the prediction
    return score