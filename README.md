# Smart-Crop-Type-Prediction-System-Project
Predicting the smart crop for the farmers
import warnings
warnings.filterwarnings("ignore")
#importing dataset from local drive

from google.colab import files
uploaded = files.upload()
#importing libraries

import pickle
import flask
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from IPython.display import display
#storing dataset in the form of a Pandas dataframe

data = pd.read_csv('train.csv')
#separate data into target variable and feature set

y_all = data['Crop']
X_all = data.drop(['Year', 'Date', 'Field', 'Field Type', 'Crop', 'DOY'], 1)
#data standardization

from sklearn.preprocessing import scale

cols = data.columns.drop(['Year', 'Date', 'Field', 'Field Type', 'Crop', 'DOY'], 1)
for col in cols:
  X_all[col] = scale(X_all[col])
#shuffle and split dataset into training and testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, train_size=0.8, random_state=2, stratify=y_all)
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label='H', average=None), sum(target == y_pred) / float(len(y_pred))
  
def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
 # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1.mean() , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1.mean(), acc))
#to measure training time
from time import time

#build some models
clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed=82)
clf_D = RandomForestClassifier(random_state=80, n_jobs=-1)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_D, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
#tuning some parameters

parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }
#Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

#Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, average='micro')

#Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)
#Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)
# Get the estimator
clf = grid_obj.best_estimator_
print (clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1.max() , acc))
    
f1, acc = predict_labels(clf, X_test, y_test)
print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1.max() , acc))
#serializing our model to a file called model.pkl
pickle.dump(clf, open("model.pkl","wb"))
