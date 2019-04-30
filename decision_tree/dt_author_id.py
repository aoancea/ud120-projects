#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time


print features_train.shape

print len(features_train[0])

clf = DecisionTreeClassifier(min_samples_split=40)

training_time = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-training_time, 3), "s"


prediction_time = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-prediction_time, 3), "s"

accuracy = accuracy_score(labels_test, pred)

print accuracy
#########################################################


