#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import os
import time

# print os.getcwd()

# os.chdir(path)
# sys.path.append("E:\\projects\\machine-learning-tutorials\\ud120-projects\\tools")

sys.path.append("../tools")

# print sys.path
# print sklearn.__version__

from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()

training_time = time.time()

clf.fit(features_train, labels_train)

print "training time:", round(time.time()-training_time, 3), "s"

prediction_time = time.time()

pred = clf.predict(features_test)

print "prediction time:", round(time.time()-prediction_time, 3), "s"

accuracy = accuracy_score(labels_test, pred)

print accuracy
#########################################################