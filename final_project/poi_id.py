#!/usr/bin/python

import sys
import pickle
from matplotlib import pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"): ## from k_means_cluster.py
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

def search_outlier(data_dict):
    features_list = ['poi', 
        'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
        'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'
        ]

    # features_list = ['poi', 
    #     'salary', 'bonus'
    #     ]

    data = featureFormat(data_dict, features_list)
    poi, finance_features = targetFeatureSplit(data)

    # for idx_feature1 in [ii for ii in range(0, len(features_list) - 2)]:
    #     for idx_feature2 in [jj for jj in range(idx_feature1 + 1, len(features_list) - 1)]:
    #         print str(idx_feature1) + '-' + str(idx_feature2)

    for idx_feature1 in [ii for ii in range(0, len(features_list) - 2)]:
        for idx_feature2 in [jj for jj in range(idx_feature1 + 1, len(features_list) - 1)]:
            for idx, point in enumerate(finance_features):

                value_feature1 = point[idx_feature1]
                value_feature2 = point[idx_feature2]

                value_poi = poi[idx]

                if value_poi:
                    plt.scatter(value_feature1, value_feature2, color='r', marker='*')
                else:
                    plt.scatter(value_feature1, value_feature2, color='b')

            name_feature1 = features_list[idx_feature1 + 1]
            name_feature2 = features_list[idx_feature2 + 1]

            plt.xlabel(name_feature1)
            plt.ylabel(name_feature2)
            plt.savefig('outlier-search.' + name_feature1 + '-' + name_feature2 + '.png')


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
features_list = ['poi', 'restricted_stock_deferred', 'director_fees']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0) # about this one we know already from previous lesson, so no need to explain it

# search_outlier(data_dict)

# salary - deferral_payments - there seems to be really high value for deferral_payments and since there isn't that much information we can get out of this feature, we might as well just drop it 


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


def study_feature_importance_1(data_dict):
    features_list = ['poi', 
        'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
        'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'
    ]
    
    features_list.remove('deferred_income')
    features_list.remove('other')
    features_list.remove('bonus')
    features_list.remove('shared_receipt_with_poi')
    features_list.remove('to_messages')
    features_list.remove('from_messages')
    features_list.remove('total_payments')
    features_list.remove('expenses')
    features_list.remove('from_poi_to_this_person')
    features_list.remove('exercised_stock_options')
    features_list.remove('long_term_incentive')
    features_list.remove('restricted_stock') # 1
    features_list.remove('salary') # 1

    features_list.remove('deferral_payments')
    features_list.remove('total_stock_value')
    features_list.remove('from_this_person_to_poi') # 1
    features_list.remove('loan_advances') # 1

    print features_list

    data = featureFormat(data_dict, features_list)
    poi, finance_features = targetFeatureSplit(data)

    X_train, X_test, y_train, y_test = train_test_split(finance_features, poi, test_size=0.3, random_state=42)

    print len(X_train)

    X_train = X_train[0:20]
    y_train = y_train[0:20]

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print accuracy

    print [(ii, clf.feature_importances_[ii]) for ii in range(0, len(clf.feature_importances_)) if clf.feature_importances_[ii] > 0.2]
    # print features_list[7]
    # print features_list[11]
    # print features_list[5]
    # print features_list[17]
    # print features_list[12]
    # print features_list[14]
    # print features_list[3]
    # print features_list[7]
    # print features_list[12]
    # print features_list[6]
    # print features_list[7]
    # print features_list[6]
    # print features_list[1]
    # print features_list[1]
    # print features_list[4]
    # print features_list[4]
    # print features_list[1]

    # => ['poi', 'restricted_stock_deferred', 'director_fees']


# study_feature_importance_1(data_dict)
