#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import cPickle

import numpy as np

enron_data_handler = open("../final_project/final_project_dataset.pkl", "rb")
enron_data = cPickle.load(enron_data_handler)
enron_data_handler.close()

# length of dataset
print len(enron_data)


# numer of features for each person in the dataset
name_of_first_person_in_the_list = list(enron_data)[0]
list_of_features_for_first_person_in_the_list = list(enron_data[name_of_first_person_in_the_list])

print len(list_of_features_for_first_person_in_the_list)


# how many persons of interest are in the dataset
names_of_all_people_in_the_list = list(enron_data)
poi_names = [names_of_all_people_in_the_list[ii] for ii in range(0, len(enron_data)) if enron_data[names_of_all_people_in_the_list[ii]]['poi'] == 1]

print len(poi_names)


# What is the total value of the stock belonging to James Prentice?
print enron_data["PRENTICE JAMES"]['total_stock_value']

# How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data["COLWELL WESLEY"]['from_this_person_to_poi']

# What's the value of stock options exercised by Jeffrey K Skilling?
print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']

# How many folks in this dataset have a quantified salary? What about a known email address?
people_with_salary = [names_of_all_people_in_the_list[ii] for ii in range(0, len(enron_data)) if enron_data[names_of_all_people_in_the_list[ii]]['salary'] != 'NaN']
people_with_known_email_address = [names_of_all_people_in_the_list[ii] for ii in range(0, len(enron_data)) if enron_data[names_of_all_people_in_the_list[ii]]['email_address'] != 'NaN']

print len(people_with_salary)
print len(people_with_known_email_address)

# How many people in the E+F dataset (as it currently exists) have 'NaN' for their total payments? What percentage of people in the dataset as a whole is this?
people_with_no_total_payments = [names_of_all_people_in_the_list[ii] for ii in range(0, len(enron_data)) if enron_data[names_of_all_people_in_the_list[ii]]['total_payments'] == 'NaN']

print len(people_with_no_total_payments)

# How many POIs in the E+F dataset have 'NaN' for their total payments? What percentage of POI's as a whole is this?
poi_names_with_no_total_payments = [poi_names[ii] for ii in range(0, len(poi_names)) if enron_data[poi_names[ii]]['total_payments'] == np.NaN]

print len(poi_names_with_no_total_payments)

# If a machine learning algorithm were to use total_payments as a feature, would you expect it to associate a “NaN” value with POIs or non-POIs?