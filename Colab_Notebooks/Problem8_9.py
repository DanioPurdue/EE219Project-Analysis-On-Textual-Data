# Question 1
import pandas as pd
import numpy as np
import json
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib
import pytz
import random
import dateutil.parser

# Question 3
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Question 8
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# matplotlib.use('TKAgg')
file_direct = "ECE219_tweet_data_updated/"
filenames = ["updated_tweets_#gohawks.txt", 
            "updated_tweets_#gopatriots.txt", 
            "updated_tweets_#nfl.txt",
            "updated_tweets_#patriots.txt",
            "updated_tweets_#sb49.txt",
            "updated_tweets_#superbowl.txt"]


def getJsonObjectsAggregate(filename, json_list):
    with open(file_direct + filename, "r") as jsonfile:
        line = jsonfile.readline()
        while line:
            json_list.append(json.loads(line))
            line = jsonfile.readline()
        return;

def creationTimeParser(unix_time, time_zone = pytz.timezone('America/Los_Angeles')):
    date_object = datetime.datetime.fromtimestamp(unix_time, time_zone)
    date_object = date_object.replace(minute=0, second=0, microsecond=0)
    return date_object

def featureExtractionCustomized(json_objects):
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]
    
    for i in range(len(json_objects)):
        parsed_time = creationTimeParser(json_objects[i]['citation_date'])
        tweet_text = json_objects[i]['text']
        if parsed_time in hr2feature:
            hr2feature[parsed_time]["num_tweets"] += 1
            hr2feature[parsed_time]["num_retweets"] += json_objects[i]['total_citations']
            hr2feature[parsed_time]["num_followers"] += json_objects[i]['followers']
            hr2feature[parsed_time]["max_followers"] = max(hr2feature[parsed_time]["max_followers"], json_objects[i]['followers'])
            hr2feature[parsed_time]["user_mentioned_cnt"] += json_objects[i]['user_mentioned_cnt']
            hr2feature[parsed_time]["url_cnt"] += json_objects[i]['url_cnt']
        else:
            hr2feature[parsed_time] = dict()
            hr2feature[parsed_time]["hr_of_day"] = parsed_time.hour
            hr2feature[parsed_time]["num_tweets"] = 1
            hr2feature[parsed_time]["num_retweets"] = json_objects[i]['total_citations']
            hr2feature[parsed_time]["num_followers"] = json_objects[i]['followers']
            hr2feature[parsed_time]["max_followers"] = json_objects[i]['followers']
            hr2feature[parsed_time]["happy_emoji_cnt"] = 0
            hr2feature[parsed_time]["sad_emoji_cnt"] = 0
            hr2feature[parsed_time]["user_mentioned_cnt"] = json_objects[i]['user_mentioned_cnt']
            hr2feature[parsed_time]["url_cnt"] = json_objects[i]['url_cnt']
        
        if hasEmoticon(happy_emoji, tweet_text):
            hr2feature[parsed_time]["happy_emoji_cnt"] += 1
        if hasEmoticon(sad_emoji, tweet_text):
            hr2feature[parsed_time]["sad_emoji_cnt"] += 1
            
    min_time = min(hr2feature.keys())
    max_time = max(hr2feature.keys())
    hour_time = min_time + timedelta(hours=1)
    while hour_time < max_time:
        if hour_time not in hr2feature:
            hr2feature[hour_time] = dict()
            hr2feature[hour_time]["hr_of_day"] = hour_time.hour
            hr2feature[hour_time]["num_tweets"] = 0
            hr2feature[hour_time]["num_retweets"] = 0
            hr2feature[hour_time]["num_followers"] = 0
            hr2feature[hour_time]["max_followers"] = 0
            hr2feature[hour_time]["sad_emoji_cnt"] = 0
            hr2feature[hour_time]["happy_emoji_cnt"] = 0
            hr2feature[hour_time]["user_mentioned_cnt"] = 0
            hr2feature[hour_time]["url_cnt"] = 0
            
        hour_time += timedelta(hours=1)
    return hr2feature

def hasEmoticon(emoji_list, tweet_text):
    for emoji in emoji_list:
        if emoji in tweet_text:
            return True
    return False

def convertDictToNumpy(parsed_features, feature_list):
    df = pd.DataFrame(parsed_features).T.sort_index()
    # create the label column
    df["next_num_tweets"] = df["num_tweets"].shift(-1)
    df = df[feature_list] # reorder the column based on feature
    df.drop(df.tail(1).index, inplace=True)
    train_set = df.values[:,:-1]
    labels = df.values[:,-1]
    labels = labels.reshape((len(labels), 1))
    return {"features" : train_set, "labels" : labels}

print ("================================ Question 8-9 ==========================")
json_objects = list()
filename = "Aggregated datafile"
for file_single in filenames:
    f_cnt = 1
    a_val = 66
    print ("+++++++++++++++++++++++++ filename: " + file_single + " +++++++++++++++++++")
    print (" Loading and parsing data ")
    getJsonObjectsAggregate(file_single, json_objects)
print ("json length: ", len(json_objects))
print (" Extract feature ")
parsed_features = featureExtractionCustomized(json_objects)
print (" Get Numpy Data ")
feature_list_updated = ['hr_of_day', 'max_followers', 
                        'num_followers', 'num_retweets', 
                        'happy_emoji_cnt', 'sad_emoji_cnt', 
                        'user_mentioned_cnt', 'url_cnt',
                        'num_tweets' ,'next_num_tweets']
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
parameters = {'max_depth': [10, 20, 40, 60, 80, 100, 200, None], 
'max_features': ['auto', 'sqrt'], 
'min_samples_leaf': [1, 2, 4], 
'min_samples_split': [2, 5, 10], 
'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

print ("--------------Random Forest Regressor------------")
clf = RandomForestRegressor()
grid_search_clf = GridSearchCV(clf, param_grid=parameters, cv=KFold(5, shuffle=True), scoring='neg_mean_squared_error', n_jobs=12)
x_train = np.squeeze(sm.add_constant(train_labels_pair["features"]))
y_train = np.squeeze(train_labels_pair["labels"])
print ("Doing Grid Search.... ")
grid_search_clf.fit(x_train, y_train)
print ("Done Grid Search!")
print ("Best Parameters: ", grid_search_clf.best_params_)
y_true, y_pred = y_train, grid_search_clf.predict(x_train)
print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
print("R2_score: %.2f" % r2_score(y_true, y_pred))
print("Grid scores (cross validation scores) on development set:")
print()
means = grid_search_clf.cv_results_['mean_test_score']
stds = grid_search_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_clf.cv_results_['params']):
    print("mean: %0.3f std: (+/-%0.03f) for %r" % (mean, std * 2, params))

print ("--------------Gradient Boost Regressor------------")
clf = GradientBoostingRegressor()
grid_search_clf = GridSearchCV(clf, param_grid=parameters, cv=KFold(5, shuffle=True), scoring='neg_mean_squared_error', n_jobs=12)
x_train = np.squeeze(sm.add_constant(train_labels_pair["features"]))
y_train = np.squeeze(train_labels_pair["labels"])
print ("Doing Grid Search.... ")
grid_search_clf.fit(x_train, y_train)
print ("Done Grid Search!")
print ("Best Parameters: ", grid_search_clf.best_params_)
y_true, y_pred = y_train, grid_search_clf.predict(x_train)
print("Mean squared error: %.2f" % mean_squared_error(y_true, y_pred))
print("R2_score: %.2f" % r2_score(y_true, y_pred))
print("Grid scores (cross validation scores) on development set:")
print()
means = grid_search_clf.cv_results_['mean_test_score']
stds = grid_search_clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search_clf.cv_results_['params']):
    print("mean: %0.3f std: (+/-%0.03f) for %r" % (mean, std * 2, params))