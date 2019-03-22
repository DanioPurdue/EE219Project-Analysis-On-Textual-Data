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


def get6xWindow(time):
    if time < datatime_start or time > datatime_end:
        parsed_time = time - datetime.timedelta(minutes=time.minute,
                                                  seconds = time.second,
                                                  microseconds = time.microsecond)
        window_end_time = parsed_time + datetime.timedelta(hours = 6)
    else:
        parsed_time = time - datetime.timedelta(minutes=time.minute%5,
                                                  seconds = time.second,
                                                  microseconds = time.microsecond)
        window_end_time = parsed_time + datetime.timedelta(minutes=30)
    return parsed_time, window_end_time

def featureExtractionCustomizedSixWindow(json_objects):
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]

    for i in range(len(json_objects)):
        time = datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles'))
        parsed_time, window_end_time = get6xWindow(time)
        tweet_text = json_objects[i]['text']
        hour_time = parsed_time + timedelta(hours=1)
        while hour_time <= window_end_time:
            if hour_time in hr2feature:
                hr2feature[hour_time]["num_tweets"] += 1
                hr2feature[hour_time]["num_retweets"] += json_objects[i]['total_citations']
                hr2feature[hour_time]["num_followers"] += json_objects[i]['followers']
                hr2feature[hour_time]["max_followers"] = max(hr2feature[parsed_time]["max_followers"], json_objects[i]['followers'])
                hr2feature[hour_time]["user_mentioned_cnt"] += json_objects[i]['user_mentioned_cnt']
                hr2feature[hour_time]["url_cnt"] += json_objects[i]['url_cnt']
            else:
                hr2feature[hour_time] = dict()
                hr2feature[hour_time]["hr_of_day"] = parsed_time.hour
                hr2feature[hour_time]["num_tweets"] = 1
                hr2feature[hour_time]["num_retweets"] = json_objects[i]['total_citations']
                hr2feature[hour_time]["num_followers"] = json_objects[i]['followers']
                hr2feature[hour_time]["max_followers"] = json_objects[i]['followers']
                hr2feature[hour_time]["happy_emoji_cnt"] = 0
                hr2feature[hour_time]["sad_emoji_cnt"] = 0
                hr2feature[hour_time]["user_mentioned_cnt"] = json_objects[i]['user_mentioned_cnt']
                hr2feature[hour_time]["url_cnt"] = json_objects[i]['url_cnt']

            if hasEmoticon(happy_emoji, tweet_text):
                hr2feature[hour_time]["happy_emoji_cnt"] += 1
            if hasEmoticon(sad_emoji, tweet_text):
                hr2feature[hour_time]["sad_emoji_cnt"] += 1
            hour_time += timedelta(hours=1)

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

def getJsonObjectsAggregate(filename, json_list):
    with open(file_direct + filename, "r") as jsonfile:
        line = jsonfile.readline()
        while line:
            json_list.append(json.loads(line))
            line = jsonfile.readline()
        return;

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

# matplotlib.use('TKAgg')
file_direct = "ECE219_tweet_data_updated/"
filenames = ["updated_tweets_#gohawks.txt",
            "updated_tweets_#gopatriots.txt",
            "updated_tweets_#nfl.txt",
            "updated_tweets_#patriots.txt",
            "updated_tweets_#sb49.txt",
            "updated_tweets_#superbowl.txt"]


print ("================================ Question 14 ==========================")
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
parsed_features = featureExtractionCustomizedSixWindow(json_objects)
print (" Get Numpy Data ")
feature_list_updated = ['hr_of_day', 'max_followers',
                        'num_followers', 'num_retweets',
                        'happy_emoji_cnt', 'sad_emoji_cnt',
                        'user_mentioned_cnt', 'url_cnt',
                        'num_tweets' ,'next_num_tweets']
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
