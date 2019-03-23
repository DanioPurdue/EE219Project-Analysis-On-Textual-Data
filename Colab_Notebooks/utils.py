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

# Question 11
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

def creationTimeParserFiveMinutes(unix_time, time_zone = pytz.timezone('America/Los_Angeles')):
    date_object = datetime.datetime.fromtimestamp(unix_time, time_zone)
    date_object = date_object - datetime.timedelta(minutes=date_object.minute%5,
                                                  seconds = date_object.second,
                                                  microseconds = date_object.microsecond)
    return date_object

def featureExtractionCustomizedFiveMinutes(json_objects):
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]

    for i in range(len(json_objects)):
        parsed_time = creationTimeParserFiveMinutes(json_objects[i]['citation_date'])
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
    minute_time = min_time + timedelta(minutes=5)
    while minute_time < max_time:
        if minute_time not in hr2feature:
            hr2feature[minute_time] = dict()
            hr2feature[minute_time]["hr_of_day"] = minute_time.hour
            hr2feature[minute_time]["num_tweets"] = 0
            hr2feature[minute_time]["num_retweets"] = 0
            hr2feature[minute_time]["num_followers"] = 0
            hr2feature[minute_time]["max_followers"] = 0
            hr2feature[minute_time]["sad_emoji_cnt"] = 0
            hr2feature[minute_time]["happy_emoji_cnt"] = 0
            hr2feature[minute_time]["user_mentioned_cnt"] = 0
            hr2feature[minute_time]["url_cnt"] = 0

        minute_time += timedelta(minutes=5)
    return hr2feature

def createTime(year, month, date, hour, minute=0, second=0):
    datetime_end = datetime.datetime(year, month, date, hour, minute,second)
    time_zone = pytz.timezone("America/Los_Angeles")
    datetime_end = time_zone.localize(datetime_end)
    print(datetime_end)
    print("year", datetime_end.year,
          " month", datetime_end.month,
          " day", datetime_end.day,
          " hour", datetime_end.hour,
          " minute", datetime_end.minute,
          " second", datetime_end.second)
    return datetime_end

def splitTweetByTimePeriodList(json_objects, datatime_start, datatime_end):
    first_period_json = list()
    second_period_json = list()
    third_period_json = list()
    for i in range(len(json_objects)):
        parsed_time = datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles'))
        if parsed_time < datatime_start:
            first_period_json.append(json_objects[i])
        elif parsed_time > datatime_end:
            third_period_json.append(json_objects[i])
        else:
            second_period_json.append(json_objects[i])

    print(len(first_period_json), len(second_period_json), len(third_period_json))
    return first_period_json, second_period_json, third_period_json


def GridSearchNeuralNetwork(train_X, train_Y):
    parameters = {'hidden_layer_sizes': [(5),(6),(7),(8),(10),(20),(40),(50),
                                    	(5,5),(5,6),(5,8),(5,10),
                                    	(7,6),(7,8),(7,10),(7,20),
                                    	(10,20),(10,40),(10,50),
                                    	(20,40),(20,50),(40,50),
                                    	(5,7,8),(5,8,10),
                                    	(10,20,50),(20,40,50)],
            	'learning_rate_init': [0.01]
                ｝
    mlp = MLPRegressor(random_state=42, max_iter = 50000)
    grid_search = GridSearchCV(mlp, param_grid=parameters, cv=KFold(5, shuffle=True, random_state=42), scoring='neg_mean_squared_error', n_jobs=4)
    grid_search.fit(train_X, train_Y)
    print ("Best layer architecture: ")
    print (grid_search.best_params_)
    print ("Best neg MSE: ")
    print (grid_search.best_score_)
    print ("All scores: ")
    print (grid_search.cv_results_['mean_test_score'])


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
