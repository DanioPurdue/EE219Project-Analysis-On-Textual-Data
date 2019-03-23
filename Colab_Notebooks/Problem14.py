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


from utils import *


def get6xWindow(time,datatime_start,datatime_end):
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

def featureExtractionCustomizedSixWindow(json_objects,datatime_start,datatime_end, i):
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]

    for i in range(len(json_objects)):
        time = datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles'))
        parsed_time, window_end_time = get6xWindow(time,datatime_start,datatime_end)
        tweet_text = json_objects[i]['text']
        hour_time = parsed_time + timedelta(hours=1)
        if(i==0):
            window_end_time = min(window_end_time, datatime_start)
        while hour_time <= window_end_time:
            if hour_time in hr2feature:
                hr2feature[hour_time]["num_tweets"] += 1
                hr2feature[hour_time]["num_retweets"] += json_objects[i]['total_citations']
                hr2feature[hour_time]["num_followers"] += json_objects[i]['followers']
                hr2feature[hour_time]["max_followers"] = max(hr2feature[hour_time]["max_followers"], json_objects[i]['followers'])
                hr2feature[hour_time]["user_mentioned_cnt"] += json_objects[i]['user_mentioned_cnt']
                hr2feature[hour_time]["url_cnt"] += json_objects[i]['url_cnt']
            else:
                hr2feature[hour_time] = dict()
                hr2feature[hour_time]["hr_of_day"] = hour_time.hour
                hr2feature[hour_time]["num_tweets"] = 1
                hr2feature[hour_time]["next_num_tweets"] = 0
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
            hr2feature[hour_time]["next_num_tweets"] = 0
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

def featureExtractionCustomizedSixWindowFiveMinutes(json_objects,datatime_start,datatime_end):
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]

    for i in range(len(json_objects)):
        time = datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles'))
        parsed_time, window_end_time = get6xWindow(time,datatime_start,datatime_end)
        tweet_text = json_objects[i]['text']
        window_end_time = min(window_end_time, datatime_end)
        hour_time = parsed_time + timedelta(minutes=5)
        while hour_time <= window_end_time:
            if hour_time in hr2feature:
                hr2feature[hour_time]["num_tweets"] += 1
                hr2feature[hour_time]["num_retweets"] += json_objects[i]['total_citations']
                hr2feature[hour_time]["num_followers"] += json_objects[i]['followers']
                hr2feature[hour_time]["max_followers"] = max(hr2feature[hour_time]["max_followers"], json_objects[i]['followers'])
                hr2feature[hour_time]["user_mentioned_cnt"] += json_objects[i]['user_mentioned_cnt']
                hr2feature[hour_time]["url_cnt"] += json_objects[i]['url_cnt']
            else:
                hr2feature[hour_time] = dict()
                hr2feature[hour_time]["hr_of_day"] = hour_time.hour
                hr2feature[hour_time]["num_tweets"] = 1
                hr2feature[hour_time]["next_num_tweets"] = 0
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
            hour_time += timedelta(minutes=5)

    min_time = min(hr2feature.keys())
    max_time = max(hr2feature.keys())
    hour_time = min_time + timedelta(minutes=5)
    while hour_time < max_time:
        if hour_time not in hr2feature:
            hr2feature[hour_time] = dict()
            hr2feature[hour_time]["hr_of_day"] = hour_time.hour
            hr2feature[hour_time]["num_tweets"] = 0
            hr2feature[hour_time]["next_num_tweets"] = 0
            hr2feature[hour_time]["num_retweets"] = 0
            hr2feature[hour_time]["num_followers"] = 0
            hr2feature[hour_time]["max_followers"] = 0
            hr2feature[hour_time]["sad_emoji_cnt"] = 0
            hr2feature[hour_time]["happy_emoji_cnt"] = 0
            hr2feature[hour_time]["user_mentioned_cnt"] = 0
            hr2feature[hour_time]["url_cnt"] = 0

        hour_time += timedelta(minutes=5)
    return hr2feature


def labelExtractionCustomizedSixWindow(json_objects, parsed_features, datatime_start,datatime_end):
    hr2feature = parsed_features
    for i in range(len(json_objects)):
        time = datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles'))
        hour_time, _ = get6xWindow(time,datatime_start,datatime_end)

        if hour_time in hr2feature:
            hr2feature[hour_time]["next_num_tweets"] += 1
        else:
            hr2feature[hour_time] = dict()
            hr2feature[hour_time]["next_num_tweets"] = 1
            hr2feature[hour_time]["hr_of_day"] = hour_time.hour
            hr2feature[hour_time]["num_tweets"] = 0
            hr2feature[hour_time]["num_retweets"] = 0
            hr2feature[hour_time]["num_followers"] = 0
            hr2feature[hour_time]["max_followers"] = 0
            hr2feature[hour_time]["sad_emoji_cnt"] = 0
            hr2feature[hour_time]["happy_emoji_cnt"] = 0
            hr2feature[hour_time]["user_mentioned_cnt"] = 0
            hr2feature[hour_time]["url_cnt"] = 0
    return hr2feature

def convertDictToNumpySixWindow(parsed_features, feature_list):
    df = pd.DataFrame(parsed_features).T.sort_index()
    df = df[feature_list] # reorder the column based on feature
    df.drop(df.tail(1).index, inplace=True)
    train_set = df.values[:,:-1]
    labels = df.values[:,-1]
    labels = labels.reshape((len(labels), 1))
    return {"features" : train_set, "labels" : labels}

def getJsonObjectsAggregate(filename, json_list):
    with open(file_direct + filename, "r") as jsonfile:
        line = jsonfile.readline()
        while line:
            json_list.append(json.loads(line))
            line = jsonfile.readline()
        return;

# def convertDictToNumpy(parsed_features, feature_list):
#     df = pd.DataFrame(parsed_features).T.sort_index()
#     # create the label column
#     df["next_num_tweets"] = df["num_tweets"].shift(-1)
#     df = df[feature_list] # reorder the column based on feature
#     df.drop(df.tail(1).index, inplace=True)
#     train_set = df.values[:,:-1]
#     labels = df.values[:,-1]
#     labels = labels.reshape((len(labels), 1))
#     return {"features" : train_set, "labels" : labels}

# matplotlib.use('TKAgg')
file_direct = "ECE219_tweet_data_updated/"
filenames = ["updated_tweets_#gohawks.txt",
            "updated_tweets_#gopatriots.txt",
            "updated_tweets_#nfl.txt",
            "updated_tweets_#patriots.txt",
            "updated_tweets_#sb49.txt",
            "updated_tweets_#superbowl.txt"]

datatime_start = createTime(year=2015, month=2, date=1, hour=8)
datatime_end = createTime(year=2015, month=2, date=1, hour=20)

feature_list_updated = ['hr_of_day', 'max_followers',
                        'num_followers', 'num_retweets',
                        'happy_emoji_cnt', 'sad_emoji_cnt',
                        'user_mentioned_cnt', 'url_cnt',
                        'num_tweets' ,'next_num_tweets']

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

print ("================================ Evaluation on Different Time Periods ==========================")
json_objects_list = splitTweetByTimePeriodList(json_objects, datatime_start, datatime_end)

feature_list = list()
label_list = list()
for i in range(len(json_objects_list)):
    print ("================================ Json Index %d ==========================" % (i))
    json_objects = json_objects_list[i]
    print (" Extract feature ")
    if(i==1):
        parsed_features = featureExtractionCustomizedSixWindowFiveMinutes(json_objects,datatime_start, datatime_end)
        parsed_data = labelExtractionCustomizedSixWindow(json_objects, parsed_features, datatime_start,datatime_end)
    else:
        parsed_features = featureExtractionCustomizedSixWindow(json_objects,datatime_start, datatime_end, i)
        parsed_data = labelExtractionCustomizedSixWindow(json_objects, parsed_features, datatime_start,datatime_end)
    print (" Get Numpy Data ")
    train_labels_pair = convertDictToNumpySixWindow(parsed_data, feature_list_updated)
    feature_list.append(train_labels_pair["features"])
    label_list.append(train_labels_pair["labels"])

train_X = np.concatenate((feature_list[0],feature_list[1],feature_list[2]), axis=0)
train_Y = np.concatenate((label_list[0],label_list[1],label_list[2]), axis=0)

print ("================================ Load test data ==========================")

test_file_direct = "ECE219_tweet_test_updated/"

def loadTestFile(file_direct, sample_num, period_num):
    json_objects = list()
    filename = "updated_sample" + str(sample_num) + "_period" + str(period_num) +".txt"
    with open(file_direct + filename, "r") as file:
        line = file.readline()
        while line:
            json_objects.append(json.loads(line))
            line = file.readline()
    parsed_json_objects = featureExtractTest(json_objects)
    return parsed_json_objects

def getMaxTime(json_objects):
    max_time = createTime(year=2000, month=2, date=1, hour=20)
    for i in range(len(json_objects)):
        max_time = max(max_time, datetime.datetime.fromtimestamp(json_objects[i]['citation_date'], pytz.timezone('America/Los_Angeles')))

    return max_time

def featureExtractTest(json_objects):
    max_time = getMaxTime(json_objects)
    hr2feature = dict()
    happy_emoji = [":)", ":-)", ":')", ":]", "=]", ":)"]
    sad_emoji = [":-(", ":'(", ":[", "=["]
    for i in range(len(json_objects)):
        tweet_text = json_objects[i]['text']
        if max_time in hr2feature:
            hr2feature[max_time]["num_tweets"] += 1
            hr2feature[max_time]["num_retweets"] += json_objects[i]['total_citations']
            hr2feature[max_time]["num_followers"] += json_objects[i]['followers']
            hr2feature[max_time]["max_followers"] = max(hr2feature[max_time]["max_followers"], json_objects[i]['followers'])
            hr2feature[max_time]["user_mentioned_cnt"] += json_objects[i]['user_mentioned_cnt']
            hr2feature[max_time]["url_cnt"] += json_objects[i]['url_cnt']
        else:
            hr2feature[max_time] = dict()
            hr2feature[max_time]["hr_of_day"] = max_time.hour
            hr2feature[max_time]["num_tweets"] = 1
            hr2feature[max_time]["num_retweets"] = json_objects[i]['total_citations']
            hr2feature[max_time]["num_followers"] = json_objects[i]['followers']
            hr2feature[max_time]["max_followers"] = json_objects[i]['followers']
            hr2feature[max_time]["happy_emoji_cnt"] = 0
            hr2feature[max_time]["sad_emoji_cnt"] = 0
            hr2feature[max_time]["user_mentioned_cnt"] = json_objects[i]['user_mentioned_cnt']
            hr2feature[max_time]["url_cnt"] = json_objects[i]['url_cnt']

        if hasEmoticon(happy_emoji, tweet_text):
            hr2feature[max_time]["happy_emoji_cnt"] += 1
        if hasEmoticon(sad_emoji, tweet_text):
            hr2feature[max_time]["sad_emoji_cnt"] += 1

    return hr2feature

json_sample0_period1 = loadTestFile(test_file_direct, 0,1)
json_sample0_period2 = loadTestFile(test_file_direct, 0,2)
json_sample0_period3 = loadTestFile(test_file_direct, 0,3)
json_sample1_period1 = loadTestFile(test_file_direct, 1,1)
json_sample1_period2 = loadTestFile(test_file_direct, 1,2)
json_sample1_period3 = loadTestFile(test_file_direct, 1,3)
json_sample2_period1 = loadTestFile(test_file_direct, 2,1)
json_sample2_period2 = loadTestFile(test_file_direct, 2,2)
json_sample2_period3 = loadTestFile(test_file_direct, 2,3)

test_file_list = [json_sample0_period1, json_sample0_period2,
                 json_sample0_period3, json_sample1_period1,
                 json_sample1_period2, json_sample1_period3,
                 json_sample2_period1, json_sample2_period2,
                 json_sample2_period3]

print ("================================ Fit on OLS model ==========================")

ols = sm.OLS(train_Y, train_X)
res_vals = ols.fit()
pred_vals = res_vals.predict(train_X)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(train_Y, pred_vals))
print("R2_score: %.2f" % r2_score(train_Y, pred_vals))
print("P values: ", res_vals.pvalues)
print ("summary: ", res_vals.summary())

for i in range(len(test_file_list)):
    df = pd.DataFrame(test_file_list[i]).T.sort_index()
    df = df[feature_list_updated[:-1]]
    test_X = df.values
    prediction = res_vals.predict(test_X)
    print ("Test File " + str(i) + ", Number of tweet in next window: ", str(prediction))


print ("================================ Fit on Random Forest Regressor ==========================")

model = RandomForestRegressor(max_depth=None, max_features='auto', min_samples_leaf=1,min_samples_split=2,n_estimators=200 )
model.fit(train_X, train_Y)
pred_vals = model.predict(train_X)
print("Mean squared error: %.2f" % mean_squared_error(train_Y, pred_vals))
print("R2_score: %.2f" % r2_score(train_Y, pred_vals))

for i in range(len(test_file_list)):
    df = pd.DataFrame(test_file_list[i]).T.sort_index()
    df = df[feature_list_updated[:-1]]
    test_X = df.values
    prediction = model.predict(test_X)
    print ("Test File " + str(i) + ", Number of tweet in next window: ", str(prediction))


print ("================================ Fit on Gradient Boost Regressor ==========================")

model = GradientBoostingRegressor(max_depth=80, max_features='sqrt', min_samples_leaf=4,min_samples_split=10,n_estimators=2000)
model.fit(train_X, train_Y)
pred_vals = model.predict(train_X)
print("Mean squared error: %.2f" % mean_squared_error(train_Y, pred_vals))
print("R2_score: %.2f" % r2_score(train_Y, pred_vals))

for i in range(len(test_file_list)):
    df = pd.DataFrame(test_file_list[i]).T.sort_index()
    df = df[feature_list_updated[:-1]]
    test_X = df.values
    prediction = model.predict(test_X)
    print ("Test File " + str(i) + ", Number of tweet in next window: ", str(prediction))


print ("================================ Fit on Neural Network ==========================")

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
# train_Y = np.ravel(train_Y)
nn = MLPRegressor(hidden_layer_sizes=(20, ), learning_rate_init=0.01, max_iter=50000)
nn.fit(train_X, train_Y)
pred_vals = nn.predict(train_X)
print("Mean squared error: %.2f" % mean_squared_error(train_Y, pred_vals))
print("R2_score: %.2f" % r2_score(train_Y, pred_vals))

for i in range(len(test_file_list)):
    df = pd.DataFrame(test_file_list[i]).T.sort_index()
    df = df[feature_list_updated[:-1]]
    test_X = df.values
    test_X = scaler.fit_transform(test_X)
    prediction = nn.predict(test_X)
    print ("Test File " + str(i) + ", Number of tweet in next window: ", str(prediction))
