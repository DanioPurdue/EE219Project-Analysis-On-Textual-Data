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

# matplotlib.use('TKAgg')
file_direct = "ECE219_tweet_data_updated/"
filenames = ["updated_tweets_#gohawks.txt",
            "updated_tweets_#gopatriots.txt",
            "updated_tweets_#nfl.txt",
            "updated_tweets_#patriots.txt",
            "updated_tweets_#sb49.txt",
            "updated_tweets_#superbowl.txt"]

print ("================================ Question 11-13 ==========================")
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
# parsed_features = featureExtractionCustomized(json_objects)
print (" Get Numpy Data ")
feature_list_updated = ['hr_of_day', 'max_followers',
                        'num_followers', 'num_retweets',
                        'happy_emoji_cnt', 'sad_emoji_cnt',
                        'user_mentioned_cnt', 'url_cnt',
                        'num_tweets' ,'next_num_tweets']
# train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
#
#
# print ("--------------Question 11: Neural Network------------")
# # parsed_features = featureExtractionCustomized(json_aggregate)
# # train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
# train_X = train_labels_pair["features"]
# train_Y = train_labels_pair["labels"]
# train_Y = np.ravel(train_Y)
# GridSearchNeuralNetwork(train_X, train_Y)
#
# print ("--------------Question 12: Neural Network + Standardization------------")
scaler = StandardScaler()
#
# # parsed_features = featureExtractionCustomized(json_aggregate)
# # train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
# # train_X = train_labels_pair["features"]
# # train_Y = train_labels_pair["labels"]
# train_X = scaler.fit_transform(train_X)
# # train_Y = scaler.fit_transform(train_Y)
# # train_Y = np.ravel(train_Y)
# GridSearchNeuralNetwork(train_X, train_Y)

print ("--------------Question 13: Neural Network + Standardization For each period------------")

datatime_start = createTime(year=2015, month=2, date=1, hour=8)
datatime_end = createTime(year=2015, month=2, date=1, hour=20)

first_period_json, second_period_json, third_period_json = splitTweetByTimePeriodList(json_objects, datatime_start, datatime_end)

print ("--------------Period 1------------")

parsed_features = featureExtractionCustomized(first_period_json)
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
train_X = train_labels_pair["features"]
train_Y = train_labels_pair["labels"]
train_X = scaler.fit_transform(train_X)
# train_Y = scaler.fit_transform(train_Y)
train_Y = np.ravel(train_Y)

GridSearchNeuralNetwork(train_X, train_Y)

print ("--------------Period 2------------")

parsed_features = featureExtractionCustomizedFiveMinutes(second_period_json)
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
train_X = train_labels_pair["features"]
train_Y = train_labels_pair["labels"]
train_X = scaler.fit_transform(train_X)
# train_Y = scaler.fit_transform(train_Y)
train_Y = np.ravel(train_Y)

GridSearchNeuralNetwork(train_X, train_Y)

print ("--------------Period 3------------")

parsed_features = featureExtractionCustomized(third_period_json)
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
train_X = train_labels_pair["features"]
train_Y = train_labels_pair["labels"]
train_X = scaler.fit_transform(train_X)
# train_Y = scaler.fit_transform(train_Y)
train_Y = np.ravel(train_Y)

GridSearchNeuralNetwork(train_X, train_Y)
