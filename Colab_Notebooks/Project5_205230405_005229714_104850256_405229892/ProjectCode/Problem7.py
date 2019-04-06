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
top_three_features = [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1)]

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

def trainAndEvaluate(train_labels_pair, feature_list):
    # OLS Top three features and P values
    global top_three_features
    # x_train = np.squeeze(sm.add_constant(train_labels_pair["features"]))
    x_train = np.squeeze(train_labels_pair["features"])
    y_train = np.squeeze(train_labels_pair["labels"])
    ols = sm.OLS(y_train, x_train)
    res_vals = ols.fit()
    pred_vals = res_vals.predict(x_train)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_train, pred_vals))
    print("P values: ", res_vals.pvalues)
    idx_pvals_pair = list()
    for i in range(res_vals.pvalues.shape[0]):
        pair = (res_vals.pvalues[i], i)
        idx_pvals_pair.append(pair)
    idx_pvals_pair = sorted(idx_pvals_pair, key=lambda x: x[0])
    print('idx_pvals_pair: ', idx_pvals_pair)
    print("Top three features")
    for i in range(3):
        top_three_features[i] = (feature_list[idx_pvals_pair[i][1]], idx_pvals_pair[i][1], res_vals.params[idx_pvals_pair[i][1]])
        print("feature:", feature_list[idx_pvals_pair[i][1]], ", P value:", idx_pvals_pair[i][0])
    print ("summary: ", res_vals.summary())
    return pred_vals

#==========================time feature===============================
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
        parsed_time = creationTimeParser(json_objects[i]['citation_date'])
        if parsed_time < datatime_start:
            first_period_json.append(json_objects[i])
        elif parsed_time > datatime_end:
            third_period_json.append(json_objects[i])
        else:
            second_period_json.append(json_objects[i])
            
    return [first_period_json, second_period_json, third_period_json]
#==========================end of time feature===============================

print ("================================ Question 7 ==========================")
json_objects = list()
filename = "Aggregated datafile"
for file_single in filenames:
    f_cnt = 1
    a_val = 66
    print ("+++++++++++++++++++++++++ filename: " + file_single + " +++++++++++++++++++")
    print (" Loading and parsing data ")
    getJsonObjectsAggregate(file_single, json_objects)
feature_list_updated = ['hr_of_day', 'max_followers', 
                        'num_followers', 'num_retweets', 
                        'happy_emoji_cnt', 'sad_emoji_cnt', 
                        'user_mentioned_cnt', 'url_cnt',
                        'num_tweets' ,'next_num_tweets']
print ("================================ Evaluation on the Aggregated Data ==========================")
def getStat(json_objects, hr2cnt):
    # Description:
    # Get the the number of tweets, also features forom the tweet
    list_cnt = 0
    total_followers = 0
    total_tweets = 0
    time_zone = pytz.timezone('America/Los_Angeles')
    for i in range(len(json_objects)):
    # parse the avg tweet per hour
        parsed_time = creationTimeParser(json_objects[i]['citation_date'],  time_zone)
        if parsed_time in hr2cnt:
          hr2cnt[parsed_time] += 1
        else:
          hr2cnt[parsed_time] = 1
        total_followers += json_objects[i]['followers']
        total_tweets += json_objects[i]['total_citations']
    min_time = min(hr2cnt.keys())
    max_time = max(hr2cnt.keys())
    hour_time = min_time + timedelta(hours=1)
    while hour_time < max_time:
        if hour_time not in hr2cnt:
            hr2cnt[hour_time] = 0
        hour_time = hour_time + timedelta(hours=1)
    hr_np = np.fromiter(hr2cnt.values(), dtype=np.int).reshape((len(hr2cnt),1))
    min_time = min(hr2cnt.keys())
    max_time = max(hr2cnt.keys())
    diff_time = max_time - min_time + timedelta(hours=1)
    avg_tweets = hr_np.sum() / (diff_time.total_seconds()/3600)
    return {'avg tweets per hour' : avg_tweets, 
          'avg followers per tweet' : total_followers / len(json_objects),
          'avg retweets per tweet' : total_tweets / len(json_objects)}

hr2cnt = dict()
stat_res = getStat(json_objects, hr2cnt)
print("Question 1 stat res: ", stat_res)

# ## Question 2
print ("================================  Question 2  ============================")
def getTweetCount(json_objects):
    hr2cnt = dict()
    stat = getStat(json_objects, hr2cnt)
    return hr2cnt
    
def getSortedKeys(hr2cnt):
    keylist = sorted(hr2cnt.keys())
    return keylist

def generatePlot(hr2cnt, filename):
    global f_cnt
    sorted_time = getSortedKeys(hr2cnt)
    sorted_vals = [hr2cnt[x] for x in sorted_time]
    plt.plot(sorted_time, sorted_vals)
    print("sorted_vals: ", len(sorted_vals))
    plt.xlabel('time')
    plt.ylabel('num of posts')
    plt.title("Question 7: " +filename)
    plt.show()

def Question2(json_objects, filename):
    hr2cnt = getTweetCount(json_objects)
    keylist = getSortedKeys(hr2cnt)
    generatePlot(hr2cnt, filename)

Question2(json_objects, "Aggregated Data")
parsed_features = featureExtractionCustomized(json_objects)
train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
pred_value = trainAndEvaluate(train_labels_pair,feature_list_updated)

print ("================================ Evaluation on Different Time Periods ==========================")
datatime_start = createTime(year=2015, month=2, date=1, hour=8)
datatime_end = createTime(year=2015, month=2, date=1, hour=20)
json_objects_list = splitTweetByTimePeriodList(json_objects, datatime_start, datatime_end)
for i in range(len(json_objects_list)):
    print ("================================ Json Index %d ==========================" % (i))
    json_objects = json_objects_list[i]
    print (" Extract feature ")
    parsed_features = featureExtractionCustomized(json_objects)
    print (" Get Numpy Data ")
    parsed_features = featureExtractionCustomized(json_objects)
    train_labels_pair = convertDictToNumpy(parsed_features, feature_list_updated)
    pred_value = trainAndEvaluate(train_labels_pair,feature_list_updated)

