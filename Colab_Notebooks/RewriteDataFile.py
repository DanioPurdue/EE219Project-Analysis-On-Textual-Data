import json

file_direct = "ECE219_tweet_test/"
for i in range(3):
    for j in range(1,4):
        filename = "sample" + str(i) + "_period" + str(j) + ".txt"
        dest_file = open("ECE219_tweet_test_updated/updated_"+filename, "w")
        with open(file_direct + filename, "r") as file:
            line = file.readline()
            while line:
                json_object = json.loads(line)
                json_lite = dict()
                json_lite['followers'] = json_object['author']['followers']
                json_lite['total_citations'] = json_object['metrics']['citations']['total']
                json_lite['citation_date'] = json_object['citation_date']
                json_lite['followers'] = json_object['author']['followers']
                json_lite['user_mentioned_cnt'] = len(json_object['tweet']['entities']['user_mentions'])
                json_lite['url_cnt'] = len(json_object['tweet']['entities']['urls'])
                json_lite['text'] = json_object['tweet']['text']
                json.dump(json_lite, dest_file)
                dest_file.write('\n')
                line = file.readline()
        dest_file.close()
