# 2017-11-15

import sys
import os
import csv


# Input: 1) subtask, which is an integer (either 1, 2 or 3; corresponding to
#           subtask 1, 2 or 3). This variable is used to access the textfile
#           corresponding to that subtask.
# Does:     reads all the tweets in the textfile and stores each tweet along
#           with its sentiment ranking, ID and topic in a dictionary, which in
#           turn is stored in a list.
# Returns:  a list containing each such "tweet_dict".
def read_file(subtask):
    # The files are located in a subfolder to SemEval. Fetch them.
    path_to_training_folder = str(os.getcwd()) + "\\semEval_train_2016\\"
    # Create a list of all the files in that directory.
    list_of_training_files = []
    for found_file in os.listdir(path_to_training_folder):
        if found_file.endswith(".txt"):
            list_of_training_files.append(found_file)
    # Fetch the correct file. If the subtask is 1, we want to fetch the file on
    # index 1-1=0 in list_of_training_files.
    path_to_training_file = path_to_training_folder + "\\" + list_of_training_files[subtask-1]

    # This dictionary of tweets will have a given tweet's number as its key
    # and its contents as the key's value.
    list_of_tweet_dicts = []
    with open(path_to_training_file, 'r') as f:
        content = f.readlines()
        tweet_number = 1
        # Each row in the training file corresponds to a tweet.
        for row in content:
            if subtask == 1:
                tweet, sentiment = split_into_tweet_and_sentiment_data(row, subtask)
                this_tweets_dict = {"ID": tweet_number, "Tweet": tweet, "Sentiment": sentiment}
            if subtask == 2 or subtask == 3:
                tweet, sentiment, topic = split_into_tweet_and_sentiment_data(row, subtask)
                this_tweets_dict = {"ID": tweet_number, "Tweet": tweet, "Sentiment": sentiment, "Topic": topic}
            list_of_tweet_dicts.append(this_tweets_dict)
            tweet_number += 1

    return list_of_tweet_dicts


# Input: a single unmodified tweet, i.e. a tweet which still contains the
#        sentiment data at the end of the tweet. This sentiment data/ranking is
#        not part of the actual tweet. Thus, they should be separated.
# Does:  separates the sentiment data (the sentiment ranking) from the tweet and
#        creates a list from the separated data. The tweet will be in index 0 of
#        the list and the sentiment data will be in index -1 of the list. If the
#        subtask also contains a topic, the topic will be saved in index 1.
# Returns: the modified tweet (which is now a list).
def split_into_tweet_and_sentiment_data(tweet, subtask):
    # print(tweet)
    if subtask == 1:
        tweet, sentiment = tweet.rsplit('\t', 1)
        sentiment = sentiment.strip("\n")
        return tweet, sentiment
    if subtask == 2 or subtask == 3:
        tweet, sentiment = tweet.rsplit('\t', 1)
        sentiment = sentiment.strip("\n")
        tweet, topic = tweet.rsplit('\t', 1)
        return tweet, sentiment, topic


def make_csv_of_tweet_dicts(input_list, subtask):
    path_to_training_folder = str(os.getcwd()) + "\\semEval_train_2016"
    list_of_training_files = os.listdir(path_to_training_folder)
    for training_file in list_of_training_files:
        if subtask == 1 and training_file.endswith("A.csv"):
            csv_file = path_to_training_folder + "\\" + training_file
            fieldnames = ['ID', 'Tweet', 'Sentiment']
        if subtask == 2 and training_file.endswith("B.csv"):
            csv_file = path_to_training_folder + "\\" + training_file
            fieldnames = ['ID', 'Tweet', 'Sentiment', 'Topic']
        if subtask == 3 and training_file.endswith("C.csv"):
            csv_file = path_to_training_folder + "\\" + training_file
            fieldnames = ['ID', 'Tweet', 'Sentiment', 'Topic']

    with open(csv_file, 'w', newline='') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        counter = 0
        for tweet_dictionary in input_list:
            writer.writerow(tweet_dictionary)
            counter += 1
        print(counter)
    return


def main():
    subtask = 3
    list_of_tweet_dicts = read_file(subtask)
    print(len(list_of_tweet_dicts))
    make_csv_of_tweet_dicts(list_of_tweet_dicts, subtask)


if __name__=="__main__":
    sys.exit(main())
