# 2017-11-14

import sys
import os


# Input: 1) train_or_test, which is a string defining whether we want to access
#           the training files or the test files.
#        2) subtask, which is an integer (either 1, 2 or 3; corresponding to
#           subtask 1, 2 or 3). This variable is used to access the textfile
#           corresponding to that subtask.
# Does:     reads all the tweets in the textfile specified by the input and
#           stores them in a dictionary.
# Returns:  dictionary of tweets, where each value is the content of a single
#           tweet and the key is the number of that tweet.
def readAndStoreTweets(train_or_test, subtask):
    # The files are located in a subfolder to SemEval. Fetch them.
    path_to_training_folder = os.path.join(str(os.getcwd()), "SemEvalDataOLD", train_or_test)
    # Create a list of all the files in that directory.
    list_of_training_files = os.listdir(path_to_training_folder)
    # Fetch the correct file. If the subtask is 1, we want to fetch the file on
    # index 1-1=0 in list_of_training_files.
    path_to_training_file = os.path.join(path_to_training_folder, list_of_training_files[subtask-1])

    # This dictionary of tweets will have a given tweet's number as its key
    # and its contents as the key's value.
    dict_of_tweets = {}
    with open(path_to_training_file, 'r') as f:
        content = f.readlines()
        tweet_number = 0
        # Each row in the training file corresponds to a tweet.
        for tweet in content:
            dict_of_tweets[tweet_number] = tweet
            tweet_number += 1

    return dict_of_tweets


# Input:  a dictionary of tweets.
# Does:   checks for tweets which contain no "sentiment data", i.e. tweets which
#         for some reason do not contain the data for whether it was ranked as
#         "neutral", "positive", "+2", "-1" or whatever. A handful of tweets in the
#         given textfiles have this error. This function removes such tweets.
# Returns: the modified dictionary of tweets (without faulty tweets).
def removeFaultyTweets(dict_of_tweets):
    tweets_before = len(dict_of_tweets)
    for tweet in dict_of_tweets.values():
        tab_counter = 0
        for character in tweet:
            # Check for tabs ("\t") in each tweet. Faulty tweets contain no tabs.
            if character == "\t":
                tab_counter += 1
        #if tab_counter != 1:
        #    print(tab_counter, tweet)

        # If the current tweet contains no tabs, it's faulty and we remove it
        # from the dict_of_tweets using dictionary comprehension.
        if tab_counter == 0:
            # print(tweet)
            dict_of_tweets = {key:val for key, val in dict_of_tweets.items() if val != tweet}

    tweets_after = len(dict_of_tweets)
    tweets_removed = tweets_before - tweets_after
    print("\n" + str(tweets_removed) + " faulty tweets were removed.")
    return dict_of_tweets


# Input: dictionary of tweets which still contain the sentiment data. For
#        example, subtask A training tweets contain either "neutral", "positive"
#        or "negative" in the end of each tweet as sentiment data for that tweet.
# Does:  modifies each tweet in the whole dictionary by separating the sentiment
#        data from the actual tweet.
# Returns: a dictionary of tweets, where each tweet has been replaced by a list.
#          Each list contains the actual tweet in index 0 and the sentiment data
#          (the sentiment ranking) in index 1.
def modifyTweetsInDictionary(tweet_dictionary, subtask):
    for key, value in tweet_dictionary.items():
        tweet = divideTweetIntoTweetAndSentimentData(value, subtask)
        tweet_dictionary[key] = tweet
    return tweet_dictionary


# Input: a single unmodified tweet, i.e. a tweet which still contains the
#        sentiment data at the end of the tweet. This sentiment data/ranking is
#        not part of the actual tweet. Thus, they should be separated.
# Does:  separates the sentiment data (the sentiment ranking) from the tweet and
#        creates a list from the separated data. The tweet will be in index 0 of
#        the list and the sentiment data will be in index 1 of the list.
# Returns: the modified tweet (which is now a list).
def divideTweetIntoTweetAndSentimentData(tweet, subtask):
    # print(tweet)
    if subtask == 1:
        tweet, sentiment = tweet.rsplit('\t', 1)
        sentiment = sentiment.strip("\n")
        tweet = [tweet, sentiment]
        # a_single_tweets_dict = {"tweet": tweet, "sentiment": sentiment}
    if subtask == 2 or subtask == 3:
        tweet, sentiment = tweet.rsplit('\t', 1)
        sentiment = sentiment.strip("\n")
        tweet, topic = tweet.rsplit('\t', 1)
        tweet = [tweet, topic, sentiment]
        # a_single_tweets_dict = {"tweet": tweet, "topic": topic, "sentiment": sentiment}
    # print(tweet)
    return tweet


def main():
    train_or_test = "train"
    # train_or_test = "test"

    # Choose either subtask 1, subtask 2 or subtask 3 by changing the integer.
    # Subtask A (subtask 1) is deciding a sentiment: POS, NEUTRAL or NEG.
    # Subtask B (subtask 2) is deciding a sentiment given a topic: POS or NEG.
    # Subtask C (subtask 3) is deciding a sentiment from -2 to 2 given a topic.
    subtask = 2

    dict_of_tweets = readAndStoreTweets(train_or_test, subtask)
    dict_of_tweets = removeFaultyTweets(dict_of_tweets)
    dict_of_tweets = modifyTweetsInDictionary(dict_of_tweets, subtask)
    print("\nThis is what a value in the output dictionary looks like:")
    print(dict_of_tweets[0])


if __name__=="__main__":
    sys.exit(main())
