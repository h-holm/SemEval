import random
import sys
import os
from collections import Counter
import nltk
import regex as re
# nltk.download()
import pandas as pd
# from emoticons import EmoticonDetector
import re as regex
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
# from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
# from plotly.graph_objs import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time
# import gensim
# from time import time
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV


# plotly.offline.init_notebook_mode()


POSITIVE_SMILEYS = """:-) :) :-] :] :-3 :3 :-> :> 8-) 8) :-} :} :o) :c) :^) =]
                      =) :-D :D 8D 8-D x-D xD x-D XD =D =3 =3 :-)) :-))) :))
                      :))) =)) :\'-) :\') :-* :* ;-) ;) ;-] ;] ;D :-P :P XP xP
                      :PP :p :-p :P :-P O:-) O:) XO""".split()
POSITIVE_PATTERN = "|".join(map(re.escape, POSITIVE_SMILEYS))

NEGATIVE_SMILEYS = """:-( :( :\'-( :\'( DX D= D; D8 D: D:< D-\': :-. :S :L =L
                        %-) %) :### :-### :-\\ :\\ >:\\ :/ :-/ """.split()
NEGATIVE_PATTERN = "|".join(map(re.escape, NEGATIVE_SMILEYS))


class TwitterData():
    data = []
    processed_data = []
    wordlist = []

    data_model = None
    data_labels = None
    is_testing = False

    def initialize(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0, names=["ID", "Tweet", "Sentiment"])
        self.data = self.data[self.data["Sentiment"].isin(["positive", "negative", "neutral"])]

        self.processed_data = self.data
        self.wordlist = []
        self.data_model = None
        self.data_labels = None


# 1) Remove URLs.
# 2) Remove usernames (mentions).
# 3) Replace emoticons with adjectives.
# 3) Remove special characters (this also removes remaining emojis missed in
#    the previous step.)
# 4) Remove numbers
class TwitterDataCleaner:
    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.replace_emoticon_with_adjective,
                               self.remove_special_chars,
                               self.remove_usernames,
                               self.remove_numbers]:
            yield cleanup_method

    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets.loc[:, "Tweet"].replace(regexp, "", inplace=True)
        return tweets

    def remove_urls(self, tweets):
        return TwitterDataCleaner.remove_by_regex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))

    def remove_usernames(self, tweets):
        return TwitterDataCleaner.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def replace_emoticon_with_adjective(self, tweets):
        # tweets.loc[:, "Tweet"].replace(POSITIVE_PATTERN, 'happy', inplace=True)
        # tweets.loc[:, "Tweet"].replace(NEGATIVE_PATTERN, 'sad', inplace=True)
        for tweet in tweets.loc[:, "Tweet"]:
            # print(tweet)
            found_positives = re.findall(POSITIVE_PATTERN, tweet)
            for emoticon in found_positives:
                tweet += ' happy'
            found_negatives = re.findall(NEGATIVE_PATTERN, tweet)
            for emoticon in found_negatives:
                tweet += ' sad'
            # print(tweet)
        return tweets

    def remove_special_chars(self, tweets):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            tweets.loc[:, "Tweet"].replace(remove, "", inplace=True)
        return tweets

    def remove_numbers(self, tweets):
        return TwitterDataCleaner.remove_by_regex(tweets, regex.compile(r'\w*\d\w*'))


class TwitterData_Cleansing(TwitterData):
    def __init__(self, previous):
        self.processed_data = previous.data

    def cleanup(self, cleaner):
        t = self.processed_data
        for cleanup_method in cleaner.iterate():
            t = cleanup_method(t)

        self.processed_data = t


class TwitterData_TokenStem(TwitterData_Cleansing):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    def stem(self, stemmer=nltk.PorterStemmer()):
        def stem_and_join(row):
            row["Tweet"] = list(map(lambda str: stemmer.stem(str.lower()), row["Tweet"]))
            return row

        self.processed_data = self.processed_data.apply(stem_and_join, axis=1)

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenize_row(row):
            row["Tweet"] = tokenizer(row["Tweet"])
            row["tokenized_text"] = [] + row["Tweet"]
            return row

        self.processed_data = self.processed_data.apply(tokenize_row, axis=1)


class TwitterData_Wordlist(TwitterData_TokenStem):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    whitelist = ["n't","not"]
    wordlist = []

    def build_wordlist(self, min_occurrences=3, max_occurences=500, stopwords=nltk.corpus.stopwords.words("english"),
                       whitelist=None):
        self.wordlist = []
        whitelist = self.whitelist if whitelist is None else whitelist

        if os.path.isfile("semEval_train_2016\\wordlist.csv"):
            word_df = pd.read_csv("semEval_train_2016\\wordlist.csv")
            word_df = word_df[word_df["Occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "Word"])
            return

        words = Counter()
        for idx in self.processed_data.index:
            words.update(self.processed_data.loc[idx, "Tweet"])

        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]

        word_df = pd.DataFrame(data={"Word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                                     "Occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
                               columns=["Word", "Occurrences"])

        word_df.to_csv("semEval_train_2016\\wordlist.csv", index_label="idx")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]


class TwitterData_ExtraFeatures(TwitterData_Wordlist):
    def __init__(self):
        pass

    def build_data_model(self):
        extra_columns = [col for col in self.processed_data.columns if col.startswith("Number_of")]
        label_column = []
        if not self.is_testing:
            label_column = ["Label"]

        columns = label_column + extra_columns + list(
            map(lambda w: w + "_bow",self.wordlist))

        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "Sentiment"]
                labels.append(current_label)
                current_row.append(current_label)

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "Tweet"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels

    def build_features(self):
        def count_by_lambda(expression, word_array):
            return len(list(filter(expression, word_array)))

        def count_occurences(character, word_array):
            counter = 0
            for j, word in enumerate(word_array):
                for char in word:
                    if char == character:
                        counter += 1

            return counter

        def count_by_regex(regex, plain_text):
            return len(regex.findall(plain_text))

        self.add_column("Splitted_text", map(lambda txt: txt.split(" "), self.processed_data["Tweet"]))

        # number of uppercase words
        uppercase = list(map(lambda txt: count_by_lambda(lambda word: word == word.upper(), txt),
                             self.processed_data["Splitted_text"]))
        self.add_column("Number_of_uppercase", uppercase)

        # number of !
        exclamations = list(map(lambda txt: count_occurences("!", txt),
                                self.processed_data["Splitted_text"]))

        self.add_column("Number_of_exclamation", exclamations)

        # number of ?
        questions = list(map(lambda txt: count_occurences("?", txt),
                             self.processed_data["Splitted_text"]))

        self.add_column("Number_of_question", questions)

        # number of ...
        ellipsis = list(map(lambda txt: count_by_regex(regex.compile(r"\.\s?\.\s?\."), txt),
                            self.processed_data["Tweet"]))

        self.add_column("Number_of_ellipsis", ellipsis)

        # number of hashtags
        hashtags = list(map(lambda txt: count_occurences("#", txt),
                            self.processed_data["Splitted_text"]))

        self.add_column("Number_of_hashtags", hashtags)

        # number of mentions
        mentions = list(map(lambda txt: count_occurences("@", txt),
                            self.processed_data["Splitted_text"]))

        self.add_column("Number_of_mentions", mentions)

        # number of quotes
        quotes = list(map(lambda plain_text: int(count_occurences("'", [plain_text.strip("'").strip('"')]) / 2 +
                                                 count_occurences('"', [plain_text.strip("'").strip('"')]) / 2),
                          self.processed_data["Tweet"]))

        self.add_column("Number_of_quotes", quotes)

        # number of urls
        urls = list(map(lambda txt: count_by_regex(regex.compile(r"http.?://[^\s]+[\s]?"), txt),
                        self.processed_data["Tweet"]))

        self.add_column("Number_of_urls", urls)

        """# number of positive emoticons
        ed = EmoticonDetector()
        positive_emo = list(
            map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
                self.processed_data["Splitted_text"]))

        self.add_column("Number_of_positive_emo", positive_emo)

        # number of negative emoticons
        negative_emo = list(map(
            lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
            self.processed_data["Splitted_text"]))

        self.add_column("Number_of_negative_emo", negative_emo)"""

    def add_column(self, column_name, column_content):
        self.processed_data.loc[:, column_name] = pd.Series(column_content, index=self.processed_data.index)


class TwitterData_BagOfWords(TwitterData_Wordlist):
    def __init__(self, previous):
        self.processed_data = previous.processed_data
        self.wordlist = previous.wordlist

    def build_data_model(self):
        label_column = []
        if not self.is_testing:
            label_column = ["Label"]

        columns = label_column + list(
            map(lambda w: w + "_bow",self.wordlist))
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "Sentiment"]
                labels.append(current_label)
                current_row.append(current_label)

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "Tweet"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels


def test_classifier(X_train, y_train, X_test, y_test, classifier):
    log("")
    log("===============================================")
    classifier_name = str(type(classifier).__name__)
    log("Testing " + classifier_name)
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    log("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    log("Predicting time {0}s".format(time() - now))

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    log("=================== Results ===================")
    log("            Negative     Neutral     Positive")
    log("F1       " + str(f1))
    log("Precision" + str(precision))
    log("Recall   " + str(recall))
    log("Accuracy " + str(accuracy))
    log("===============================================")

    return precision, recall, accuracy, f1


def cv(classifier, X_train, y_train):
    log("===============================================")
    classifier_name = str(type(classifier).__name__)
    now = time()
    log("Crossvalidating " + classifier_name + "...")
    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]
    log("Crosvalidation completed in {0}s".format(time() - now))
    log("Accuracy: " + str(accuracy[0]))
    log("Average accuracy: " + str(np.array(accuracy[0]).mean()))
    log("===============================================")
    return accuracy


def log(x):
    #can be used to write to log file
    print(x)


def main():
    # Choose either subtask 1, subtask 2 or subtask 3 by changing the integer.
    # Subtask A (subtask 1) is deciding a sentiment: POS, NEUTRAL or NEG.
    # Subtask B (subtask 2) is deciding a sentiment given a topic: POS or NEG.
    # Subtask C (subtask 3) is deciding a sentiment from -2 to 2 given a topic.
    subtask = 1

    path_to_training_folder = str(os.getcwd()) + "\\semEval_train_2016"
    list_of_training_files = os.listdir(path_to_training_folder)
    for training_file in list_of_training_files:
        if subtask == 1 and training_file.endswith("A.csv"):
            csv_file = path_to_training_folder + "\\" + training_file
        if subtask == 2 and training_file.endswith("B.csv"):
            csv_file = path_to_training_folder + "\\" + training_file
        if subtask == 3 and training_file.endswith("C.csv"):
            csv_file = path_to_training_folder + "\\" + training_file

    # twitter_data = TwitterData()
    twitter_data = TwitterData_ExtraFeatures()
    twitter_data.initialize(csv_file)
    # print(twitter_data.data.head(20))

    twitter_data.build_features()
    # twitter_data = TwitterData_Cleansing(twitter_data)
    twitter_data.cleanup(TwitterDataCleaner())
    # print(twitter_data.processed_data.head(20))

    # twitter_data = TwitterData_TokenStem(twitter_data)
    twitter_data.tokenize()
    twitter_data.stem()
    # print(twitter_data.processed_data.head(20))

    # twitter_data = TwitterData_Wordlist(twitter_data)
    twitter_data.build_wordlist()

    data_model, labels = twitter_data.build_data_model()

    # words = pd.read_csv("semEval_train_2016\\wordlist.csv")

    # twitter_data = TwitterData_BagOfWords(twitter_data)
    # bow, labels = twitter_data.build_data_model()
    # print(bow.head(10))

    """grouped = bow.groupby(["Label"]).sum()
    words_to_visualize = []
    sentiments = ["positive","neutral","negative"]

    #get the most 7 common words for every sentiment
    for sentiment in sentiments:
        words = grouped.loc[sentiment,:]
        words.sort_values(inplace=True,ascending=False)
        for w in words.index[:7]:
            if w not in words_to_visualize:
                words_to_visualize.append(w)"""


    """#visualize it
    plot_data = []
    for sentiment in sentiments:
        plot_data.append(graph_objs.Bar(
                x = [w.split("_")[0] for w in words_to_visualize],
                y = [grouped.loc[sentiment,w] for w in words_to_visualize],
                name = sentiment
        ))

    py.plot({
            "data":plot_data,
            "layout":graph_objs.Layout(title="Most common words across sentiments")
        })"""

    sentiments = ["positive", "neutral", "negative"]
    plots_data_ef = []
    for what in map(lambda o: "Number_of_"+o,["exclamation","hashtags","question"]):
        ef_grouped = data_model[data_model[what]>=1].groupby(["Label"]).count()
        plots_data_ef.append({"data":[graph_objs.Bar(
                x = sentiments,
                y = [ef_grouped.loc[s,:][0] for s in sentiments],
        )], "title":"How feature \""+what+"\" separates the tweets"})


    for plot_data_ef in plots_data_ef:
        py.plot({
                "data":plot_data_ef["data"],
                "layout":graph_objs.Layout(title=plot_data_ef["title"])
        })

    seed = 666
    random.seed(seed)

    # Got 0.601197263398 accuracy with BernoulliNB like this.
    X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow.iloc[:, 0],
                                                    train_size=0.7, stratify=bow.iloc[:, 0],
                                                    random_state=seed)
    precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())

    # adding "def cv"
    nb_acc = cv(BernoulliNB(), bow.iloc[:,1:], bow.iloc[:,0])

if __name__=="__main__":
    sys.exit(main())
