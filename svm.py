import numpy as np
import sys

from scipy.sparse import coo_matrix, hstack

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.metrics import f1_score, recall_score, accuracy_score

from nltk.tokenize import TweetTokenizer
from lexicon import lexicon
from readTweetsFromFile import get_tweets

from random import shuffle, seed

from moreFeatures import more_features

# Configuration for feature extractors
tkzr = TweetTokenizer(reduce_len=True)
vectorizer = CountVectorizer(ngram_range=(2, 5), analyzer="char")
bow = CountVectorizer(tokenizer=tkzr.tokenize)
tfidf = TfidfTransformer()
threashold = VarianceThreshold(threshold=0.00001)


def get_ngrams(data, fit=True):
    """
    Creates feature matrix of n-grams forom a list of tweets

    """
    corpus, targets = zip(*data)
    if fit:
        x = vectorizer.fit_transform(corpus)
        x = tfidf.fit_transform(x)
        x = threashold.fit_transform(x)
    else:
        x = vectorizer.transform(corpus)
        x = tfidf.transform(x)
        x = threashold.transform(x)

    # This actually does not make a lot of difference
    pos = set(lexicon.positive)
    neg = set(lexicon.negative)
    x2 = np.zeros((x.shape[0], 2))
    for i, line in enumerate(corpus):
        words = set(line.split(" "))
        x2[i, 0] = len(words & pos) / len(words)
        x2[i, 1] = len(words & neg) / len(words)

    x = hstack((x, coo_matrix(x2), coo_matrix(more_features(corpus))))

    y = targets
    return x, np.array(y)


def get_bow(data, fit=True):
    """
    Creates features matrix with bag-of-words and a few
    additional handpicked features form a list of tweets

    """
    corpus, targets = zip(*data)
    if fit:
        x = bow.fit_transform(corpus)
        x = tfidf.fit_transform(x)
        x = threashold.fit_transform(x)
    else:
        x = bow.transform(corpus)
        x = tfidf.transform(x)
        x = threashold.transform(x)

    # This actually does not make a lot of difference
    pos = set(lexicon.positive)
    neg = set(lexicon.negative)
    x2 = np.zeros((x.shape[0], 2))
    for i, line in enumerate(corpus):
        words = set(line.split(" "))
        x2[i, 0] = len(words & pos) / len(words)
        x2[i, 1] = len(words & neg) / len(words)

    x = hstack((x, coo_matrix(x2), coo_matrix(more_features(corpus))))

    y = targets
    return x, np.array(y)


def main():
    """
    Use: python3 svm.py <set_number>
    set_number: 1, 2 or 3

    """
    set_id = int(sys.argv[1])
    tweets = [(fields[0], fields[-1]) for id, fields in get_tweets(set_id).items()]

    if set_id == 2:
        tweets = [(tw, sent) for tw, sent in tweets if sent != "neutral"]

    seed(4)
    shuffle(tweets)

    split = int(len(tweets)*0.8)
    # This should do even splits based on classes
    train, test = tweets[:split], tweets[split:]

    results = []
    for ff_name, feature_function in [("n-grams", get_ngrams), ("BOW", get_bow)]:
        x, y = feature_function(train)
        X, Y = feature_function(test, fit=False)
        print("Number of features:", x.shape[1])

        classes = np.unique(y)

        for name, data in [("training", y), ("testing", Y)]:
            # Note how un-evenly split the data set is between pos, neut, neg. This is a problem...
            print("Number of %s examples:" % name, len(data))
            for cl in classes:
                print("\t", cl, np.sum(data == cl))

        clfrs = []

        clf = SVC(decision_function_shape='ovr', kernel='linear', class_weight='balanced')
        clfrs.append(("SVC linear, balanced", clf))

        clf = LinearSVC()
        clfrs.append(("LinearSVC", clf))

        clf = LinearSVC(class_weight='balanced')
        #clfrs.append(("LinearSVC balanced", clf))

        depths = [30, 40, 50, 60, 70, 80]
        mins = [5, 10, 15, 20]
        from itertools import product
        for depth, min_samples in product(depths, mins):
            clf = RandomForestClassifier(class_weight='balanced',
                                         max_depth=depth,
                                         n_estimators=100,
                                         min_samples_split=min_samples)
            #clfrs.append(("RndForest %d:%d" % (depth, min_samples), clf))

        print("Start training and testing")
        for name, clf in clfrs:
            print("Training: %s" % name)
            clf.fit(x, y)
            p = clf.predict(X)

            # Semeval report uses macro averaging, orders by recall
            results.append((
                name,
                ff_name,
                recall_score(Y, p, average='macro'),
                f1_score(Y, p, average='macro'),
                accuracy_score(Y, p),
                ", ".join("%2.2f" % score for score in recall_score(Y, p, average=None, labels=classes))
            ))

    results = sorted(results, key=lambda x: x[2], reverse=True)  # Order by recall
    width = (25+10+10+10+10+4+30)
    print("=" * width)
    print("%-25s%10s%10s%10s%10s    %-30s" % ("Classifier", "Features", "Recall", "F1", "Accuracy", "Class recall (%s)" % ",".join(classes)))
    print("-" * width)
    for result in results:
        print("%-25s%10s%10.3f%10.3f%10.3f    %-30s" % result)
    print("=" * width)


if __name__ == "__main__":
    main()
