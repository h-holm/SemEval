import numpy as np

from scipy.sparse import coo_matrix, hstack

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, recall_score, accuracy_score

from lexicon import lexicon
from readTweetsFromFile import get_tweets


def read_data(filename):
    lines = []
    with open(filename) as infile:
        for line in infile:
            values = line.strip().split("\t")
            if len(values) == 2:
                lines.append(values)
    return lines


def get_features(vectorizer, data, fit=True):
    corpus, targets = zip(*data)
    if fit:
        x = vectorizer.fit_transform(corpus)
    else:
        x = vectorizer.transform(corpus)

    pos = set(lexicon.positive)
    neg = set(lexicon.negative)
    x2 = np.zeros((x.shape[0], 2))
    for i, line in enumerate(corpus):
        words = set(line.split(" "))
        x2[i, 0] = len(words & pos) / len(words)
        x2[i, 1] = len(words & neg) / len(words)

    x = hstack((x, coo_matrix(x2)))

    y = targets
    return x, y


def main():
    #train = read_data("data2/semeval_train_A.txt")
    #test  = read_data("data/semeval_test_A.txt")
    # train, test = test, train

    tweets = [(tweet, sentiment) for id, (tweet, _, sentiment) in get_tweets(2).items()]

    from random import shuffle, seed
    seed(2342341)
    shuffle(tweets)

    split = int(len(tweets)*0.8)
    train, test = tweets[:split], tweets[split:]


    targets = list(zip(*train))[1]
    # Note how un-evenly split the data set is between pos, neut, neg. This is a problem...
    print("Number of training examples:", len(targets))
    print("\tPositive examples:", len(list(t for t in targets if t == "positive")))
    print("\tNeutral examples:", len(list(t for t in targets if t == "neutral")))
    print("\tNegative examples:", len(list(t for t in targets if t == "negative")))

    targets = list(zip(*test))[1]
    # Note how un-evenly split the data set is, this is a problem...
    print("Number of training examples:", len(targets))
    print("\tPositive examples:", len(list(t for t in targets if t == "positive")))
    print("\tNeutral examples:", len(list(t for t in targets if t == "neutral")))
    print("\tNegative examples:", len(list(t for t in targets if t == "negative")))

    clfrs = []

    clf = SVC(decision_function_shape='ovr', kernel='linear', class_weight='balanced')
    clfrs.append(("SVC linear", clf))

    clf = LinearSVC()
    clfrs.append(("LinearSVC", clf))

    clf = LinearSVC(class_weight='balanced')
    clfrs.append(("LinearSVC balanced", clf))

    depths = [30, 40, 50, 60, 70, 80]
    mins = [5, 10, 15, 20]
    from itertools import product
    for depth, min_samples in product(depths, mins):
        clf = RandomForestClassifier(class_weight='balanced', max_depth=depth, min_samples_split=min_samples)
        clfrs.append(("%d:%d" % (depth, min_samples), clf))

    vectorizer = CountVectorizer(ngram_range=(2, 6), analyzer="char")
    x, y = get_features(vectorizer, train)

    for name, clf in clfrs:
        clf.fit(x, y)

        X, Y = get_features(vectorizer, test, fit=False)
        p = clf.predict(X)

        # Semeval report uses macro averaging, orders by recall
        print(name)
        print(recall_score(Y, p, average='macro'), f1_score(Y, p, average='macro'), accuracy_score(Y, p), recall_score(Y, p, average=None, labels=["positive", "neutral", "negative"]))


if __name__ == "__main__":
    main()
