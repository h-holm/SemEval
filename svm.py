import numpy as np
import sys

from scipy.sparse import coo_matrix, hstack

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, recall_score, accuracy_score

from lexicon import lexicon
from readTweetsFromFile import get_tweets

from collections import defaultdict
from random import shuffle, seed


def get_features(vectorizer, data, fit=True):
    corpus, targets = zip(*data)
    if fit:
        x = vectorizer.fit_transform(corpus)
    else:
        x = vectorizer.transform(corpus)

    # This actually does not make a lot of difference
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
    """
    Use: python3 svm.py <set_number>
    set_number: 1, 2 or 3

    """
    set_id = int(sys.argv[1])
    tweets = [(fields[0], fields[-1]) for id, fields in get_tweets(set_id).items()]

    seed(4)
    shuffle(tweets)

    split = int(len(tweets)*0.8)
    # This should do even splits based on classes
    train, test = tweets[:split], tweets[split:]
    classes = set()

    for name, data in [("training", train), ("testing", test)]:
        targets = list(zip(*data))[1]
        # Note how un-evenly split the data set is between pos, neut, neg. This is a problem...
        print("Number of %s examples:" % name, len(targets))
        clc = defaultdict(list)
        for target in targets:
            classes.add(target)
            clc[target].append(target)
        for target, lst in clc.items():
            print("\t", target, len(lst))

    clfrs = []

    clf = SVC(decision_function_shape='ovr', kernel='linear', class_weight='balanced')
    clfrs.append(("SVC linear, balanced", clf))

    clf = LinearSVC()
    clfrs.append(("LinearSVC", clf))

    clf = LinearSVC(class_weight='balanced')
    clfrs.append(("LinearSVC balanced", clf))

    depths = [30, 40, 50, 60, 70, 80]
    mins = [5, 10, 15, 20]
    from itertools import product
    for depth, min_samples in product(depths, mins):
        clf = RandomForestClassifier(class_weight='balanced', max_depth=depth, min_samples_split=min_samples)
        clfrs.append(("RndForest %d:%d" % (depth, min_samples), clf))

    vectorizer = CountVectorizer(ngram_range=(2, 6), analyzer="char")
    x, y = get_features(vectorizer, train)

    print("Start training and testing")
    results = []
    for name, clf in clfrs:
        print("Training: %s" % name)
        clf.fit(x, y)

        X, Y = get_features(vectorizer, test, fit=False)
        p = clf.predict(X)

        # Semeval report uses macro averaging, orders by recall
        results.append((
            name,
            recall_score(Y, p, average='macro'),
            f1_score(Y, p, average='macro'),
            accuracy_score(Y, p),
            ", ".join("%2.2f" % score for score in recall_score(Y, p, average=None, labels=list(classes)))
        ))

    results = sorted(results, key=lambda x: x[1], reverse=True)  # Order by recall
    width = (25+10+10+10+30)
    print("=" * width)
    print("%-25s%10s%10s%10s%30s" % ("Classifier", "Recall", "F1", "Accuracy", "Class recall (%s)" % str(list(classes))))
    print("-" * width)
    for result in results:
        print("%-25s%10.2f%10.2f%10.2f%30s" % result)
    print("=" * width)


if __name__ == "__main__":
    main()
