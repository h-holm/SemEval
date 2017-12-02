import numpy as np

from random import seed, shuffle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, metrics
from mord import OrdinalRidge, LogisticAT, LogisticIT, LogisticSE

from readTweetsFromFile import get_tweets
from svm import get_features


def main():
    """
    Use: python3 ordinalRegression.py

    """
    set_id = 3
    tweets = [(fields[0], fields[-1]+2) for id, fields in get_tweets(set_id).items()]
    # +2 since we have to start at zero for Logistical classifiers
    # it does not change anything really, the scale is just from 0 to 4 instead of -2 to 2

    seed(4)
    shuffle(tweets)

    split = int(len(tweets)*0.8)
    # This should do even splits based on classes
    train, test = tweets[:split], tweets[split:]

    vectorizer = CountVectorizer(ngram_range=(2, 6), analyzer="char")
    x, y = get_features(vectorizer, train)
    X, Y = get_features(vectorizer, test, fit=False)

    classes = np.unique(y)

    for name, data in [("training", y), ("testing", Y)]:
        # Note how un-evenly split the data set is between pos, neut, neg. This is a problem...
        print("Number of %s examples:" % name, len(data))
        for cl in classes:
            print("\t", cl, np.sum(data == cl))

    clfrs = []

    clf = OrdinalRidge()
    clfrs.append(("OrdinalRidge", clf))

    clf = LogisticAT(alpha=1.)
    clfrs.append(("LogisticAT", clf))

    clf = LogisticIT(alpha=1.)
    clfrs.append(("LogisticIT", clf))

    clf = LogisticSE(alpha=1.)
    clfrs.append(("LogisticSE", clf))

    clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
    clfrs.append(("Logistic Regr. lbfgs", clf))

    clf = linear_model.LogisticRegression(solver='newton-cg', multi_class='multinomial')
    clfrs.append(("Logistic Regr. newton-cg", clf))

    clf = linear_model.LogisticRegression(solver='sag', multi_class='multinomial')
    clfrs.append(("Logistic Regr. sag", clf))

    clf = linear_model.LogisticRegression(solver='saga', multi_class='multinomial')
    clfrs.append(("Logistic Regr. saga", clf))

    clf = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', class_weight='balanced')
    clfrs.append(("Logistic Regr. balanced", clf))

    print("Start training and testing")
    results = []
    for name, clf in clfrs:
        print("Training: %s" % name)
        clf.fit(x, y)
        p = clf.predict(X)
        score = metrics.mean_absolute_error(p, Y)
        results.append((
            name,
            score
        ))

    results = sorted(results, key=lambda x: x[1])  # Order by recall
    width = (25+10)
    print("=" * width)
    print("%-25s%10s" % ("Classifier", "MAE"))
    print("-" * width)
    for result in results:
        print("%-25s%10.2f" % result)
    print("=" * width)


if __name__ == "__main__":
    main()
