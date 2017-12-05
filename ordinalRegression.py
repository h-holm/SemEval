import numpy as np

from random import seed, shuffle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, metrics
from mord import OrdinalRidge, LogisticAT, LogisticIT, LogisticSE

from readTweetsFromFile import get_tweets
from svm import get_bow, get_ngrams


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
        for name, clf in clfrs:
            clf.fit(x, y)
            p = clf.predict(X)
            if "Regr." in name:
                p = [round(c) for c in p]
            score = metrics.mean_absolute_error(p, Y)
            from collections import defaultdict
            mscores = defaultdict(list)
            for cls, pred in zip(Y, p):
                mscores[cls].append(cls-pred)
            macroscore = sum([abs(sum(v))/len(v) for k,v in mscores.items()])/len(classes)

            print(name, ff_name, score)
            results.append((
                name,
                ff_name,
                score,
                macroscore
            ))

    results = sorted(results, key=lambda x: x[3])  # Order by recall
    width = (25+10+10+10)
    print("=" * width)
    print("%-25s%10s%10s%10s" % ("Classifier", "Features", "MAE", "MAE"))
    print("-" * width)
    for result in results:
        print("%-25s%10s%10.3f%10.3f" % result)
    print("=" * width)


if __name__ == "__main__":
    main()
