import numpy as np
from nltk.tokenize import TweetTokenizer


def tokenize(data):
    tkzr = TweetTokenizer(reduce_len=True)
    corpus = []
    for line in data:
        corpus.append(tkzr.tokenize(line))
    return corpus


def more_features(data):
    corpus = tokenize(data)
    features = []

    # Uppercase
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token == token.upper()]))
    features.append(uppercase)

    # !!!
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token == "!"]))
    features.append(uppercase)

    # ...
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token == "..."]))
    features.append(uppercase)

    # !!!
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token == "?"]))
    features.append(uppercase)

    # hashtags
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token.startswith("#")]))
    features.append(uppercase)

    # mentions
    uppercase = []
    for tweet in corpus:
        uppercase.append(len([token for token in tweet if token.startswith("@")]))
    features.append(uppercase)

    feature_matrix = np.matrix(features).T
    return feature_matrix
