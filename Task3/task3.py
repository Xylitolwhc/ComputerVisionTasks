import sys
import mnist
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.neural_network
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

num_samples = 400
reg_lambda = 0.0001
epsilon = 0.001
accuracy = 0.975
hidden_layer_neural = 10

sys.path.append('mnist')
def init_minist():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    n_train, w, h = train_images.shape
    X_train = train_images.reshape((n_train, w * h))
    Y_train = train_labels

    n_test, w, h = test_images.shape
    X_test = test_images.reshape((n_test, w * h))
    Y_test = test_labels

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    return X_train, Y_train, X_test, Y_test


def sklearn_train(model):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)

    # max_features is an important parameter. You should adjust it.
    vectorizer = TfidfVectorizer(max_features=1000)

    X = vectorizer.fit_transform(newsgroups_train.data + newsgroups_test.data)
    X_train = X[0:num_train, :]
    X_test = X[num_train:num_train + num_test, :]

    Y_train = newsgroups_train.target
    Y_test = newsgroups_test.target

    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    model.fit(X_train, Y_train)

    Y_predict = model.predict(X_test)

    print(Y_test)
    print(Y_predict)

    ncorrect = 0
    for dy in (Y_test - Y_predict):
        if 0 == dy:
            ncorrect += 1

    print('text classification accuracy is {}%'.format(round(100.0 * ncorrect / len(Y_test))))


def __main__():
    X_train, Y_train, X_test, Y_test = init_minist()
    activations = ["tanh", "relu", "logistic"]
    for activation in activations:
        clf = sklearn.neural_network.MLPClassifier(activation=activation, learning_rate="constant",
                                                   learning_rate_init=0.001, max_iter=1000,
                                                   hidden_layer_sizes=(20, 20, 20), verbose=False,
                                                   early_stopping=True)

        clf.fit(X_train, Y_train)
        print(activation, clf.score(X_test, Y_test))


if __name__ == '__main__':
    __main__()
