import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.neural_network
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

num_samples = 200
reg_lambda = 0.01
epsilon = 0.001
error = 0.99


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(X, y, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def train(model, X, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    while True:
        # forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct = 0
        for i, prob in enumerate(probs):
            if prob[0] > prob[1]:
                if y[i] == 0:
                    correct += 1
            elif y[i] == 1:
                correct += 1
        print(correct / num_samples)
        if correct / num_samples >= error:
            break

        # Backpropagation
        delta3 = probs
        delta3[range(num_samples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # ADD regularization
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # print(dW1, db1, dW2, db2)
        model['W1'], model['b1'], model['W2'], model['b2'] = W1, b1, W2, b2

    return model


def __main__():
    X, y = sklearn.datasets.make_moons(num_samples, noise=0.10)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    model = dict()
    model['W1'] = np.random.rand(2, 4)
    model['W2'] = np.random.rand(4, 2)
    model['b1'] = np.random.rand(1, 4)
    model['b2'] = np.random.rand(1, 2)
    model = train(model, X, y)
    plot_decision_boundary(X, y, lambda x: predict(model, x))
    plt.show()


def sklearn_train(model):
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    num_train = len(newsgroups_train.data)
    num_test = len(newsgroups_test.data)

    # max_features is an important parameter. You should adjust it.
    vectorizer = TfidfVectorizer(max_features=40)

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


def main2():
    clf = sklearn.linear_model.LogisticRegressionCV(max_iter=1000, multi_class="auto", cv=5)
    sklearn_train(clf)
    clf = sklearn.neural_network.MLPClassifier(activation="tanh", max_iter=1000, hidden_layer_sizes=(50, 50),
                                               early_stopping=True)
    sklearn_train(clf)
    pass


if __name__ == '__main__':
    # __main__()
    main2()
