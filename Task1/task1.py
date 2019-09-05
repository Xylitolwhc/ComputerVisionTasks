import numpy as np
import matplotlib.pyplot as plt

rate = 0.01
error = 0.1


def classify(train_data, train_label):
    # initialize weight = 0
    w = np.zeros(len(train_data[0]))
    bias = 0.0
    num = len(train_data)
    while True:
        for i in range(num):
            x = train_data[i]
            y = train_label[i]
            if y * (np.dot(w, x) + bias) <= 0:  # if prediction is wrong
                w = w + rate * y * x.T  # update weight:w = w + r * x_i * y_i
                bias = bias + rate * y  # update b:b = b + r * y_i

        loss = 0
        for i in range(num):
            y = train_label[i]
            output = np.dot(w, train_data[i]) + bias
            if output * y <= 0:
                loss += 1
        if (loss / num) <= error:
            return w, bias

    return w, bias


def __main__():
    size = 10
    mean = (0, 0)
    cov = [[1, 0.75], [0.75, 1]]
    # generate negative samples
    x1 = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    mean = (2, 4)
    # generate positive samples
    x2 = np.random.multivariate_normal(mean=mean, cov=cov, size=size)
    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((-np.ones(size), np.ones(size)))

    w, bias = classify(X, Y)
    print("w:", w, "bias:", bias)

    for d, sample in enumerate(X):
        # Plot the negative samples
        if d < size:
            plt.scatter(sample[0], sample[1], s=120, marker='_')
        # Plot the positive samples
        else:
            plt.scatter(sample[0], sample[1], s=120, marker='+')
    plt.plot([-2, 5], [w[0] / w[1] * 2 - bias / w[1], w[0] / w[1] * -5 - bias / w[1]])
    plt.show()
    pass


if __name__ == '__main__':
    __main__()
