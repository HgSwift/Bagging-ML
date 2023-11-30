import numpy as np
import matplotlib.pyplot as plt
import math
from Bagging_Utils import visualization, get_dataset_fixed


class Stump():
    def __init__(self, data, labels, weights):
        '''
        Initializes a stump (one-level decision tree) which minimizes
        a weighted error function of the input dataset.

        In this function, you will need to learn a stump using the weighted
        datapoints. Each datapoint has 2 features, whose values are bounded in
        [-1.0, 1.0]. Each datapoint has a label in {+1, -1}, and its importance
        is weighted by a positive value.

        The stump will choose one of the features, and pick the best threshold
        in that dimension, so that the weighted error is minimized.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
            labels: An ndarray with shape (n, ). Values are +1 or -1.
            weights: An ndarray with shape (n, ). The weights of each
                datapoint, all positive.
        '''
        # You may choose to use the following variables as a start

        n = data.shape[0]
        # The feature dimension which the stump will decide on
        # Either 0 or 1, since the datapoints are 2D
        self.dimension = 0

        # The threshold in that dimension
        # May be midpoints between datapoints or the boundaries -1.0, 1.0
        self.threshold = -1.0

        # The predicted sign when the datapoint's feature in that dimension
        # is greater than the threshold
        # Either +1 or -1
        self.sign = 1
        self.min_error = float('inf')
        # greedy search to find best threshold and dimension
        for dim in range(2):
            Col = data[:, dim]
            X = np.sort(Col)
            candidates = []
            candidates.append(min(X))
            candidates.append(max(X))
            for i in range(X.shape[0] - 1):
                candidates.append(0.5*(X[i]+ X[i+1]))
            thresholds = np.unique(candidates)
            for threshold in thresholds:
                # predict with sign 1
                sign = 1
                predictions = np.ones(n)
                predictions[Col < threshold] = -1
                #print(predictions)
                # Error = sum of weights of misclassified samples
                misclassified = weights[labels != predictions]
                error = sum(misclassified)
                #print(error)
                if error > (0.5*sum(weights)):
                    error = 1 - error
                    sign = -1
                    # store the best configuration
                if error < self.min_error:
                    self.sign = sign
                    self.threshold = threshold
                    self.dimension = dim
                    self.min_error = error
        #print(self.dimension)
        #print(self.threshold)
        #print(self.sign)
        print(self.min_error)
        pass

    def predict(self, data):
        '''
        Predicts labels of given datapoints.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].

        Returns:
            prediction: An ndarray with shape (n, ). Values are +1 or -1.
        '''
        n = data.shape[0]
        X = data[:, self.dimension]
        predictions = np.ones(n)
        if self.sign == 1:
            predictions[X < self.threshold] = -1
        else:
            predictions[X > self.threshold] = -1

        return predictions
        pass


def bagging(data, labels, n_classifiers, n_samples, seed=0):
    '''
    Runs Bagging algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.
        n_samples: Number of samples to train each classifier.
        seed: Random seed for NumPy.

    Returns:
        classifiers: A list of classifiers.
    '''
    classifiers = []
    n = data.shape[0]

    for i in range(n_classifiers):
        np.random.seed(seed + i)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)
        X = data[sample_indices]
        Y = labels[sample_indices]
        weights = np.ones(n)
        #print(sample_indices)
        classifier = Stump(X, Y, weights[sample_indices])
        classifiers.append(classifier)
        pass

    return classifiers


def adaboost(data, labels, n_classifiers):
    '''
    Runs AdaBoost algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.

    Returns:
        classifiers: A list of classifiers.
        weights: A list of weights assigned to the classifiers.
    '''
    classifiers = []
    weights = []
    n = data.shape[0]
    data_weights = np.ones(n) / n

    for i in range(n_classifiers):
        classifier = Stump(data, labels, data_weights)
        predictions = classifier.predict(data)
        error = sum(data_weights[labels != predictions]) + 1e-10
        #print(f"Error in Class: {classifier.min_error}, Error in ada: {error}")
        alpha = 0.5 * np.log((1.0 - error) / (error))
        # calculate predictions and update weights
        for j in range(n):
            if predictions[j] != labels[j]:
                ers = "wrong"
            else:
                ers = "right"
            #print(f"{j}, weight: {data_weights[j]}, class: {ers}")
        data_weights *= np.exp(-alpha * labels * predictions)
        Z_t = np.sum(data_weights)
        c_t = 0
        for j in range(n):
            c_t += (data_weights[j]*labels[j]*predictions[j])
        alpha_a = np.log(math.sqrt((1-c_t)/c_t))
        alpha_b = np.log(math.sqrt((1+c_t)/(1-c_t)))
        alpha_c = np.log((1-c_t)/c_t)
        alpha_d = np.log((1+c_t)/(1-c_t))
        data_weights /= Z_t
        #data_weights /= np.sum(data_weights)
        classifiers.append(classifier)
        #print(f"Alpha: {alpha}, error: {error}, Z_t: {Z_t}, SumWeight: {sum(data_weights)}")
        print(f"Alpha: {alpha}, alpha_a: {alpha_a}, alpha_b: {alpha_b}, alpha_c: {alpha_c}, alpha_d: {alpha_d}, ")
        weights.append(alpha)
        pass

    return classifiers, weights


if __name__ == '__main__':
    data, labels = get_dataset_fixed()
    weights = np.ones(data.shape[0])/data.shape[0]
    #print(weights)
    # You can play with the dataset and your algorithms here
    #classifier = [Stump(data, labels, weights)]
    classifiers, weights = adaboost(data, labels, 20)
    #classyfiers = bagging(data, labels, 20, 15)
    visualization(data, labels, classifiers, weights)
    #visualization(data, labels, classyfiers, weights)