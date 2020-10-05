import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    label = []
    for i in range(feature_matrix.shape[0]):
        y_pred = np.dot(theta, feature_matrix[i,:]) + theta_0
        if y_pred > 0:
            label.append(1)
        else:
            label.append(-1)
    return(np.array(label))

def accuracy(preds, targets):
    """
	Given length-N vectors containing predicted and target labels,
	returns the percentage and number of correct predictions.
	"""
    return (preds == targets).mean()

def classifier_accuracy(classifier, train_feature_matrix, val_feature_matrix, train_labels, val_labels, **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    y_pred_train = classify(train_feature_matrix, theta, theta_0)
    y_pred_val = classify(val_feature_matrix, theta, theta_0)
    
    acc_train = accuracy(y_pred_train, train_labels)
    acc_val = accuracy(y_pred_val, val_labels)

    return(acc_train, acc_val)


classifier_accuracy(1,1,1,1,1,theta = np.array([0,1]), theta_0 = 1)




# Example for linear classifier -----------------------------------------------------------------------

f = np.array([
    [1,0],
    [1,2],
    [0,-1],
    [-2,-2]
])
t = np.array([1,-1])
t0 = -1

print(classify(f, t, t0))

