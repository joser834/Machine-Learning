from string import punctuation, digits
import numpy as np
import random
import os

def get_order(n_samples):
    var = 'C:/Users/JR29/Desktop/GitHub/Machine-Learning/Sentiment_Analysis/resources_sentiment_analysis/'
    if os.path.isfile(var + str(n_samples) + '.txt'):
        with open(var + str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    else:
        FileNotFoundError = IOError
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    import numpy as np
    
    theta = current_theta
    theta_0 = current_theta_0
    y_pred = np.dot(theta, feature_vector) + theta_0

    if label*y_pred <= 0:
        theta += label*feature_vector
        theta_0 += label

    return(theta, theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    import numpy as np
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    for j in range(T):
        print("----- Iteration (" + str(j+1) + "/" + str(T) + ") -----")
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_0)
    
    return(theta, theta_0)

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    c = 0

    u = np.zeros(feature_matrix.shape[1])
    b = 0

    for j in range(T):
        for i in get_order(feature_matrix.shape[0]):
            y_pred = np.dot(theta, feature_matrix[i,:]) + theta_0

            if labels[i]*y_pred <= 0:
                theta += labels[i]*feature_matrix[i,:]
                theta_0 += labels[i]
                u += labels[i]*c*feature_matrix[i,:]
                b += labels[i]*c
            c += 1
    theta = theta - u/c
    theta_0 = theta_0 - b/c
    return(theta, theta_0)

# Example for single update -----------------------------------------------------------------------
# f = np.array([1, 2])
# l, t, t0 = 1, np.array([-1, 1]), -1.5

# print(perceptron_single_step_update(f,l,t,t0))

# Example for full perceptron -----------------------------------------------------------------------
f = np.array([
 [ 0.07807965,  0.35486968, -0.09192713, -0.18515078, -0.05963511,  0.31373566,
  -0.26108445, -0.29508217, -0.28867579, -0.19791986],
 [-0.37436567,  0.48205132, -0.05831441,  0.35996561, -0.09275385,  0.03934837,
  -0.0482767 ,  0.22078687,  0.06614133, -0.30852791],
 [-0.32533154, -0.42366336,  0.42998388,  0.30435934,  0.28806414 ,-0.28359362,
  -0.02482462,  0.4723071 ,  0.26101321, -0.01499132],
 [ 0.34938193, -0.1098686 ,  0.18568327, -0.21795992, -0.05951992, -0.12686793,
   0.05183573,  0.32333281,  0.42091467, -0.45750666],
 [ 0.25139458,  0.43330154, -0.31872001, -0.21165255,  0.43737074, -0.49756002,
  -0.23664196, -0.34626735, -0.31029739,  0.41959844]
])
l = [-1, 1, -1, 1, 1]
t = 5

print(average_perceptron(f, l, t))
