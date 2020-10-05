import numpy as np
import math

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    theta = current_theta
    theta_0 = current_theta_0
    y_pred = np.dot(theta, feature_vector) + theta_0

    if label*y_pred <= 1:
        theta = ((1-(eta*L))*theta) + (eta*label*feature_vector)
        theta_0 += eta*label
    else:
        theta = (1-(eta*L))*theta
        theta_0 = theta_0
    
    return(theta, theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    counter = 0
    
    for t in range(T):
        for i in range(feature_matrix.shape[0]):
            counter += 1
            eta = 1/math.sqrt(counter)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i,:], labels[i], L, eta, theta, theta_0)
    
    return(theta, theta_0)



# Example for Pegasus single update -----------------------------------------------------------------------
f = np.array([0.39070867, -0.05240042, 0.10690338, -0.19161465, 0.28920972, 0.00650771, 0.15822319, 0.38478464, 0.39611079, -0.26242874])
l = 1
lam = 0.8614432452732785
e = 0.3627833399252254
t = np.array([0.22251002, -0.49497394, 0.25019883, -0.49806464, -0.46723159, -0.00589721, 0.22167011, 0.45498801, -0.32837568, 0.49495199])
t0 = 1.0632151862260786

print(pegasos_single_step_update(f, l, lam, e, t, t0))