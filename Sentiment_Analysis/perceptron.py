from string import punctuation, digits
import numpy as np
import random

def get_order(n_samples):
    try:
        with open('C:/Users/JR29/Desktop/GitHub/Machine-Learning/Sentiment_Analysis/resources_sentiment_analysis/' + str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except NameError:
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
    import numpy as np
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0

    theta_arr = []
    theta_0_arr = []

    for j in range(T):
        print("----- Iteration (" + str(j+1) + "/" + str(T) + ") -----")
        for i in get_order(feature_matrix.shape[0]):
            # Single update perceptron
            y_pred = np.dot(theta, feature_matrix[i,:]) + theta_0

            if labels[i]*y_pred <= 0:
                theta += labels[i]*feature_matrix[i,:]
                theta_0 += labels[i]
            
            theta_arr.append(theta)
            theta_0_arr.append(theta_0)
    return(np.array(theta_arr).mean(axis = 0), np.mean(theta_0_arr))

# Example for single update -----------------------------------------------------------------------
# f = np.array([1, 2])
# l, t, t0 = 1, np.array([-1, 1]), -1.5

# print(perceptron_single_step_update(f,l,t,t0))

# Example for full perceptron -----------------------------------------------------------------------
f = np.array([
    [-0.13714734, -0.1014197, -0.34854003, 0.27693084, 0.45009371, -0.24048786, 0.2279144, 0.14703612, -0.11220196, -0.00366032],
    [-0.10397916, -0.3866046, -0.18837448, -0.28206775, -0.34864876, -0.28787942, 0.45339883, -0.08607866, -0.24583239, -0.39027705],
    [-0.20057219, -0.36504427, 0.1316528 , 0.07279378, -0.08926369, 0.29200623, 0.24144614, -0.22884339, -0.3269274 , -0.34802841],
    [-0.38678468, 0.46717285, -0.09754879, -0.23747704, -0.43218194, -0.21709596, 0.27751231, 0.14997779, 0.49223438, -0.03573181],
    [0.02947019, 0.44506145, -0.21756791, 0.25508455, 0.05569321, -0.43702505, 0.20148895, 0.17175176, 0.47210888, 0.22201971]
])
l = [-1, 1, -1, -1, 1]
t = 5

print(average_perceptron(f, l, t))
