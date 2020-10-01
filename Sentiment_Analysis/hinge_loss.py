def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """

    import numpy as np

    y_pred = np.dot(theta, feature_vector) + theta_0
    hinge_loss = max(0, 1-(label*y_pred))

    return(hinge_loss)

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """

    import numpy as np

    hinge_loss = 0
    for i in range(0,feature_matrix.shape[0]):
        y_pred = np.dot(theta, feature_matrix[i,:]) + theta_0
        hinge_loss += max(0, 1-(labels[i]*y_pred))

    res = hinge_loss/feature_matrix.shape[0]
    
    return(res)


import numpy as np

# Example -----------------------------------------------------------------------
f = np.array([[1, 2], [1, 2]])
l, t, t0 = np.array([1, 1]), np.array([-1, 1]), -0.2

print(hinge_loss_full(f, l, t, t0))
