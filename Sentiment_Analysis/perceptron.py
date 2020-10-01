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

# Example -----------------------------------------------------------------------
import numpy as np
f = np.array([1, 2])
l, t, t0 = 1, np.array([-1, 1]), -1.5

print(perceptron_single_step_update(f,l,t,t0))
