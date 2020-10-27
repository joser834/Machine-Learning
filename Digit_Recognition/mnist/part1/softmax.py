import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """

    for i in range(X.shape[0]):
        c = np.max(np.dot(theta, X[i].T)/temp_parameter)
        A = 1/np.sum(np.exp((np.dot(theta, X[i].T)/temp_parameter)-c))
        h = (A * (np.exp((np.dot(theta, X[i].T)/temp_parameter)-c)))
        if i == 0:
            H = h
        else:
            H = np.column_stack((H,h))
    return(H)

# n, d, k = 3, 5, 7
# x = np.arange(0, n * d).reshape(n, d)
# zeros = np.zeros((k, d))
# temp = 0.2
# theta = np.arange(0, k * d).reshape(k, d)
# print(compute_probabilities(x, theta, temp))

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    A = 0
    for i in range(X.shape[0]):
        probs = compute_probabilities(X, theta, temp_parameter)[Y[i], i]
        A += np.log(probs)
    
    c = (-1/X.shape[0])*A + (lambda_factor/2)*np.sum(theta)
    return(c)

# x = np.array([[ 1., 88., 58., 86., 90., 35., 64.,  7., 89.,  7., 32.],
#  [ 1., 62., 83., 66., 47., 45., 72., 32., 42., 21., 85.],
#  [ 1., 22., 32., 13., 18., 46., 73., 51.,  3., 38., 34.],
#  [ 1., 77., 52., 50., 18., 54., 44., 12., 92., 66., 57.],
#  [ 1., 94., 82., 48., 16.,  6., 20., 89., 12., 73., 57.],
#  [ 1., 30., 27., 61., 88.,  8., 64., 17., 75., 70., 56.],
#  [ 1., 41., 98., 10., 77., 50., 85., 67., 18., 11.,  4.],
#  [ 1., 91., 59., 66., 31., 66., 55., 28., 26., 44., 84.],
#  [ 1., 43.,  3., 70., 97., 29., 24., 10., 54., 31., 84.],
#  [ 1., 44., 59., 35., 40., 48., 28., 17., 99., 94., 56.]])
# theta = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# temp = 1.0
# lf = 0.0001
# y=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# print(compute_cost_function(x, y, theta, lf, temp))

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #n, k, d = X.shape[0], theta.shape[0], X.shape[1]
    n, k, d = X.shape[0], theta.shape[0], X.shape[1]
    A = np.zeros((k,d))
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    
    for i in range(X.shape[0]):
        probs = compute_probabilities(X[i].reshape(1,len(X[i])), theta, temp_parameter)
        A += np.dot((M[i]-probs).reshape(-1,1), X[i].reshape(1, -1))
    
    cost = (-1/(temp_parameter*n))*A + lambda_factor*theta
    theta += theta - alpha*(cost)
    return(theta)

x = np.array([[ 1., 66., 10.,  6., 17., 66., 64., 82., 54.,  8., 63.],
 [ 1., 28., 41., 18., 10., 71., 84., 65.,  4., 17., 11.],
 [ 1., 56., 22., 59., 70., 75., 10., 52., 40., 40., 56.],
 [ 1., 81., 60., 72., 61., 99., 63.,  8., 32., 43., 48.],
 [ 1., 53., 42., 27., 34., 69.,  1., 10., 52., 35.,  8.],
 [ 1., 17., 34., 26., 84., 78., 53., 47., 92., 80., 82.],
 [ 1.,  5., 20.,  7., 60., 46.,  1., 97., 17., 49., 82.],
 [ 1., 60., 78., 94., 50., 68., 86., 22., 94., 86., 52.],
 [ 1., 17., 55.,  8., 12., 77., 87., 98., 52., 46., 10.],
 [ 1.,  7., 71.,  9., 20., 66., 16.,  2.,  6., 92., 62.]])
theta = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
temp = 1.0
lf = 0.0001
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
print(run_gradient_descent_iteration(x,y,theta,0.01,lf,temp))

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
