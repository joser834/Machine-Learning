import numpy as np

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    A = np.dot(X.T,X)
    B = lambda_factor*np.eye(X.shape[1])
    C = np.dot(X.T,Y)
    theta = np.dot(np.linalg.inv(A+B),C)
    return(theta)

# Example -----------------------------------------------------------------------
# x = np.array([
#     [0.22710231, 0.22990637],
#     [0.41483016, 0.66602953],
#     [0.36395381, 0.08855222],
#     [0.10284558, 0.63802993],
#     [0.60193562, 0.50336436],
#     [0.4398084 , 0.86708208],
#     [0.15649144, 0.16755288],
#     [0.32097052, 0.37398561],
#     [0.39521456, 0.67778334],
#     [0.41899049, 0.8217538 ],
#     [0.08517112, 0.98952968],
#     [0.5689853 , 0.51093225],
#     [0.58355097, 0.61763799],
#     [0.29104132, 0.09186201],
#     [0.59548743, 0.32567937],
#     [0.16733378, 0.64308253],
#     [0.4626983 , 0.16623097],
#     [0.20936176, 0.57699717],
#     [0.75323588, 0.42076979]
# ])

# y = [0.6180502, 0.11414268, 0.16354983, 0.45525746, 0.12089474, 0.78347523, 0.90503838 ,0.72594625, 0.22589511, 0.08951521, 0.69720362, 0.60441882,
# 0.68373848 ,0.86066428, 0.9487789 , 0.99579397, 0.45937451, 0.73058468, 0.71347492]

# lambda_f = 0.23314019247300644

# print(closed_form(x, y, lambda_f))

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
