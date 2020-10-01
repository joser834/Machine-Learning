import numpy as np

def perceptron_origin(epochs, x_train, y_train):
    theta = np.array([0,0,0])
    progression = []
    progression.append(theta)
    mistakes = 0
    
    for i in range(0,epochs):
        print("----- Iteration (" + str(i+1) + "/" + str(epochs) + ") -----")
        for j in range(0,x_train.shape[0]):
            y_pred = np.dot(theta, x_train[j])
            print("Original y is " + str(y_train[j]) + " and Predicted is " + str(y_pred))

            if y_train[j]*y_pred <= 0: #and i != 0 and j!=0:
                print("MISTAKE!")
                mistakes += 1
                theta = theta + y_train[j]*x_train[j]
                progression.append(theta)

    return(theta, mistakes, progression)

def perceptron_with_offset(epochs, x_train, y_train, theta, theta_not):
    theta = theta
    theta_not = theta_not
    mistakes = 0
    
    for i in range(0,epochs):
        print("----- Iteration (" + str(i+1) + "/" + str(epochs) + ") -----")
        for j in range(0,x_train.shape[0]):
            y_pred = np.dot(theta, x_train[j]) + theta_not
            print("Original y is " + str(y_train[j]) + " and Predicted is " + str(y_pred))

            if y_train[j]*y_pred <= 0: #and i != 0 and j!=0:
                print("MISTAKE!")
                mistakes += 1
                theta += y_train[j]*x_train[j]
                theta_not += y_train[j]

    return(theta, theta_not, mistakes)

#Example using Perceptron from Origin ------------------------------------------------------
train_data_x = np.array(
    [
        [0,0,1],
        [-1,0,0],
        [0,1,0]
    ]
)
train_data_y = np.array([1,1,1])
t, m, p = perceptron_origin(3, train_data_x, train_data_y)
print(t)
print(m)
print(p)

#Example using Perceptron with Offset -----------------------------------------------------------------------
x_raw = np.array(
    [
        [-4,2],
        [-2,1],
        [-1,-1],
        [2,2],
        [1,-2]
    ]
)
y_raw = [1,1,-1,-1,-1]

theta_init = np.array([-3,3])
theta_not_init = -3

#t, t0, m = perceptron_with_offset(4, x_raw, y_raw, theta_init, theta_not_init)

#print(t)
#print(t0)