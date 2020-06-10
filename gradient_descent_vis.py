#! /usr/bin/python3
#! -*- coding: utf-8 -*-

'''

Gradient Descent to Solve Linear Function Example

author: flztiii

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Q = np.array([[6,8],[1, 4]])
b = np.array([5.2, 1.3])

# the objective funtion
def f(x):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) - np.dot(b.T, x)

# first derivative of the objective function
def df(x):
    return np.dot(Q, x) - b

def main():
    # calculate the true result of the funtion
    ground_truth = np.dot(np.linalg.inv(Q), b)
    print('groud truth is: ', ground_truth)
    print('true min value is:', f(ground_truth))

    # visualization prepare
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # set initial point
    x = np.array([0.0, 0.0]).T
    learning_rate = 0.01
    threshold = 1e-10
    iteration_num = 0
    
    gradient = df(x)
    # start iteration
    while np.linalg.norm(gradient) > threshold:
        # update new point
        x = x - learning_rate * gradient
        # update gradient
        gradient = df(x)
        # update num
        iteration_num += 1

        # visualization surface
        if iteration_num%10 == 0:
            X = np.arange(-2, 2, 0.1)
            Y = np.arange(-2, 2, 0.1)
            X, Y = np.meshgrid(X,Y)
            Z = np.zeros_like(X)
            for i in range(0, X.shape[0]):
                for j in range(0, X.shape[1]):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]).T)
            ax.plot_surface(X, Y, Z)
            plt.pause(0.01)

    print('result is: ',x)
    print('calculated min value is: ', f(x))
    print('iteration number is: ', iteration_num)


if __name__ == "__main__":
    main()