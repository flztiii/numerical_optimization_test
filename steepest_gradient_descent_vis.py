# /usr/bin/python3
# -*- coding: utf-8 -*-

'''

Steepest Gradient Descent to Solve Linear Function Example
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
    return 0.5 * np.dot(np.dot(x.T, Q), x) - np.dot(b.T, x)

# first derivative of the objective function
def df(x):
    return np.dot(Q, x) - b

def main():
    # calculate the true result of the funtion
    ground_truth = np.dot(np.linalg.inv(Q), b)
    print('groud truth is: ', ground_truth)
    print('true min value is:', f(ground_truth))

    # visualization prepare
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111, projection='3d')
    
    # set initial point
    x = np.array([10.0, 10.0]).T
    threshold = 1e-10
    iteration_num = 0
    recorder = [x]

    gradient = df(x)
    # start iteration
    while np.linalg.norm(gradient) > threshold:
        # calculate the learning rate
        learning_rate = np.dot(gradient.T, gradient) / np.dot(np.dot(gradient.T, Q), gradient)
        # update point
        x = x - learning_rate * gradient
        # update gradient
        gradient = df(x)
        # update iteration number
        iteration_num += 1
        # update recorder
        recorder.append(x)

        # visualization
        if iteration_num%2 == 0:
            plt.cla()
            # visualize surface
            X = np.arange(-15, 15, 0.1)
            Y = np.arange(-15, 15, 0.1)
            X, Y = np.meshgrid(X,Y)
            Z = np.zeros_like(X)
            for i in range(0, X.shape[0]):
                for j in range(0, X.shape[1]):
                    Z[i, j] = f(np.array([X[i, j], Y[i, j]]).T)
            ax.plot_surface(X, Y, Z)
            # visualize gradient descent
            recorder_x, recorder_y, recorder_z = [], [], []
            for i in range(0, len(recorder)):
                recorder_x.append(recorder[i][0])
                recorder_y.append(recorder[i][1])
                recorder_z.append(f(np.array(recorder[i]).T))
            # print(recorder_x, recorder_y, recorder_z)
            ax.plot3D(recorder_x, recorder_y, recorder_z, color='r', marker= '.')
            plt.pause(0.1)
    print('result is: ',x)
    print('calculated min value is: ', f(x))
    print('iteration number is: ', iteration_num)

    plt.cla()
    # visualize surface
    X = np.arange(-15, 15, 0.1)
    Y = np.arange(-15, 15, 0.1)
    X, Y = np.meshgrid(X,Y)
    Z = np.zeros_like(X)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]).T)
    ax.plot_surface(X, Y, Z)
    # visualize gradient descent
    recorder_x, recorder_y, recorder_z = [], [], []
    for i in range(0, len(recorder)):
        recorder_x.append(recorder[i][0])
        recorder_y.append(recorder[i][1])
        recorder_z.append(f(np.array(recorder[i]).T))
    ax.plot3D(recorder_x, recorder_y, recorder_z, color='r', marker= '.')
    ax.scatter([recorder_x[-1]], [recorder_y[-1]], [recorder_y[-1]], marker='*')
    plt.show()

if __name__ == "__main__":
    main()