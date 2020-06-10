#! /usr/bin/python3
#! -*- coding: utf-8 -*-

'''

Gradient Descent to Solve Quadratic Function Example

author: flztiii

'''

import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[6,8,1],[1, 4, 5],[7, 1, 3]])
b = np.array([5.2, 1.3, 4.4])

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
    
    # set initial point
    x = np.array([0.0, 0.0, 0.0]).T
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
    print('result is: ',x)
    print('calculated min value is: ', f(x))
    print('iteration number is: ', iteration_num)


if __name__ == "__main__":
    main()