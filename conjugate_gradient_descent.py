#! /usr/bin/python3
#! -*- coding: utf-8 -*-

'''

Gradient Descent to Solve Linear Function Example

author: flztiii

'''

import numpy as np

Q = np.array([[6, 3, 5],[3, 2, 2],[5, 2, 7]])
b = np.array([[5.2, 1.3, 4.4]]).T

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

    # set initial point
    x = np.array([[10.0, 10.0, 10.0]]).T
    threshold = 1e-10
    n = 3

    gradient = df(x)
    d = -gradient

    # start iteration
    for iteration_num in range(0, n):
        # calculate learning_rate
        alpha = -np.dot(gradient.T, d) / np.dot(np.dot(d.T, Q), d)
        # update point
        x = x + alpha * d
        # update gradient
        gradient = df(x)
        # calculate descent direction param
        beta = np.dot(np.dot(gradient.T, Q), d) / np.dot(np.dot(d.T, Q), d)
        # update descent direction
        d = -gradient + beta * d
    print('result is: ',x)
    print('calculated min value is: ', f(x))
    print('iteration number is: ', n)

if __name__ == "__main__":
    main()