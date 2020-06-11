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

# calculate learning rate
def calcLearningRateWithArmijoCondition(x, d):
    alpha = 1.0
    rho = 0.8
    c = 1e-4
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(df(x).T, d):
        alpha *= rho
    return alpha

def main():
    # calculate the true result of the funtion
    ground_truth = np.dot(np.linalg.inv(Q), b)
    print('groud truth is: ', ground_truth)
    print('true min value is:', f(ground_truth))

    # set initial point
    x = np.array([[10.0, 10.0, 10.0]]).T
    threshold = 1e-8
    iteration_num = 0

    gradient = df(x)
    d = -gradient

    # start iteration
    while np.linalg.norm(gradient) > threshold:
        # calculate learning_rate
        alpha = calcLearningRateWithArmijoCondition(x, d)
        # alpha = -np.dot(gradient.T, d) / np.dot(np.dot(d.T, Q), d)
        # print("alpha2", alpha)
        # update point
        x = x + alpha * d
        # update gradient
        old_gradient = gradient
        gradient = df(x)
        # calculate descent direction param
        beta = np.dot(gradient.T, gradient) / np.dot(old_gradient.T, old_gradient)
        # update descent direction
        d = -gradient + beta * d
        iteration_num += 1
    print('result is: ',x)
    print('calculated min value is: ', f(x))
    print('iteration number is: ', iteration_num)

if __name__ == "__main__":
    main()