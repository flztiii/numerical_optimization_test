#coding=utf-8
# @Author: yangenneng
# @Time: 2018-05-11 22:56
# @Abstract：Conjugate Gradient  Version
import numpy as np
from numpy import *

def CG_Preliminary(A,b,x_k):
    k = 0
    r_k  = A * x_k  - b  # r0
    p_k = -r_k           # p0

    xarrar = [x_k]
    rkarrar = [r_k]

    while r_k[0] >= 1e-8:
        r_k.shape=(3,1)
        p_k.shape=(3,1)
        alpha_k = - (np.transpose(r_k) * p_k) / (np.transpose(p_k) * A * p_k)
        x_k = x_k +  p_k * alpha_k
        r_k = A * x_k - b
        beta_k = (np.transpose(r_k) * A * p_k) / (np.transpose(p_k) * A * p_k)
        p_k = - r_k +   p_k * beta_k

        xarrar.append(x_k)
        rkarrar.append(r_k)
        k += 1

    print("xarrar",str(xarrar))
    print("\nrkarrar",str(rkarrar))
    return x_k


if __name__ == "__main__":
    A= mat([[6, 8, 1],[1, 4, 5],[7, 1, 3]])
    b = mat([[5.2], [1.3], [4.4]])
    x_0 = mat([[10.0], [10.0], [10.0]])
    xStar = CG_Preliminary(A, b, x_0)
    print("\n x_star:",str(xStar))
