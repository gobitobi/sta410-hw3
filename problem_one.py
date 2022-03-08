import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# import math


def newton_raphson_iteration(f, df, x0, t, method='default'):
    '''
    The function implements the Newton-Raphson method for finding root x* so `f(x*)=0`

    x0     : intial value
    t      : number of iterations (resulting in x_t or x_(t-2) with acceleration)
    df     : derivative of f
    f      : (default) lambda x: log(x)/(1+x)
    method : either 'aikens_accelerated' or 'default' (no acceleration)

    returns final_x,f(final_x)
    '''

    xt = np.ones(t+1)
    xt[0] = x0

    for i in range(1, t+1):
        xt[i] = xt[i-1] - f(xt[i-1])/df(xt[i-1])
        if f(xt[i]) == 0:
            break

    if method == 'aikens_accelerated':
        pass  # <complete>

    return xt, f(xt)
    return xt[-1], f(xt[-1])


def fixed_point_iteration(f, df, x0, t, a, method='default'):
    '''
    The function implements scaled Fixed-Point iteration finding root 
    `f(x*)=0` iff `x* = x* + af(x*)`

    x0     : intial value
    t      : number of iterations (resulting in x_t or x_(t-2) with acceleration)
    a      : scaling factor alpha
    df     : derivative of f
    f      : (default) lambda x: log(x)/(1+x)
    method : either 'aikens_accelerated' or 'default' (no acceleration)    

    returns final_x,f(final_x)
    '''

    xt = np.zeros(t+1)
    xt[0] = x0
    for i in range(1, t+1):
        xt[i] = a * df(xt[i-1]) + xt[i-1]
        # print(xt[i])

    if method == 'steffensens_method':
        pass  # <complete>

    return xt, f(xt)
    return xt[-1], f(xt[-1])


#################################
# np.log uses ln(x) and np.log10 uses log_10(x)
def f(x): return np.log(x)/(1+x)
def df(x): return (1+(1/x)-np.log(x)) / ((x+1)**2)
# def df(x): return (1+x-x*np.log(x)) / (x*((x+1)**2))
# def df2(x): return (2*(x**2)*np.log(x) - 1 -
#                     4*x - 3*(x**2)) / ((x**2)*(1+x)**3)


x0 = 0.1
t = 20
a = -0.5
eps = 1e-15
# xt1, f_of_xt1 = newton_raphson_iteration(f, df, x0, t, method='default')
# print('xt1: ', xt1)
# print('f_of_xt1: ', f_of_xt1)
# print('|f_of_xt1| < 1e-15: ', np.abs(f_of_xt1) < eps)

xt2, f_of_xt2 = fixed_point_iteration(f, df, x0, t, a, method='default')
print('xt2: ', xt2)
print('f_of_xt2: ', f_of_xt2)
#
#
#
#
# Questions 1-4 (2 points)
# For questions following questions use f, x0, a = lambda x: np.log(x)/(1+x), 0.1, -0.5.

# For questions 1 and 2, use method = "default".

# (0.5 points) What is the smallest input t for which newton_raphson_iteration | f(x_t) | <1e-15?
# (0.5 points) What is the smallest input t for which fixed_point_iteration | f(x_t) | <1e-15?
# For questions 3 and 4, use Aitken's
# acceleration.

# (0.5 points) What is the smallest input t for which newton_raphson_iteration | f(x_t) | <1e-15?
# (0.5 points) What is the smallest input t for which fixed_point_iteration | f(x_t) | <1e-15?
# # 2 points (0.5 each)
p1q1 = 0  # an integer, e.g., 10
p1q2 = 0  # an integer, e.g., 10
p1q3 = 0  # an integer, e.g., 10
p1q4 = 0  # an integer, e.g., 10
#
#
#
# Questions 5-8 (2 points)
# For the following questions, choose one of

# "newton_raphson_iteration - default"
# "newton_raphson_iteration - aikens_accelerated"
# "fixed_point_iteration - default"
# "fixed_point_iteration - steffensens_method"
# "neither newton_raphson_iteration nor fixed_point_iteration"
# (0.5 points) Which method is preferable for f, x0, a = lambda x: np.log(x)/(1+x), 2.0, -0.5?
# (0.5 points) Which method is preferable for f, x0, a = lambda x: np.log(x)/(1+x), 4.0, -0.5?
# (0.5 points) Which method is preferable for f, x0, a = lambda x: np.log(x)/(1+x), 4.0, 0.5?
# (0.5 points) Which method is preferable for f, x0, a = lambda x: np.log(x)/(1+x), 0.5, 0.5?
# # 2 points (0.5 each)
p1q5 = ""  # a string "<option chosen from above>"
p1q6 = ""  # a string "<option chosen from above>"
p1q7 = ""  # a string "<option chosen from above>"
p1q8 = ""  # a string "<option chosen from above>"
