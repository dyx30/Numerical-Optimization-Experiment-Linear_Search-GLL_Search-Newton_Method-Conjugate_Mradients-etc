# 实现阻尼牛顿方法的迭代
import numpy as np
import sympy as sym
import sys
import os
import math
import functools
from damped_newton import damp_newton
from conjugate_gradients import conjugate,calc_beta
from quasi_newton import L_BFGS



# 实验1+Freudenstein and Roth function
def FR(X):
    x1 = X[0]
    x2 = X[1]
    f1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    return (f1 ** 2 + f2 ** 2)

def diff_FR(X):
    x1, x2 = sym.symbols("x1,x2")
    f1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    FR = (f1 ** 2 + f2 ** 2)
    diff_x1 = sym.diff(FR, x1)
    diff_x2 = sym.diff(FR, x2)
    return np.array([diff_x1.subs({x1: X[0], x2: X[1]}), diff_x2.subs({x1: X[0], x2: X[1]})], dtype = float)

def hess_FR(X):
    x1, x2 = sym.symbols("x1,x2")
    f1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    f2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    FR = (f1 ** 2 + f2 ** 2)
    gx_1 = sym.diff(FR, x1)
    gx_2 = sym.diff(FR, x2)
    Gx_11 = sym.diff(gx_1, x1)
    Gx_12 = sym.diff(gx_1, x2)
    Gx_22 = sym.diff(gx_2, x2)
    G = [[Gx_11.subs({x1: X[0], x2: X[1]}), Gx_12.subs({x1: X[0], x2: X[1]})],
         [Gx_12.subs({x1: X[0], x2: X[1]}), Gx_22.subs({x1: X[0], x2: X[1]})]]
    return np.array(G, dtype = float)


# X=[0.5,-2]
# X=[8,0]
# X=[5,4.5]
# damp_newton(X, FR, diff_FR, hess_FR, cholesky=False, search_mode="ILS")
# damp_newton(X, FR, diff_FR, hess_FR, M=10,search_mode="GLL")

#实验1+Box three-dimensional function
def f_i_box(x, i):
    t_i = 0.1 * i
    return (sym.exp(-t_i * x[0]) - sym.exp(-t_i * x[1]) - x[2] * (sym.exp(-t_i) - sym.exp(-10 * t_i)))

def Box3(X):
    result = 0
    for i in range(1, 4):  # 假设i从1到3
        result += f_i_box(X, i) ** 2
    return result

def diff_Box3(X):
    x1, x2, x3 = sym.symbols("x1,x2,x3")
    result = 0
    for i in range(1, 4):
        f = f_i_box([x1, x2, x3], i)
        result += f ** 2
    diff_x1 = sym.diff(result, x1)
    diff_x2 = sym.diff(result, x2)
    diff_x3 = sym.diff(result, x3)
    return np.array([diff_x1.subs({x1: X[0], x2: X[1], x3: X[2]}),
                     diff_x2.subs({x1: X[0], x2: X[1], x3: X[2]}),
                     diff_x3.subs({x1: X[0], x2: X[1], x3: X[2]})], dtype=float)

def hess_Box3(X):
    x1, x2, x3 = sym.symbols("x1,x2,x3")
    result = 0
    for i in range(1, 4):
        f = f_i_box([x1, x2, x3], i)
        result += f ** 2
    gx_1 = sym.diff(result, x1)
    gx_2 = sym.diff(result, x2)
    gx_3 = sym.diff(result, x3)
    Gx_11 = sym.diff(gx_1, x1)
    Gx_12 = sym.diff(gx_1, x2)
    Gx_13 = sym.diff(gx_1, x3)
    Gx_22 = sym.diff(gx_2, x2)
    Gx_23 = sym.diff(gx_2, x3)
    Gx_33 = sym.diff(gx_3, x3)
    G = [[Gx_11.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_12.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_13.subs({x1: X[0], x2: X[1], x3: X[2]})],
         [Gx_12.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_22.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_23.subs({x1: X[0], x2: X[1], x3: X[2]})],
         [Gx_13.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_23.subs({x1: X[0], x2: X[1], x3: X[2]}), Gx_33.subs({x1: X[0], x2: X[1], x3: X[2]})]]
    return np.array(G, dtype=float)
    

# X=[0,10,15]
# damp_newton(X, Box3, diff_Box3, hess_Box3, cholesky=True, search_mode="ILS")    
# damp_newton(X, Box3, diff_Box3, hess_Box3, M=15,search_mode="GLL")


#实验1+Variably dimensional function

def f_i_VD(x, i): 
    n = len(x)  # 获取向量 x 的长度
    if 1 <= i <= n: 
        return x[i - 1] - 1 
    elif i == n + 1: 
        return np.sum([j * (x[j - 1] - 1) for j in range(1, n + 1)]) 
    elif i == n + 2: 
        return np.sum([j * (x[j - 1] - 1) for j in range(1, n + 1)]) ** 2

# 定义 VD 函数
def VD(X): 
    result = 0 
    for i in range(1, len(X) + 3):  # 循环范围：1 到 len(X) + 2
        result += f_i_VD(X, i) ** 2 
    return result

# 定义 diff_VD 函数，计算梯度
def diff_VD(X): 
    n = len(X)
    x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))  # 创建符号变量 x1, x2, ..., xn
    result = 0 
    for i in range(1, n + 3):  # 循环范围：1 到 n + 2
        f = f_i_VD(x_sym, i) 
        result += f ** 2  # 累加每个 f_i(x)^2
    grad = [] 
    for j in range(n):  # 对每个变量 x_j 进行求导
        grad.append(sym.diff(result, x_sym[j]))
    # 计算梯度的值并返回
    return np.array([g.subs({x_sym[k]: X[k] for k in range(n)}) for g in grad], dtype=float)

# 定义 hess_VD 函数，计算海森矩阵
def hess_VD(X): 
    n = len(X)
    x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))  # 创建符号变量 x1, x2, ..., xn
    result = 0 
    for i in range(1, n + 3):  # 循环范围：1 到 n + 2
        f = f_i_VD(x_sym, i) 
        result += f ** 2  # 累加每个 f_i(x)^2
    hess = [] 
    for j in range(n):  # 对每个变量 x_j 计算二阶导数
        row = [] 
        for k in range(n): 
            row.append(sym.diff(sym.diff(result, x_sym[j]), x_sym[k])) 
        hess.append(row)
    # 计算海森矩阵的值并返回
    return np.array([[hess[j][k].subs({x_sym[m]: X[m] for m in range(n)}) for k in range(n)] for j in range(n)], dtype=float)


# n=3
# X=np.array([1 - (j / n) for j in range(1, n + 1)])
# damp_newton(X, VD, diff_VD, hess_VD, search_mode="ILS")
# damp_newton(X, VD, diff_VD, hess_VD, M=5,search_mode="GLL")



#实验2+Rosenbrock function

def R(X):
    x1 = X[0]
    x2 = X[1]
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


def diff_R(X):
    x1, x2 = sym.symbols("x1 x2")
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    diff_x1 = sym.diff(f, x1)
    diff_x2 = sym.diff(f, x2)
    return np.array([diff_x1.subs({x1: X[0], x2: X[1]}), diff_x2.subs({x1: X[0], x2: X[1]})], dtype=float)


def hess_R(X):
    x1, x2 = sym.symbols("x1 x2")
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    gx_1 = sym.diff(f, x1)
    gx_2 = sym.diff(f, x2)
    Gx_11 = sym.diff(gx_1, x1)
    Gx_12 = sym.diff(gx_1, x2)
    Gx_22 = sym.diff(gx_2, x2)
    G = [[Gx_11.subs({x1: X[0], x2: X[1]}), Gx_12.subs({x1: X[0], x2: X[1]})],
         [Gx_12.subs({x1: X[0], x2: X[1]}), Gx_22.subs({x1: X[0], x2: X[1]})]]
    return np.array(G, dtype=float)

# X=[-1.2,1]
# damp_newton(X, R, diff_R, hess_R, search_mode="ILS")
# damp_newton(X, R, diff_R, hess_R, cholesky=True, search_mode="ILS")

# n=10
# X=generate_x0(n)
# damp_newton(X, VD, diff_VD, hess_VD, search_mode="ILS")

def PBS(X):
    x1, x2 = X
    f1 = 10000 * x1 * x2 - 1
    f2 = math.exp(-x1) + math.exp(-x2) - 1.0001
    return f1 ** 2 + f2 ** 2

def diff_PBS(X):
    x1, x2 = sym.symbols("x1 x2")
    f1 = 10000 * x1 * x2 - 1
    f2 = sym.exp(-x1) + sym.exp(-x2) - 1.0001
    result = (f1 ** 2 + f2 ** 2)
    diff_x1 = sym.diff(result, x1)
    diff_x2 = sym.diff(result, x2)
    return np.array([diff_x1.subs({x1: X[0], x2: X[1]}), diff_x2.subs({x1: X[0], x2: X[1]})], dtype=float)

def hess_PBS(X):
    x1, x2 = sym.symbols("x1 x2")
    f1 = 10000 * x1 * x2 - 1
    f2 = sym.exp(-x1) + sym.exp(-x2) - 1.0001
    result = (f1 ** 2 + f2 ** 2)
    gx_1 = sym.diff(result, x1)
    gx_2 = sym.diff(result, x2)
    Gx_11 = sym.diff(gx_1, x1)
    Gx_12 = sym.diff(gx_1, x2)
    Gx_22 = sym.diff(gx_2, x2)
    G = [[Gx_11.subs({x1: X[0], x2: X[1]}), Gx_12.subs({x1: X[0], x2: X[1]})],
         [Gx_12.subs({x1: X[0], x2: X[1]}), Gx_22.subs({x1: X[0], x2: X[1]})]]
    return np.array(G, dtype=float)


# conjugate(X0,extended_rosenbrock,diff_extended_rosenbrock,hess_extended_rosenbrock, N = 100000,eps = 1e-6,mode="FR")


# variably dimensioned function上文已经定义，VD,diff_VD,hess_VD
# n=10
# X0=np.array([1 - (j / n) for j in range(1, n + 1)])
# L_BFGS(X0,VD,diff_VD,m=5, N=100000,eps = 1e-6)


# discrete_boundary_value function
def discrete_boundary_value(x):
    n = len(x)
    h = 1 / (n + 1)
    result = 0
    for i in range(n):
        t_i = i * h
        term1 = 2 * x[i]
        term2 = x[i - 1] if i > 0 else 0
        term3 = x[i + 1] if i < n-1 else 0
        term4 = h ** 2 * ((x[i] + t_i + 1) ** 3 / 2)
        result += (term1 - term2 - term3 + term4) ** 2
    return result

def diff_discrete_boundary_value(x):
    n = len(x)
    x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))
    h = 1 / (n + 1)
    result = 0
    for i in range(n):
        t_i = i * h
        term1 = 2 * x[i]
        term2 = x[i - 1] if i > 0 else 0
        term3 = x[i + 1] if i < n-1 else 0
        term4 = h ** 2 * ((x[i] + t_i + 1) ** 3 / 2)
        result += (term1 - term2 - term3 + term4) ** 2
    grad = []
    for j in range(n):
        grad.append(sym.diff(result, x_sym[j]))
    return np.array([g.subs({x_sym[k]: x[k] for k in range(n)}) for g in grad], dtype=float)

def hess_discrete_boundary_value(x):
    n = len(x)
    x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))
    h = 1 / (n + 1)
    result = 0
    for i in range(n):
        t_i = i * h
        term1 = 2 * x[i]
        term2 = x[i - 1] if i > 0 else 0
        term3 = x[i + 1] if i < n-1 else 0
        term4 = h ** 2 * ((x[i] + t_i + 1) ** 3 / 2)
        result += (term1 - term2 - term3 + term4) ** 2
    hess = []
    for j in range(n):
        row = []
        for k in range(n):
            row.append(sym.diff(sym.diff(result, x_sym[j]), x_sym[k]))
        hess.append(row)
    return np.array([[h.subs({x_sym[m]: x[m] for m in range(n)}) for h in row] for row in hess], dtype=float)
# n=60
# X0=np.array([(j/(n+1)) * ((j/(n+1))-1) for j in range(1,n+1)])
# L_BFGS(X0,discrete_boundary_value,diff_discrete_boundary_value,m=5, N=100000,eps = 1e-6)