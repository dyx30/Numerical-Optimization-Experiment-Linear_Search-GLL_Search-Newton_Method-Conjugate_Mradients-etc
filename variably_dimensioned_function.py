import numpy as np
import sympy as sym
import time
import functools
from damped_newton import damp_newton
from conjugate_gradients import conjugate,calc_beta
from quasi_newton import L_BFGS
from numba import jit

def variably_dimensioned(X):
    """
    Compute the value of the Variably Dimensioned function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        float: Function value.
    """
    n = len(X)
    f_n1 = np.sum([(j + 1) * (X[j] - 1) for j in range(n)])
    f_n2 = f_n1**2
    f_val = np.sum((X - 1)**2) + f_n1**2 + f_n2**2
    return f_val

def diff_variably_dimensioned(X):
    """
    Compute the gradient of the Variably Dimensioned function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        ndarray: Gradient vector.
    """
    n = len(X)
    f_n1 = np.sum([(j + 1) * (X[j] - 1) for j in range(n)])
    f_n2 = f_n1**2
    grad = np.zeros_like(X)
    for i in range(n):
        grad[i] = 2 * (X[i] - 1) + 2 * f_n1 * (i + 1) + 4 * f_n2 * f_n1 * (i + 1)
    return grad

def hess_variably_dimensioned(X):
    """
    Compute the Hessian matrix of the Variably Dimensioned function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        ndarray: Hessian matrix of size (n, n).
    """
    n = len(X)
    f_n1 = np.sum([(j + 1) * (X[j] - 1) for j in range(n)])
    f_n2 = f_n1**2
    H = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                H[i, k] = 2 + 2 * (i + 1)**2 + 4 * f_n2 * (i + 1)**2
            else:
                H[i, k] = 2 * (i + 1) * (k + 1) + 4 * f_n2 * (i + 1) * (k + 1)
    return H
'''
class VariablyDimensionalOptimizer:
    def __init__(self, n):
        self.n = n
        self.x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))
        self.result_expr = self._build_function_expr()
        self.grad_expr = self._compute_gradient(self.result_expr)
        self.hess_expr = self._compute_hessian(self.result_expr)

        # Create functions for evaluation
        self.result_func = sym.lambdify(self.x_sym, self.result_expr, 'numpy')
        self.grad_func = [sym.lambdify(self.x_sym, g, 'numpy') for g in self.grad_expr]
        self.hess_func = [[sym.lambdify(self.x_sym, h, 'numpy') for h in row] for row in self.hess_expr]

    def _build_function_expr(self):
        h = 1 / (self.n + 1)
        result = 0
        for i in range(1, n + 3):  # 循环范围：1 到 n + 2
            if 1 <= i <= n: 
                result+=(self.x_sym[i - 1] - 1)**2
            elif i == n + 1: 
                result+=(np.sum([j * (self.x_sym[j - 1] - 1) for j in range(1, n + 1)]))**2 
            elif i == n + 2: 
                result+= (np.sum([j * (self.x_sym[j - 1] - 1) for j in range(1, n + 1)]) ** 2)**2
        return result

    def _compute_gradient(self, expr):
        grad = []
        for j in range(self.n):
            grad.append(sym.diff(expr, self.x_sym[j]))
        return grad

    def _compute_hessian(self, expr):
        hess = []
        for j in range(self.n):
            row = []
            for k in range(self.n):
                row.append(sym.diff(sym.diff(expr, self.x_sym[j]), self.x_sym[k]))
            hess.append(row)
        return hess

    def variably_dimensional_value(self, x):
        return float(self.result_func(*x))

    def diff_variably_dimensional_value(self, x):
        grad = np.array([g(*x) for g in self.grad_func], dtype=float)
        return grad

    def hess_variably_dimensional_value(self, x):
        hess = np.array([[h(*x) for h in row] for row in self.hess_func], dtype=float)
        return hess'''
'''
# Usage example
n = 100
start_time=time.time()
X0 = np.array([1 - (j / n) for j in range(1, n + 1)])

# optimizer = VariablyDimensionalOptimizer(n)
# Create partial functions for convenience
# variably_dimensional_value_fn = functools.partial(optimizer.variably_dimensional_value)
# diff_variably_dimensional_value_fn = functools.partial(optimizer.diff_variably_dimensional_value)
# hess_variably_dimensional_value_fn = functools.partial(optimizer.hess_variably_dimensional_value)


diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
# 使用 L-BFGS 优化
# L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=3, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=5, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
# L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=15, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
# L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=30, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
# L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=50, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)

# 使用共轭梯度法优化
# conjugate(X0,variably_dimensioned, diff_variably_dimensioned,hess_variably_dimensioned,N=100000, eps=1e-6,funcname="VD",diff_time=diff_time,mode="FR")

n = 500
start_time=time.time()
X0 = np.array([1 - (j / n) for j in range(1, n + 1)])
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=5, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
'''

n = 1000
start_time=time.time()
X0 = np.array([1 - (j / n) for j in range(1, n + 1)])
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=5, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)


n = 5000
start_time=time.time()
X0 = np.array([1 - (j / n) for j in range(1, n + 1)])
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=5, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)

n = 10000
start_time=time.time()
X0 = np.array([1 - (j / n) for j in range(1, n + 1)])
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
L_BFGS(X0, variably_dimensioned, diff_variably_dimensioned, m=5, N=100000, eps=1e-6, funcname="VD",diff_time=diff_time)
