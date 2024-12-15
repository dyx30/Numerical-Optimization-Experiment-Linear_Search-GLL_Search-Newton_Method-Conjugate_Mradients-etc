import numpy as np
import sympy as sym
import time
from conjugate_gradients import conjugate,calc_beta
from quasi_newton import L_BFGS

def extended_rosenbrock(X):
    """
    Compute the value of the Extended Rosenbrock function.
    Args:
        X (ndarray): Input vector of length n (even).
    Returns:
        float: Function value.
    """
    n = len(X)
    assert n % 2 == 0, "Dimension of input must be even."
    f_val = 0
    for l in range(n // 2):
        x2l_1 = X[2 * l]
        x2l = X[2 * l + 1]
        f_val += 100 * (x2l - x2l_1**2)**2 + (1 - x2l_1)**2
    return f_val

def diff_extended_rosenbrock(X):
    """
    Compute the gradient of the Extended Rosenbrock function.
    Args:
        X (ndarray): Input vector of length n (even).
    Returns:
        ndarray: Gradient vector.
    """
    n = len(X)
    assert n % 2 == 0, "Dimension of input must be even."
    grad = np.zeros_like(X)
    for l in range(n // 2):
        x2l_1 = X[2 * l]
        x2l = X[2 * l + 1]
        grad[2 * l] = -400 * (x2l - x2l_1**2) * x2l_1 - 2 * (1 - x2l_1)
        grad[2 * l + 1] = 200 * (x2l - x2l_1**2)
    return grad

def hess_extended_rosenbrock(X):
    """
    Compute the Hessian matrix of the Extended Rosenbrock function.
    Args:
        X (ndarray): Input vector of length n (even).
    Returns:
        ndarray: Hessian matrix of size (n, n).
    """
    n = len(X)
    assert n % 2 == 0, "Dimension of input must be even."
    H = np.zeros((n, n))
    for l in range(n // 2):
        x2l_1 = X[2 * l]
        x2l = X[2 * l + 1]
        H[2 * l, 2 * l] = -400 * x2l + 1200 * x2l_1**2 + 2
        H[2 * l + 1, 2 * l + 1] = 200
        H[2 * l, 2 * l + 1] = H[2 * l + 1, 2 * l] = -400 * x2l_1
    return H
'''
class RosenbrockOptimizer:
    def __init__(self, n):
        self.n = n
        self.x_sym = sym.symbols(" ".join([f"x{i}" for i in range(n)]))
        self.result_expr = self._build_function_expr()
        self.grad_expr = self._compute_gradient(self.result_expr)
        self.hess_expr = self._compute_hessian(self.result_expr)

        # 将符号表达式转换为数值计算的函数
        self.result_func = sym.lambdify(self.x_sym, self.result_expr, 'numpy')
        self.grad_func = [sym.lambdify(self.x_sym, g, 'numpy') for g in self.grad_expr]
        self.hess_func = [[sym.lambdify(self.x_sym, h, 'numpy') for h in row] for row in self.hess_expr]

    def _build_function_expr(self):
        result = 0
        for i in range(int(self.n / 2)):
            term1 = 10 * (self.x_sym[2 * i] - self.x_sym[2 * i - 1] ** 2)
            term2 = 1 - self.x_sym[2 * i - 1]
            result += term1 ** 2 + term2 ** 2
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
    def extended_rosenbrock(self, x):
        return float(self.result_func(*x))
    def diff_extended_rosenbrock(self, x):
        grad = np.array([g(*x) for g in self.grad_func], dtype=float)
        return grad
    def hess_extended_rosenbrock(self, x):
        hess = np.array([[h(*x) for h in row] for row in self.hess_func], dtype=float)
        return hess
'''
# 设置参数
'''
n = 100
X0 = np.array([1 if x % 2 == 0 else -1.2 for x in range(1, n + 1)])
start_time=time.time()
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="FR",funcname="ERB",diff_time=diff_time)

n = 500
X0 = np.array([1 if x % 2 == 0 else -1.2 for x in range(1, n + 1)])
start_time=time.time()
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="PRP+",funcname="ERB",diff_time=diff_time)

n = 1000
X0 = np.array([1 if x % 2 == 0 else -1.2 for x in range(1, n + 1)])
start_time=time.time()
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="FR",funcname="ERB",diff_time=diff_time)
'''
n = 5000
X0 = np.array([1 if x % 2 == 0 else -1.2 for x in range(1, n + 1)])
start_time=time.time()
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="FR",funcname="ERB",diff_time=diff_time)

n = 10000
X0 = np.array([1 if x % 2 == 0 else -1.2 for x in range(1, n + 1)])
start_time=time.time()
diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="FR",funcname="ERB",diff_time=diff_time)
# 使用共轭梯度法优化
# conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="FR",funcname="ERB",diff_time=diff_time)
# conjugate(X0,extended_rosenbrock, diff_extended_rosenbrock,hess_extended_rosenbrock, eps=1e-6,mode="PRP+",funcname="ERB",diff_time=diff_time)