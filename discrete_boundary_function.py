import numpy as np
import sympy as sym
import time
import functools
from damped_newton import damp_newton
from conjugate_gradients import conjugate,calc_beta
from quasi_newton import L_BFGS


def discrete_boundary(X):
    """
    Compute the value of the discrete integral equation function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        float: Function value.
    """
    n = len(X)
    h = 1 / (n + 1)
    t = np.array([i * h for i in range(1, n + 1)])
    f = np.zeros(n)

    for i in range(n):
        xi = X[i]
        xim1 = X[i - 1] if i > 0 else 0
        xip1 = X[i + 1] if i < n - 1 else 0
        f[i] = 2 * xi - xim1 - xip1 + (h**2) * ((xi + t[i] + 1)**3) / 2

    return np.sum(f**2)

def diff_discrete_boundary(X):
    """
    Compute the gradient of the discrete integral equation function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        ndarray: Gradient vector.
    """
    n = len(X)
    h = 1 / (n + 1)
    t = np.array([i * h for i in range(1, n + 1)])
    f = np.zeros(n)
    grad = np.zeros(n)

    for i in range(n):
        xi = X[i]
        xim1 = X[i - 1] if i > 0 else 0
        xip1 = X[i + 1] if i < n - 1 else 0
        f[i] = 2 * xi - xim1 - xip1 + (h**2) * ((xi + t[i] + 1)**3) / 2

    for k in range(n):
        for i in range(n):
            if k == i:
                grad[k] += 2 * f[i] * (2 - (h**2) * 3 * (X[k] + t[k] + 1)**2 / 2)
            elif k == i - 1 or k == i + 1:
                grad[k] += 2 * f[i] * -1

    return grad
    

def hess_discrete_boundary(X):
    """
    Compute the Hessian matrix of the discrete integral equation function.
    Args:
        X (ndarray): Input vector of length n.
    Returns:
        ndarray: Hessian matrix.
    """
    n = len(X)
    h = 1 / (n + 1)
    t = np.array([i * h for i in range(1, n + 1)])
    H = np.zeros((n, n))

    for i in range(n):
        if i > 0:
            H[i, i - 1] = -2  # Off-diagonal
            H[i - 1, i] = -2  # Off-diagonal symmetry
        H[i, i] = 4 + 2 * (h**2) * 3 * (X[i] + t[i] + 1)

    return H
   

'''class DiscreteBoundaryOptimizer:
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
        for i in range(self.n):
            t_i = i * h
            term1 = 2 * self.x_sym[i]
            term2 = self.x_sym[i - 1] if i > 0 else 0
            term3 = self.x_sym[i + 1] if i < self.n - 1 else 0
            term4 = h ** 2 * ((self.x_sym[i] + t_i + 1) ** 3 / 2)
            result += (term1 - term2 - term3 + term4) ** 2
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

    def discrete_boundary_value(self, x):
        return float(self.result_func(*x))

    def diff_discrete_boundary_value(self, x):
        grad = np.array([g(*x) for g in self.grad_func], dtype=float)
        return grad

    def hess_discrete_boundary_value(self, x):
        hess = np.array([[h(*x) for h in row] for row in self.hess_func], dtype=float)
        return hess'''

# Usage example
n = 1000
start_time=time.time()
X0 = np.array([(j/(n+1)) * ((j/(n+1))-1) for j in range(1,n+1)])
#optimizer = DiscreteBoundaryOptimizer(n)
# Create partial functions for convenience
# discrete_boundary_value_fn = functools.partial(optimizer.discrete_boundary_value)
# diff_discrete_boundary_value_fn = functools.partial(optimizer.diff_discrete_boundary_value)
# hess_discrete_boundary_value_fn = functools.partial(optimizer.hess_discrete_boundary_value)


diff_time=time.time()-start_time
print("微分用时{}".format(diff_time))
# 使用 L-BFGS 优化
# L_BFGS(X0, discrete_boundary, diff_discrete_boundary, m=3, N=100000, eps=1e-6, funcname="DB",diff_time=diff_time)
# L_BFGS(X0, discrete_boundary, diff_discrete_boundary, m=5, N=100000, eps=1e-6, funcname="DB",diff_time=diff_time)
# L_BFGS(X0, discrete_boundary, diff_discrete_boundary, m=15, N=100000, eps=1e-6, funcname="DB",diff_time=diff_time)
# L_BFGS(X0, discrete_boundary, diff_discrete_boundary, m=30, N=100000, eps=1e-6, funcname="DB",diff_time=diff_time)
# L_BFGS(X0, discrete_boundary, diff_discrete_boundary, m=50, N=100000, eps=1e-6, funcname="DB",diff_time=diff_time)

# 使用共轭梯度法优化
conjugate(X0,discrete_boundary, diff_discrete_boundary,hess_discrete_boundary,mode="FR",N=100000, eps=1e-6,funcname="DB",diff_time=diff_time)