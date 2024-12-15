import numpy as np


def inexact_line_search(func, gfunc, X, d, rho=1e-4, sigma=0.9, start=0, end=1e10, alpha0=1e-6, t=5):
    """
    Args:
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        start ([int]): [步长下界]. 默认为 0.
        end ([type]): [步长上界]. 默认为 1e10.
        t([int]):[alpha增大的速率]. 默认为5.
    Returns:
        alpha_next([float]): [搜索得到的步长]]
    """
    alpha = alpha0
    func_k = 1
    f0, gf0 = func(X), gfunc(X)
    gkdk = gf0.dot(d)
    strong_wolfe_boundary = sigma * abs(gkdk)
    while True:
        func_k += 1
        fAlpha, gfAlpha = func(X + alpha * d), gfunc(X + alpha * d)
        if abs(start - end) < 1e-15:
            alpha_next = alpha
            break
        armijo_boundary = f0 + rho * gkdk * alpha
        armijo_condition = (fAlpha <= armijo_boundary)

        gkAlpha_dk = gfAlpha.dot(d)
        wolfe_condition = (abs(gkAlpha_dk) <= strong_wolfe_boundary)

        # update start or end point or stop iteration
        if armijo_condition == False:
            end = alpha
            alpha = (start + end) / 2
        elif wolfe_condition == False:
            start = alpha
            if end < 1e10:
                alpha = (start + end) / 2
            else:
                alpha = t * alpha
        else:
            alpha_next = alpha
            break

    return alpha_next, func_k

