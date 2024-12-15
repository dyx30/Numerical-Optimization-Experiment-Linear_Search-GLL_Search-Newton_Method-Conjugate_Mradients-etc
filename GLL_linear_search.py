
def GLL_search(func, gfunc, X, d, window, a=1e5, sigma=0.5, rho=0.5):
    """ 非单调线搜索GLL准则

    Args:
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        X ([np.array]]): [初值点]
        d ([np.array]]): [下降方向]
        window ([np.array]]): [之前步的函数值],window的尺度可取5,10,15.
        last_m ([int]]): [m_(k-1)]
        M (int, optional): [用于限制m(k)上限的参数]. Defaults to 10.
        a (int, optional): [初始步长]. Defaults to 0.5.
        sigma (int, optional): [决定调整步长快慢的系数]. 默认为0.5.
        rho (float, optional): [用来控制步长选择的条件]. 默认为0.1.
    Returns:
        [float]: [搜索得到的步长]]
    """

    func_k = 0 # 函数调用次数
    gf0 = gfunc(X)
    gkdk = gf0.dot(d)
    max_fx = max(window)
    hk = 0
    while True:
        alpha = sigma ** hk * a
        func_k += 1
        if func(X + alpha * d) <= max_fx + rho * alpha * gkdk:
            return alpha, func_k
        else:
            hk += 1