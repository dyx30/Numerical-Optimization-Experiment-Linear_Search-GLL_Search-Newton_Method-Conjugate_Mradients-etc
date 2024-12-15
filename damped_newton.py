# 实现基本牛顿方法的迭代
# 参数cholesky来控制是否对G进行修正
import numpy as np
import utils
from linear_search import inexact_line_search
from GLL_linear_search import GLL_search


def damp_newton(X, func, gfunc, hess_func, search_mode, M=10, cholesky=False, epsilon=1e-8, max_epoch=10000):
    """[阻尼牛顿法求步长公式:d = -G_k^{-1} * g_k]

    Args:
        X ([np.array]): [输入的X值]
        func ([回调函数]): [目标函数]
        gfunc ([回调函数]): [目标函数的一阶导函数]
        hess_func ([回调函数]): [目标函数的Hessian矩阵]
        search_mode (str, optional): [线搜索的模式（选择精确线搜索还是非精确线搜索）]. ['ILS' or 'GLL']
        M ([int]):[GLL法window的大小]
        epsilon ([float], optional): [当函数值下降小于epsilon,迭代结束]. 默认为1e-8.
        max_epoch (int, optional): [最大允许的迭代次数]. 默认为1000.

    Returns:
        返回求解得到的极小值点,极小值点对应的函数值和迭代次数
    """

    k = 1
    function_k = 0 #函数调用次数
    F0 = func(X)
    func_values = [F0]*M #记录每一步的函数值,在GLL中有用
    mk = 0 #GLL当中的mk初始值
    #计算下降方向d_k

    while True:
        G = hess_func(X)
        g = gfunc(X)
        # 把当前函数值加入func_values
        F = func(X)
        function_k += 1
        func_values[(k-1)%M]=F  #储存近M个函数迭代值

        if cholesky:
            G_ = utils.modified_Cholesky(G, cholesky)
            inv_hass = np.linalg.inv(G_)
            d = -np.dot(inv_hass , g)
        else:
            if utils.is_pos_def(G)==False:
                print("G非正定")
            inv_hass = np.linalg.inv(G)
            d = -np.dot(inv_hass , g)
        
        #计算步长
        if search_mode == "ILS":
            print("迭代第{iter}轮，函数调用次数{func_k},X取值为{X},下降方向为{d},当前函数值为{func_x}".format(iter=k,func_k=function_k,X=X,d=d,func_x=round(F, 8)))
            alpha_star, add_func_k = inexact_line_search(func, gfunc, X, d) 
        elif search_mode == "GLL":
            print("迭代第{iter}轮,函数调用次数{func_k},X取值为{X},下降方向为{d},当前函数值为{func_x}".format(iter=k,func_k=function_k,X=X,d=d,func_x=round(F, 8)))
            alpha_star, add_func_k,= GLL_search(func, gfunc, X, d, func_values) 
        else:
            raise ValueError("参数search_mode必须从['ILS', 'GLL']当中选择")
        
        X_new = X + d * alpha_star
        function_k = function_k + add_func_k + 1
        func_X_new = func(X_new)
        if abs(func_X_new - F) <= epsilon:
            print("因为函数值下降在{epsilon}以内,{mode}的阻尼牛顿法迭代结束,迭代轮次{iter},函数调用次数{func_k},最终X={X},最终函数值={func_X_new}".format(epsilon=epsilon, mode=search_mode, iter=k, func_k=function_k, X=X,func_X_new=F))
            return X_new, func_X_new, k, function_k
        if k > max_epoch:
            print("超过最大迭代次数：%d", max_epoch)
            return X_new, func_X_new, k, function_k
        X = X_new
        k += 1