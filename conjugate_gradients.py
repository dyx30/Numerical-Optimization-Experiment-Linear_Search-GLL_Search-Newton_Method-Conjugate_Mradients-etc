import numpy as np
import time
import os
from linear_search import inexact_line_search

def calc_beta(G,X,X_new,mode):
   
    g_new = G(X_new)  # 对新点处的梯度相关向量元素限制最大值为 1e10
    g = G(X)          # 对旧点处的梯度相关向量元素限制最大值为 1e10

    beta=0
    # 按Fletcher-Reeves公式计算beta值
    if mode == "FR":
        beta = np.dot(g_new, g_new) / np.dot(g, g)  # 按FR公式计算beta
    # 按Polak-Ribière-Polyak相关修正规则计算beta值
    elif mode =="PRP+":
        beta = np.dot(g_new, (g_new - g)) / np.dot(g, g) # 按PRP+规则算beta
    return beta


def conjugate(X0, F, G, H, N=100000, eps=1e-6, mode="FR", output_dir="./outputs", funcname="", diff_time=-1): 
    # 获取当前时间并格式化为字符串，方便用作文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tic = time.time()  # 记录开始时间
    fun_iter = 0  # 函数调用计数器
    n = len(X0)  # 变量的维数
    X = X0
    X_new = np.zeros(n)
    d = -G(X)

    # 格式化输出文件名：L_BFGS_m{m}_N{N}_{timestamp}.txt
    output_file = f"{output_dir}/conjugate_n{n}_mode-{mode}_func{funcname}_{timestamp}.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录及其父目录

    # 打开输出文件进行写入
    with open(output_file, 'w') as f:
        f.write(f"微分用时{diff_time}\n")
        for i in range(N):  # 迭代最大次数
            if np.linalg.norm(G(X)) < eps:  # 判断 当前梯度是否满足终止准则
                ela = time.time() - tic  # 记录算法运行时间
                f.write(f"共轭梯度下降法迭代结束,运行时间：{ela:.4f},迭代{i+1}轮，函数调用次数{fun_iter},X*为{np.array_str(X_new)},当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
                break
            else:
                X = X_new.copy()  # 更新当前点
                al_k, ite = inexact_line_search(F, G, X, d)  # 线搜索，找到步长 al_k
                fun_iter += ite
                X_new = X + al_k * d
                f.write(f"迭代第{i+1}轮，函数调用次数{fun_iter},X取值为{X_new[:5]},当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")         
                beta = calc_beta(G, X, X_new, mode)
                # print("beta:{}\n".format(beta))
                d = -G(X_new) + beta * d
        # 如果没有提前结束，说明已达到最大迭代次数
        if np.linalg.norm(G(X)) >= eps:
            ela = time.time() - tic  # 记录算法运行时间
            f.write(f"已达最大轮数，共轭梯度下降法迭代结束,运行时间：{ela:.4f},迭代{N}轮，函数调用次数{fun_iter},X*为{np.array_str(X_new)},当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")