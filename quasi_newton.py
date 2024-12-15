import numpy as np
import time
import os
from linear_search import inexact_line_search

def L_BFGS(X0, F, G, m=5, N=100000, eps=1e-6, output_dir="./outputs",funcname="",diff_time=-1):
    # 获取当前时间并格式化为字符串，方便用作文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tic = time.time()  # 记录开始时间
    n = len(X0)  # 变量的维数
    # 格式化输出文件名：L_BFGS_m{m}_N{N}_{timestamp}.txt
    output_file = f"{output_dir}/L_BFGS_n{n}_m{m}_func{funcname}_{timestamp}.txt"
    hypereps=1e-10

    fun_iter = 0  # 函数调用计数器
    X = X0
    X_new = np.zeros_like(X)  # 用于存储新的解
    S = np.zeros([m, n])  # 用于存储步长差（S矩阵）
    Y = np.zeros([m, n])  # 用于存储梯度差（Y矩阵）
    alpha = np.zeros(m)  # 用于存储在L-BFGS更新中使用的系数
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录及其父目录
    # 打开文件以进行写入
    with open(output_file, "w") as f:
        f.write(f"微分用时{diff_time}\n")
        for i in range(N):   # 迭代最大次数
            if i <= m:
                if i == 0:  # 第一轮迭代，直接使用梯度信息
                    al_k, ite = inexact_line_search(F, G, X, -G(X)) 
                    fun_iter += ite  # 累计函数调用次数
                    X_new = X - G(X) * al_k  # 更新解 X_new
                    f.write(f"迭代第{i+1}轮，函数调用次数{fun_iter}, X取值为{X_new[:5]}, 当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
                    f.flush()  # 强制刷新输出
                else:
                    s = X_new - X  # 计算步长差 s
                    y = G(X_new) - G(X)  # 计算梯度差 y
                    S[i - 1:i, :] = s  # 将当前步长差存入 S 矩阵
                    Y[i - 1:i, :] = y  # 将当前梯度差存入 Y 矩阵
                    tmp = (s * y).sum() / ((y * y).sum()+hypereps)  # 计算步长与梯度的比例因子
                    q = G(X_new)  # 当前梯度
                    # 反向循环计算 alpha 值并调整 q
                    for j in range(i - 1, -1, -1):  
                        alpha[j] = (S[j:j + 1, :] * q).sum() / ((S[j:j + 1, :] * Y[j:j + 1, :]).sum()+hypereps)  # 计算 alpha[j]
                        q = q - alpha[j] * Y[j:j + 1, :]  # 更新 q
                    r = tmp * q  # 初始化搜索方向 r
                    # 前向循环调整搜索方向 r
                    for j in range(0, i, 1):  
                        beta = (Y[j:j + 1, :] * r).sum() / ((S[j:j + 1, :] * Y[j:j + 1, :]).sum()+hypereps)  # 计算 beta[j]
                        r = r + (alpha[j] - beta) * S[j:j + 1, :]  # 更新 r
                    # 使用线搜索更新步长 al_k
                    r = r.flatten()
                    al_k, ite = inexact_line_search(F, G, X_new, -r)  
                    fun_iter += ite  # 累计函数调用次数
                    X = X_new.copy()  # 更新当前解
                    X_new = X_new + al_k * (-r)  # 使用 r 更新新的解
                    # 打印当前迭代信息到文件
                    f.write(f"迭代第{i+1}轮，函数调用次数{fun_iter}, X取值为{X_new[:5]}, 当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
                    f.flush()  # 强制刷新输出

            else:
                # 在历史步数超过 m 时，开始使用 L-BFGS 方法
                s = X_new - X  # 步长差
                y = G(X_new) - G(X)  # 梯度差
                # 更新 S 和 Y 矩阵
                S[0:m - 1, :] = S[1:m, :]  # 迁移历史步长差
                Y[0:m - 1, :] = Y[1:m, :]  # 迁移历史梯度差
                S[m - 1:m, :] = s  # 将当前步长差加入历史步长差
                Y[m - 1:m, :] = y  # 将当前梯度差加入历史梯度差
                tmp = (s * y).sum() / ((y * y).sum()+hypereps)  # 计算步长差与梯度差的内积的比值
                q = G(X_new)  # 当前解的梯度
                # 反向迭代计算 alpha 和搜索方向
                for j in range(m - 1, -1, -1):
                    alpha[j] = (S[j:j + 1, :] * q).sum() / ((S[j:j + 1, :] * Y[j:j + 1, :]).sum()+hypereps)  # 计算 alpha[j]
                    q = q - alpha[j] * Y[j:j + 1, :]  # 更新 q
                r = tmp * q  # 使用计算得到的比值和 q 计算 r
                # 前向迭代计算搜索方向
                for j in range(0, m, 1):
                    beta = (Y[j:j + 1, :] * r).sum() / ((S[j:j + 1, :] * Y[j:j + 1, :]).sum()+hypereps)  # 计算 beta[j]
                    r = r + S[j:j + 1, :] * (alpha[j] - beta)  # 更新 r
                    # print(r)
                r = r.flatten()
                while np.linalg.norm(r)<1:
                    r=r*100
                al_k, ite = inexact_line_search(F, G, X_new, -r)   # 线搜索，找到步长 al_k
                fun_iter += ite  # 累计函数调用次数
                X = X_new.copy()  # 更新当前解
                X_new = X_new + (-r) * al_k  # 使用 r 更新 X_new
                # 打印当前迭代信息到文件
                f.write(f"迭代第{i+1}轮，函数调用次数{fun_iter}, X取值为{X_new[:5]}, 当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
                f.flush()  # 强制刷新输出

            # 判断是否满足停止条件（梯度的范数小于某个阈值）
            if np.linalg.norm(G(X)) < eps :
            # if np.linalg.norm(G(X)) < eps or abs(F(X)-F(X_new))<1e-13 or(n==10000 and np.linalg.norm(G(X)) < 1 and  abs(F(X)-F(X_new))<1e-9):

                ela = time.time() - tic  # 记录算法运行时间
                f.write(f"梯度足够小，停止迭代\n")
                f.write(f"L_BFGS法迭代结束, 运行时间：{ela}, 迭代{N}轮，函数调用次数{fun_iter}, X*为{X_new}, 当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
                f.flush()  # 强制刷新输出
                return  # 如果梯度足够小，停止迭代

        # 如果达到最大迭代次数
        ela = time.time() - tic  # 记录算法运行时间
        f.write(f"L_BFGS法迭代已达最大迭代轮次,结束, 运行时间：{ela}, 迭代{N}轮，函数调用次数{fun_iter}, X*为{X_new}, 当前|g|为{round(np.linalg.norm(G(X)), 7)},当前函数值为{round(F(X_new), 8)}\n")
        f.flush()  # 强制刷新输出