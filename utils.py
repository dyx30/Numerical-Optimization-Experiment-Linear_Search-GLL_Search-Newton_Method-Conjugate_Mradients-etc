import math 
import copy
import numpy as np

def is_pos_def(A):
    """
    判断对称矩阵是否正定
    """
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def modified_Cholesky(G, u=1e-10):
    """修正Cholesky分解
    参数:
        G: 用于分解的二维矩阵
        u: 机器精度
    """

    # 步骤1：初始化
    G = np.array(G)
    gamma = 0 # 对角元最大元素
    ksai = 0  # 非对角元最大元素
    n = len(G)
    for i in range(n):
        for j in range(n):
            if i == j:
                gamma = max(gamma, abs(G[i][i])) 
            else:
                ksai = max(ksai, abs(G[i][j]))
    beta2 = max(gamma, ksai / math.sqrt(n ** 2 - 1), u)
    delta = u * max(gamma + ksai, 1)
    
    assert delta > 0 , "must have delta > 0" 
    L = np.eye(n, dtype=float)
    D = np.zeros((n,n), dtype=float)
    C = np.zeros((n,n), dtype=float)
    #按列计算
    j = 1 #表示当前计算的列的indx
    while j <= n:
        # 步骤2：计算dj_prime
        dj_prime = max(delta, abs(G[j - 1][j - 1] - sum((C[j - 1][r - 1] ** 2 / (D[r - 1][r - 1]) for r in range(1, j))) ) )  
        # 步骤3：计算Cij
        for i in range(j + 1, n + 1):
            C[i - 1][j - 1] = G[i - 1][j - 1] - sum(( L[j - 1][r - 1] * C[i - 1][r - 1] for r in range(1, j)))
        # 步骤4：计算theta_j
        theta_j = 0
        if j < n:
            theta_j = max(( abs(C[i - 1][j - 1]) for i in range(j + 1, n + 1)))
        # 步骤5：计算d_j
        D[j - 1][j - 1] = max(dj_prime, theta_j ** 2 / beta2)
        # 步骤6：计算l_ij
        for i in range(j + 1, n + 1):
            L[i - 1][j - 1] = C[i - 1][j - 1] / D[j - 1][j - 1]
        j += 1
    LT = copy.deepcopy(L).T
    C = np.dot(L, D)
    return np.dot(C, LT)
