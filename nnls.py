# coding: utf-8
import numpy as np 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split

def nnls(A, y, eps=2.22e-16):
    """
    A python implementation of NNLS algorithm, seen on https://en.wikipedia.org/wiki/Non-negative_least_squares
    A: 自变量, shape is [m, n]
    y: 因变量, shape is [m,]
    eps: 误差项
    
    return x is regression coef, subject x >= 0
    """
    # initialize 
    m, n = A.shape
    P = []
    R = list(range(n))
    x = np.zeros(n)
    w = np.dot(A.T, y - np.dot(A, x))
    tol = 10 * eps * np.linalg.norm(A, 1) * (max(A.shape) + 1)
    
    # loop 
    while R and np.max(w) > tol:
        j = np.argmax(w) 
        P.append(j)
        R.remove(j) 

        AP = np.zeros(A.shape)
        AP[:, P] = A[:, P]
        s = np.dot(np.linalg.pinv(AP), y) 
        s[R] = 0

        while np.min(s) < 0:
            alpha = min([x[i] / (x[i] - s[i]) for i in P if s[i] <= 0])
            x = x + alpha * (s - x)
            idx = [j for j in P if x[j] <= 0]
            if idx:
                R.append(*j)
                P.remove(j)
            AP = np.zeros(A.shape)
            AP[:, P] = A[:, P]
            s = np.dot(np.linalg.pinv(AP), y) 
            s[R] = 0
        x = s 
        w = np.dot(A.T, y - np.dot(A, x))
    return np.array(x) 

if __name__ == "__main__":
    np.random.seed(42)
    # 随机产生 200 X 50 的矩阵
    n_samples, n_features = 200, 50
    X = np.random.randn(n_samples, n_features)  
    true_coef = 3 * np.random.randn(n_features)
    # 阈值系数使它们非负，数据情景设定为非负
    true_coef[true_coef < 0] = 0
    y = np.dot(X, true_coef)
    # 施加随机扰动
    y += 5 * np.random.normal(size=(n_samples,))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # 通过实现的nnls求解coef
    coef0 = nnls(X_train, y_train)

    # 通过sklearn求解coef
    from sklearn.linear_model import LinearRegression 
    reg_nnls = LinearRegression(positive=True)
    reg_nnls.fit(X_train, y_train)
    y_pred_nnls = reg_nnls.predict(X_test)
    coef1 = reg_nnls.coef_

    # 画图
    fig, ax = plt.subplots()
    # 画轴,画出两种方法的回归系数W1,W2,.....Wp
    ax.plot(coef0, coef1, linewidth=0, marker=".")
    # 画图，以数据点为图的四周的极限
    low_x, high_x = ax.get_xlim()
    low_y, high_y = ax.get_ylim()
    low = max(low_x, low_y)
    high = min(high_x,high_y)
    # 同一张图上再画一条虚线，指定虚线，颜色。从（low,low）到（high,high）画图
    ax.plot([low, high], [low, high], ls="--", c=".3", alpha=.5)
    # 写坐标轴标签和设置字体
    ax.set_xlabel("self nnls regression coefficients", fontweight="bold")
    ax.set_ylabel("NNLS regression coefficients", fontweight="bold")
    plt.savefig('cmp.png')
    

