import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# https://gemini.google.com/app/af79eb623033086a
# 设置随机种子以保证结果可复现
np.random.seed(42)


# ==========================================
# 1. 模拟数据生成 (Synthetic Data Generation)
# ==========================================
def generate_synthetic_data(n_units=10, n_periods=20, treatment_period=12):
    """
    生成包含 1 个处理对象和 n_units-1 个控制对象的面板数据
    """
    time = np.arange(1, n_periods + 1)

    # 不可观测的潜在因子 (Unobserved factor)
    factor = np.sin(time / 2)

    # 模拟特征 (Covariates): 人均收入、教育水平
    covariate1 = np.random.uniform(20, 50, size=n_units)
    covariate2 = np.random.uniform(5, 15, size=n_units)

    # 生成结果变量 Y (基础趋势 + 因子影响 + 协变量影响 + 随机噪声)
    Y = np.zeros((n_units, n_periods))
    for i in range(n_units):
        base_level = 10 + i * 2
        Y[i, :] = (base_level
                   + 2.0 * factor
                   + 0.5 * covariate1[i]
                   + 1.2 * covariate2[i]
                   + np.random.normal(0, 0.5, size=n_periods))

    # 为真实处理对象 (Unit 0) 在政策发生后增加真实的因果效应 (Treatment Effect = +8)
    treatment_effect = 8.0
    Y[0, treatment_period:] += treatment_effect

    return Y, covariate1, covariate2


# ==========================================
# 2. 合成控制法核心算法类 (SCM Implementation)
# ==========================================
class SyntheticControl:
    def __init__(self):
        self.W = None  # 求解出的控制组权重
        self.V = None  # 求解出的预测变量重要性权重

    def _loss_W(self, W, X1, X0, V):
        """
        第一层优化目标函数：求解给定 V 下的权重 W
        Minimize: (X1 - X0 * W)' * V * (X1 - X0 * W)
        """
        diff = X1 - np.dot(X0, W)
        return np.dot(np.dot(diff.T, V), diff)

    def _fit_W(self, X1, X0, V):
        """
        在约束条件 (sum(W) = 1, W >= 0) 下优化 W
        """
        n_controls = X0.shape[1]
        w0 = np.full(n_controls, 1.0 / n_controls)  # 初始权重均分

        # 约束 1: 权重和为 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # 约束 2: 权重非负 [0, 1]
        bounds = [(0.0, 1.0) for _ in range(n_controls)]

        res = minimize(
            self._loss_W,
            w0,
            args=(X1, X0, V),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return res.x

    def _loss_V(self, v_diag, X1, X0, Y1_pre, Y0_pre):
        """
        第二层优化目标函数：求解最优特征权重 V，使得政策前结果变量的 MSPE 最小
        """
        # 将传入的一维向量转换为对角矩阵 V
        V = np.diag(v_diag)

        # 步骤 1: 根据当前的 V 计算最优 W
        W_opt = self._fit_W(X1, X0, V)

        # 步骤 2: 计算政策前结果变量的均方预测误差 (MSPE)
        mspe = np.mean((Y1_pre - np.dot(Y0_pre, W_opt)) ** 2)
        return mspe

    def fit(self, X1, X0, Y1_pre, Y0_pre):
        """
        模型拟合入口
        X1: 处理对象政策前特征向量 (n_features,)
        X0: 控制池政策前特征矩阵 (n_features, n_controls)
        Y1_pre: 处理对象政策前时间序列 (T0,)
        Y0_pre: 控制池政策前时间序列 (T0, n_controls)
        """
        n_features = len(X1)
        v0 = np.full(n_features, 1.0 / n_features)

        # V 对角线元素的非负约束
        bounds_v = [(0.0, None) for _ in range(n_features)]

        # 优化求解 V
        res_v = minimize(
            self._loss_V,
            v0,
            args=(X1, X0, Y1_pre, Y0_pre),
            method='L-BFGS-B',
            bounds=bounds_v
        )

        # 归一化 V 对角线
        v_opt_diag = res_v.x / np.sum(res_v.x)
        self.V = np.diag(v_opt_diag)

        # 最终计算最优权重 W
        self.W = self._fit_W(X1, X0, self.V)
        return self


# ==========================================
# 3. 运行完整数据分析流程
# ==========================================
if __name__ == "__main__":
    n_units = 6  # 1 个处理对象 + 5 个控制对象
    n_periods = 20  # 共 20 个时间点
    T0 = 12  # 第 12 期实施政策 (政策前 1-12，政策后 13-20)

    # 1. 获取模拟数据
    Y, cov1, cov2 = generate_synthetic_data(n_units=n_units, n_periods=n_periods, treatment_period=T0)

    # 2. 构造特征矩阵 (X) 与 历史结果序列 (Y_pre)
    # 特征：取政策前的协变量均值 + 政策前部分年份的历史结果
    X1 = np.array([cov1[0], cov2[0], Y[0, 2], Y[0, 8]])
    X0 = np.array([
        cov1[1:],
        cov2[1:],
        Y[1:, 2],
        Y[1:, 8]
    ])

    # 标准化特征 (Standardization)
    scaler_mean = np.mean(X0, axis=1)
    scaler_std = np.std(X0, axis=1)

    X1_scaled = (X1 - scaler_mean) / scaler_std
    X0_scaled = ((X0.T - scaler_mean) / scaler_std).T

    Y1_pre = Y[0, :T0]
    Y0_pre = Y[1:, :T0].T  # 转置为 (T0, n_controls)

    # 3. 求解合成控制模型
    model = SyntheticControl()
    model.fit(X1_scaled, X0_scaled, Y1_pre, Y0_pre)

    # 4. 计算合成结果与因果效应
    Y0_all = Y[1:, :].T  # 所有时期控制池数据 (n_periods, n_controls)
    Y1_synthetic = np.dot(Y0_all, model.W)  # 合成对象的全期轨迹
    Y1_actual = Y[0, :]  # 真实对象的全期轨迹

    effect = Y1_actual[T0:] - Y1_synthetic[T0:]  # 政策后的纯因果效应

    # 5. 打印结果
    print("=" * 40)
    print("【合成控制法优化求解完成】")
    print("=" * 40)
    print("分配给各控制实体的权重 W:")
    for idx, w in enumerate(model.W):
        print(f"  - 控制实体 {idx + 1}: {w:.4f}")

    print("\n政策实施后估计出的平均因果效应 (ATT):", np.mean(effect))
    print("实际设定的地面真值因果效应 (Ground Truth): 8.0")

    # 6. 可视化绘制结果图表
    plt.figure(figsize=(10, 5))
    time_steps = np.arange(1, n_periods + 1)

    plt.plot(time_steps, Y1_actual, label='Real Unit (Treated)', color='blue', linewidth=2.5)
    plt.plot(time_steps, Y1_synthetic, label='Synthetic Unit (Control)', color='red', linestyle='--', linewidth=2.5)

    # 标记政策实施节点
    plt.axvline(x=T0 + 0.5, color='black', linestyle=':', label='Treatment Period')

    plt.title("Synthetic Control Method (SCM) - Cause & Effect Estimation")
    plt.xlabel("Time Period")
    plt.ylabel("Outcome Variable (Y)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()