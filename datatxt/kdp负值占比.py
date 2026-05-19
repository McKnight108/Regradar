import pandas as pd
import numpy as np

input_path = r"kdp_lsf_300_15.txt"

df = pd.read_csv(input_path)

# 修正变量名
cols = ["raw_kdp", "reg_kdp", "kdp_lsf"]

# ================= 1. 新的直方图算法 (两端截断归并) =================
hist_out = pd.DataFrame()
# 强制设定您想要的 X 轴节点
hist_out["bin_center"] = [-1, 1, 3, 5]

for col in cols:
    x = df[col].to_numpy(dtype=float)
    x_valid = x[~np.isnan(x)]  # 过滤盲区空值
    total = len(x_valid)

    if total > 0:
        # 手动精确划分区间，确保 <=-1 和 >5 的数据被正确归并
        c1 = np.sum(x_valid <= -1)  # <= -1 全算到 -1
        c2 = np.sum((x_valid > -1) & (x_valid <= 1))  # 1 正常分
        c3 = np.sum((x_valid > 1) & (x_valid <= 3))  # 3 正常分
        c4 = np.sum(x_valid > 3)  # > 5 全算到 5 (即包含正常3~5以及>5的所有值)

        counts = [c1, c2, c3, c4]
        freqs = [c / total for c in counts]
    else:
        counts = [0, 0, 0, 0]
        freqs = [0.0, 0.0, 0.0, 0.0]

    hist_out[col + "_count"] = counts
    hist_out[col + "_frequency"] = freqs

save_hist = "kdp_histogram_wide.csv"
hist_out.to_csv(save_hist, index=False, encoding="utf-8-sig")

# ================= 2. 梯度(差分)数据 (保持原样) =================
grad_out = pd.DataFrame()
grad_out["dis_mid"] = (df["dis"].to_numpy(dtype=float)[:-1] + df["dis"].to_numpy(dtype=float)[1:]) / 2

for col in cols:
    x = df[col].to_numpy(dtype=float)
    grad_out[col + "_abs_diff"] = np.abs(np.diff(x))

save_grad = "kdp_abs_diff_wide.csv"
grad_out.to_csv(save_grad, index=False, encoding="utf-8-sig")

print(f"已生成 直方图数据: {save_hist}")
print(f"已生成 梯度(差分)数据: {save_grad}")