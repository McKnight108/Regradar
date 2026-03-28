import pandas as pd
import numpy as np

# 1. 读取回归结果文件
dat = pd.read_csv("./datatxt/phdp_300_reg.txt", index_col=False)

# 2. 检查并保留需要的列
dat = dat[["dis", "phi", "reg_phi"]].copy()

# 3. 对 reg_phi 做五点滑动平均
dat["smt_phi"] = dat["reg_phi"].rolling(
    window=5,
    center=True,
    min_periods=1
).mean()

# 4. 计算 smt_kdp
r = dat["dis"].to_numpy()
phi_smt = dat["smt_phi"].to_numpy()

dr = np.diff(r)
dphi = np.diff(phi_smt)

smt_kdp = 0.5 * dphi / dr

# 5. 为了和原表长度一致，在最后补一个 NaN
dat["smt_kdp"] = np.append(smt_kdp, np.nan)

# 6. 只保留你要的四列
out = dat[["dis", "phi", "reg_phi", "smt_kdp"]]

# 7. 导出
save_path = "./kdp_300_smt.txt"
out.to_csv(save_path, index=False, float_format="%.6f")

print("已保存:", save_path)
print(out.head(10))
print(out.tail(10))