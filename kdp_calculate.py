import pandas as pd
import numpy as np

# 1. 读取回归后的数据
dat = pd.read_csv("./datatxt/phdp_300_reg.txt")

# 2. 取出变量
r = dat["dis"].to_numpy()
phi = dat["phi"].to_numpy()
reg_phi = dat["reg_phi"].to_numpy()

# 3. 计算差分（相邻点）
dr = np.diff(r)

# 原始KDP
dphi = np.diff(phi)
kdp = 0.5 * dphi / dr

# 回归后KDP
dphi_reg = np.diff(reg_phi)
kdp_reg = 0.5 * dphi_reg / dr

# 4. 对应距离（取中点更合理）
r_mid = (r[:-1] + r[1:]) / 2

# 5. 组织输出
out = pd.DataFrame({
    "dis": r_mid,
    "phi": phi[:-1],
    "reg_phi": reg_phi[:-1],
    "kdp": kdp,
    "reg_kdp": kdp_reg
})

# 6. 保存
save_path = "datatxt/kdp_300.txt"
out.to_csv(save_path, index=False)

print("已保存:", save_path)
print(out.head())
print(out.tail())