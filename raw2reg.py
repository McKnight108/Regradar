import pandas as pd
from sklearn.isotonic import IsotonicRegression

# 1. 读取原始phi数据
dat = pd.read_csv("datatxt/phdp_300.txt", index_col=False)

# 2. 去掉缺测或异常值
dat = dat.dropna(subset=["dis", "phdp"])
dat = dat[dat["phdp"] >= -900].copy()

# 3. 取出距离和原始phi
x = dat["dis"].to_numpy()
y = dat["phdp"].to_numpy()

# 4. 单调回归
ir = IsotonicRegression(out_of_bounds="clip")
y_reg = ir.fit_transform(x, y)

# 5. 保存结果
out = pd.DataFrame({
    "dis": x,
    "phi": y,
    "reg_phi": y_reg
})

save_path = "datatxt/phdp_300_reg.txt"
out.to_csv(save_path, index=False)

print("已保存:", save_path)
print(out.head())
print(out.tail())