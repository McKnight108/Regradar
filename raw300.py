import cinrad
import numpy as np
import pandas as pd

# 1. 读取雷达文件
f = cinrad.io.StandardData(r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2")

# 2. 读取 PHI
ds = f.get_data(2, 140, "PHI")

# 3. 直接选择第147号径向（不插值）
az_idx = 147
radial = ds.isel(azimuth=az_idx)

# 4. 看一下变量名
print(radial)
print("data_vars =", list(radial.data_vars))

# 5. 自动取出唯一的数据变量
var_name = list(radial.data_vars)[0]
print("使用变量名:", var_name)

# 顺便打印这一条径向对应的真实方位角（原始数据是弧度）
azi_rad = ds["azimuth"].values[az_idx]
azi_deg = np.rad2deg(azi_rad) % 360
print(f"选中的径向编号 = {az_idx}, 方位角 = {azi_deg:.3f}°")

# 6. 转成 DataFrame
dat = radial[[var_name]].to_dataframe().reset_index()

# 7. 只保留距离和 PHI 值两列
dat = dat[["distance", var_name]].rename(columns={
    "distance": "dis",
    var_name: "phdp"
})

# 8. 去掉缺测值
dat = dat.dropna()

# 9. 保存成 txt/csv
save_path = "datatxt/phdp_300.txt"
dat.to_csv(save_path, index=False)

#print("已保存:", save_path)
print(dat.head())
print(dat.tail())
print(f"{azi_deg:.3f}°")