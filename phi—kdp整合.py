import cinrad
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ================= 1. 读取雷达整圈数据 =================
f = cinrad.io.StandardData(
    r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2")

# 获取仰角层(参数里的 2, 140 根据你实际情况保留)
ds_phi = f.get_data(2, 140, "PHI")
ds_kdp = f.get_data(2, 140, "KDP")

# 提取所有的真实方位角并转为度数
az_rads = ds_phi["azimuth"].values
az_degs = np.rad2deg(az_rads) % 360

# ================= 2. 根据方位角精准筛选径向 =================
# 找到方位角在 145° 到 190° 之间的所有径向编号
valid_indices = np.where((az_degs >= 145) & (az_degs <= 190))[0]

if len(valid_indices) == 0:
    raise ValueError("未找到 145° 到 190° 范围内的径向数据，请检查该仰角是否存在此方位数据！")

# 满足要求：输出一头一尾的径向方位角以及对应的径向编号
selected_az = az_degs[valid_indices]
idx_near_145 = valid_indices[np.argmin(np.abs(selected_az - 145))]
idx_near_190 = valid_indices[np.argmin(np.abs(selected_az - 190))]

print(f"=== 范围筛选结果 ===")
print(f"起始 (最接近 145°): 径向编号 = {idx_near_145}, 实际方位角 = {az_degs[idx_near_145]:.3f}°")
print(f"结束 (最接近 190°): 径向编号 = {idx_near_190}, 实际方位角 = {az_degs[idx_near_190]:.3f}°")
# ----------------------

print(f"共计找到 {len(valid_indices)} 条径向参与计算...\n")

# ================= 3. 逐径向执行算法并收集数据 =================
all_radials_data = []
grad_out_list = []  # 用于保存跨径向隔离的差分(梯度)数据

for az_idx in valid_indices:
    az_current_deg = az_degs[az_idx]

    # 切片单条径向
    radial_phi = ds_phi.isel(azimuth=az_idx)
    radial_kdp = ds_kdp.isel(azimuth=az_idx)

    var_phi = list(radial_phi.data_vars)[0]
    var_kdp = list(radial_kdp.data_vars)[0]

    dat_phi = radial_phi[[var_phi]].to_dataframe().reset_index()
    dat_kdp = radial_kdp[[var_kdp]].to_dataframe().reset_index()

    # 对齐距离坐标(保留NaN占位，保证0.075步长不变)
    dat = pd.merge(dat_phi[["distance", var_phi]], dat_kdp[["distance", var_kdp]], on="distance", how="outer")
    dat = dat.sort_values("distance")
    dat = dat.rename(columns={"distance": "dis", var_phi: "phdp", var_kdp: "raw_kdp"})
    dat['dis'] = np.round(dat['dis'], 3)

    # 标记当前方位角（方便后续排查）
    dat["azimuth_deg"] = az_current_deg

    # --- 算法 1：Isotonic Regression ---
    dat["reg_phi"] = np.nan
    valid_mask = dat["phdp"].notna() & (dat["phdp"] >= -900)
    x = dat.loc[valid_mask, "dis"].to_numpy()
    y = dat.loc[valid_mask, "phdp"].to_numpy()

    if len(x) > 0:
        ir = IsotonicRegression(out_of_bounds="clip")
        dat.loc[valid_mask, "reg_phi"] = ir.fit_transform(x, y)

    # --- 算法 2 & 3：计算 reg_kdp 和 kdp_lsf ---
    r = dat["dis"].values
    reg_phi = dat["reg_phi"].values

    kdp_reg = np.full(len(r), np.nan)
    kdp_lsf = np.full(len(r), np.nan)

    val_mask = ~np.isnan(reg_phi)
    val_indices = np.where(val_mask)[0]

    if len(val_indices) > 0:
        r_valid = r[val_mask]
        reg_phi_valid = reg_phi[val_mask]

        # 梯度求导 (reg_kdp)
        kdp_reg[val_mask] = 0.5 * np.gradient(reg_phi_valid, r_valid)

        # 最小二乘拟合 (kdp_lsf)
        window = 15
        half_w = window // 2
        if len(val_indices) > window:
            for i in range(half_w, len(val_indices) - half_w):
                idx = val_indices[i]
                r_win = r[val_indices[i - half_w: i + half_w + 1]]
                phi_win = reg_phi[val_indices[i - half_w: i + half_w + 1]]

                A = np.vstack([r_win, np.ones(len(r_win))]).T
                a, _ = np.linalg.lstsq(A, phi_win, rcond=None)[0]

                kdp_lsf[idx] = max(0.0, 0.5 * a)  # 截断负值

    dat["reg_kdp"] = kdp_reg
    dat["kdp_lsf"] = kdp_lsf

    # 存入总列表
    all_radials_data.append(dat)

    # --- 单径向差分数据生成 (防止跨径向相减产生超大异常值) ---
    cols_to_diff = ["raw_kdp", "reg_kdp", "kdp_lsf"]
    g_out = pd.DataFrame()
    g_out["dis_mid"] = (r[:-1] + r[1:]) / 2
    g_out["azimuth_deg"] = az_current_deg
    for c in cols_to_diff:
        g_out[c + "_abs_diff"] = np.abs(np.diff(dat[c].to_numpy(dtype=float)))
    grad_out_list.append(g_out)

# ================= 4. 汇总数据与导出txt =================
final_df = pd.concat(all_radials_data, ignore_index=True)
final_df.to_csv("kdp_lsf.txt", index=False, na_rep='NaN')
print("已成功导出汇总数据: kdp_lsf.txt")

grad_out = pd.concat(grad_out_list, ignore_index=True)
grad_out.to_csv("kdp_abs_diff_wide.csv", index=False, encoding="utf-8-sig", na_rep='NaN')

# ================= 5. 直方图占比统计算法 (4个边界中心) =================
cols = ["raw_kdp", "reg_kdp", "kdp_lsf"]
hist_out = pd.DataFrame()
hist_out["bin_center"] = [0, 1, 3, 5]

for col in cols:
    x = final_df[col].to_numpy(dtype=float)
    x_valid = x[~np.isnan(x)]  # 过滤掉占位的NaN
    total = len(x_valid)

    if total > 0:
        # 手动精准划分区间，溢出部分自动并入两端
        c1 = np.sum(x_valid < 0)  # <= -1
        c2 = np.sum((x_valid >= 0) & (x_valid <= 1))  # 1
        c3 = np.sum((x_valid > 1) & (x_valid <= 3))  # 3
        c4 = np.sum(x_valid > 3)  # > 5 都合并算在 5 这个节点里

        counts = [c1, c2, c3, c4]
        freqs = [c / total for c in counts]
    else:
        counts = [0, 0, 0, 0]
        freqs = [0.0, 0.0, 0.0, 0.0]

    hist_out[col + "_count"] = counts
    hist_out[col + "_frequency"] = freqs

hist_out.to_csv("kdp_histogram_wide0.csv", index=False, encoding="utf-8-sig")
print("已成功导出占比直方图数据: kdp_histogram_wide.csv")
print("已成功导出差分(梯度)数据: kdp_abs_diff_wide.csv")