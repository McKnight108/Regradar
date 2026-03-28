import cinrad
import numpy as np
import pandas as pd

# =========================
# 1. 文件路径
# =========================
file_path = r"D:\Chan\Documents\radar_raw_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"

# =========================
# 2. 可调参数
# =========================
max_range_km = 230          # 读取半径
phi_min_points = 100        # PHI有效点数要超过这个值
phi_max_threshold = 45      # PHI最大值阈值
rho_threshold = 0.85        # RHO阈值
rho_ratio_threshold = 0.85  # “绝大部分”这里先定义为 >=85%
output_csv = r"D:\Chan\Documents\radar_raw_data\ZA003\qualified_radials.csv"

# =========================
# 3. 读取数据
# =========================
f = cinrad.io.StandardData(file_path)

phi_tilts = set(map(int, f.available_tilt("PHI")))
rho_tilts = set(map(int, f.available_tilt("RHO")))
common_tilts = sorted(phi_tilts & rho_tilts)

print("PHI可用仰角:", sorted(phi_tilts))
print("RHO可用仰角:", sorted(rho_tilts))
print("共同可用仰角:", common_tilts)

results = []

# =========================
# 4. 逐仰角、逐径向筛选
# =========================
for tilt in common_tilts:
    print(f"\n正在处理 tilt = {tilt}, 仰角 = {float(f.el[tilt]):.3f} deg")

    ds_phi = f.get_data(tilt, max_range_km, "PHI")
    ds_rho = f.get_data(tilt, max_range_km, "RHO")

    # 提取二维数组: (azimuth, distance)
    phi_data = ds_phi["PHI"].values
    rho_data = ds_rho["RHO"].values

    azimuth_rad = ds_phi["azimuth"].values
    distance_km = ds_phi["distance"].values

    n_azi = min(phi_data.shape[0], rho_data.shape[0])

    for i in range(n_azi):
        phi_line = phi_data[i, :]
        rho_line = rho_data[i, :]

        # ---- PHI有效值 ----
        # 既防 NaN，也防异常填充值
        phi_valid_mask = np.isfinite(phi_line) & (phi_line > -900)
        phi_valid = phi_line[phi_valid_mask]

        phi_valid_count = phi_valid.size
        if phi_valid_count <= phi_min_points:
            continue

        phi_max = np.nanmax(phi_valid)
        if phi_max <= phi_max_threshold:
            continue

        # ---- RHO有效值 ----
        rho_valid_mask = np.isfinite(rho_line) & (rho_line > -900)
        rho_valid = rho_line[rho_valid_mask]

        if rho_valid.size == 0:
            continue

        # “绝大部分距离都大于0.85”
        rho_good_ratio = np.sum(rho_valid > rho_threshold) / rho_valid.size

        if rho_good_ratio < rho_ratio_threshold:
            continue

        # 记录结果
        results.append({
            "tilt_index": tilt,
            "elevation_deg": float(f.el[tilt]),
            "azimuth_index": i,
            "azimuth_deg": float(np.rad2deg(azimuth_rad[i]) % 360),
            "phi_valid_points": int(phi_valid_count),
            "phi_max": float(phi_max),
            "rho_valid_points": int(rho_valid.size),
            "rho_gt_0.85_ratio": float(rho_good_ratio)
        })

# =========================
# 5. 输出结果
# =========================
df = pd.DataFrame(results)

if len(df) == 0:
    print("\n没有找到符合条件的径向。")
else:
    df = df.sort_values(
        by=["tilt_index", "azimuth_deg"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print("\n符合条件的径向如下：")
    print(df)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_csv}")