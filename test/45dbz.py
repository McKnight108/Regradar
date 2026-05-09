import cinrad
import numpy as np
import pandas as pd

# =========================
# 1. 文件路径
# =========================
file_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"

# =========================
# 2. 可调参数
# =========================
max_range_km = 120
ref_threshold = 45
ref_min_points = 10
rho_threshold = 0.9
rho_ratio_threshold = 0.85

# =========================
# 3. 读取数据
# =========================
f = cinrad.io.StandardData(file_path)

ref_tilts = set(map(int, f.available_tilt("REF")))
rho_tilts = set(map(int, f.available_tilt("RHO")))
common_tilts = sorted(ref_tilts & rho_tilts)

print("REF可用仰角:", sorted(ref_tilts))
print("RHO可用仰角:", sorted(rho_tilts))
print("共同可用仰角:", common_tilts)

results = []

# =========================
# 4. 逐仰角、逐径向筛选
# =========================
for tilt in common_tilts:
    print(f"\n正在处理 tilt = {tilt}, 仰角 = {float(f.el[tilt]):.3f} deg")

    ds_ref = f.get_data(tilt, max_range_km, "REF")
    ds_rho = f.get_data(tilt, max_range_km, "RHO")

    ref_data = ds_ref["REF"].values
    rho_data = ds_rho["RHO"].values
    azimuth_data = ds_ref["azimuth"].values

    n_azi = min(ref_data.shape[0], rho_data.shape[0])

    for i in range(n_azi):
        ref_line = ref_data[i, :]
        rho_line = rho_data[i, :]

        # REF有效值
        ref_valid_mask = np.isfinite(ref_line) & (ref_line > -900)
        ref_valid = ref_line[ref_valid_mask]

        if ref_valid.size == 0:
            continue

        # 核心条件：REF > 45 的点数必须 > 10
        ref_gt_threshold_count = np.sum(ref_valid > ref_threshold)
        if ref_gt_threshold_count <= ref_min_points:
            continue

        # RHO有效值
        rho_valid_mask = np.isfinite(rho_line) & (rho_line > -900)
        rho_valid = rho_line[rho_valid_mask]

        if rho_valid.size == 0:
            continue

        # RHO条件：有效点中大于0.85的比例达到阈值
        rho_good_ratio = np.sum(rho_valid > rho_threshold) / rho_valid.size
        if rho_good_ratio < rho_ratio_threshold:
            continue

        # 方位角转为度
        azimuth_deg = float(np.rad2deg(azimuth_data[i]) % 360)

        results.append({
            "tilt_index": tilt,
            "elevation_deg": float(f.el[tilt]),
            "azimuth_index": i,
            "azimuth_deg": azimuth_deg,
            "ref_gt_45_points": int(ref_gt_threshold_count),
            "ref_max": float(np.nanmax(ref_valid)),
            "rho_valid_points": int(rho_valid.size),
            "rho_gt_0.85_ratio": float(rho_good_ratio)
        })

# =========================
# 5. 输出结果
# =========================
df = pd.DataFrame(results)

if df.empty:
    print("\n没有找到符合条件的径向。")
else:
    df = df.sort_values(
        by=["tilt_index", "azimuth_deg"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print("\n符合条件的径向如下：")
    print(df.to_string(index=False))

# =========================
# 5. 输出结果
# =========================
df = pd.DataFrame(results)

if df.empty:
    print("\n没有找到符合条件的径向。")
else:
    df = df.sort_values(
        by=["tilt_index", "azimuth_deg"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print("\n符合条件的径向如下：")
    print(df.to_string(index=False))

    # 新增：统计总数
    print(f"\n总共找到 {len(df)} 条符合条件的径向")