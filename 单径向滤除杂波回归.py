import os
import cinrad
import numpy as np
from sklearn.isotonic import IsotonicRegression

file_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
radial_idx = 147

f = cinrad.io.StandardData(file_path)

ds_ref = f.get_data(2, 140, "REF")
ds_rho = f.get_data(2, 140, "RHO")
ds_zdr = f.get_data(2, 140, "ZDR")
ds_phi = f.get_data(2, 140, "PHI")

zh = ds_ref["REF"].values.astype(float)
rho = ds_rho["RHO"].values.astype(float)
zdr = ds_zdr["ZDR"].values.astype(float)
phi = ds_phi["PHI"].values.astype(float)

distance = ds_phi["distance"].values.astype(float)
if distance.ndim == 1:
    distance_2d = np.broadcast_to(distance, phi.shape).astype(float)
else:
    distance_2d = distance.astype(float)

zh_ray = zh[radial_idx, :].copy()
rho_ray = rho[radial_idx, :].copy()
zdr_ray = zdr[radial_idx, :].copy()
phi_ray = phi[radial_idx, :].copy()
dist_ray = distance_2d[radial_idx, :].copy()

for i in range(zdr_ray.shape[0]):
    if not np.isfinite(zdr_ray[i]) and i > 0:
        zdr_ray[i] = zdr_ray[i - 1]

for i in range(phi_ray.shape[0]):
    if not np.isfinite(phi_ray[i]) and i > 0:
        phi_ray[i] = phi_ray[i - 1]

window = 7
pad = window // 2

zdr_pad = np.pad(zdr_ray, (pad, pad), mode="constant", constant_values=np.nan)
phi_pad = np.pad(phi_ray, (pad, pad), mode="constant", constant_values=np.nan)

zdr_win = np.lib.stride_tricks.sliding_window_view(zdr_pad, window_shape=window)
phi_win = np.lib.stride_tricks.sliding_window_view(phi_pad, window_shape=window)

valid_zdr_count = np.sum(np.isfinite(zdr_win), axis=-1)
valid_phi_count = np.sum(np.isfinite(phi_win), axis=-1)

zdr_sum = np.nansum(zdr_win, axis=-1)
phi_sum = np.nansum(phi_win, axis=-1)

zdr_mean = np.full(zdr_ray.shape, np.nan, dtype=float)
phi_mean = np.full(phi_ray.shape, np.nan, dtype=float)

zdr_mean[valid_zdr_count > 0] = zdr_sum[valid_zdr_count > 0] / valid_zdr_count[valid_zdr_count > 0]
phi_mean[valid_phi_count > 0] = phi_sum[valid_phi_count > 0] / valid_phi_count[valid_phi_count > 0]

for i in range(zdr_mean.shape[0]):
    if not np.isfinite(zdr_mean[i]) and i > 0:
        zdr_mean[i] = zdr_mean[i - 1]

for i in range(phi_mean.shape[0]):
    if not np.isfinite(phi_mean[i]) and i > 0:
        phi_mean[i] = phi_mean[i - 1]

sd_zdr = np.full(zdr_ray.shape, np.nan, dtype=float)
sd_phi = np.full(phi_ray.shape, np.nan, dtype=float)

zdr_var_sum = np.nansum((zdr_win - zdr_mean[:, None]) ** 2, axis=-1)
phi_var_sum = np.nansum((phi_win - phi_mean[:, None]) ** 2, axis=-1)

sd_zdr[valid_zdr_count > 0] = np.sqrt(zdr_var_sum[valid_zdr_count > 0] / valid_zdr_count[valid_zdr_count > 0])
sd_phi[valid_phi_count > 0] = np.sqrt(phi_var_sum[valid_phi_count > 0] / valid_phi_count[valid_phi_count > 0])

for i in range(sd_zdr.shape[0]):
    if not np.isfinite(sd_zdr[i]) and i > 0:
        sd_zdr[i] = sd_zdr[i - 1]

for i in range(sd_phi.shape[0]):
    if not np.isfinite(sd_phi[i]) and i > 0:
        sd_phi[i] = sd_phi[i - 1]

non_meteo_mask = (rho_ray < 0.9) | (sd_zdr > 2.0) | (sd_phi > 25.0)

zh_qc = zh_ray.copy()
rho_qc = rho_ray.copy()
zdr_qc = zdr_ray.copy()
phi_qc = phi_ray.copy()

zh_qc[non_meteo_mask] = np.nan
rho_qc[non_meteo_mask] = np.nan
zdr_qc[non_meteo_mask] = np.nan
phi_qc[non_meteo_mask] = np.nan

phi_reg = np.full_like(phi_qc, np.nan, dtype=float)

valid = np.isfinite(dist_ray) & np.isfinite(phi_qc)
if np.sum(valid) >= 2:
    xv = dist_ray[valid]
    yv = phi_qc[valid]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    y_iso = ir.fit_transform(xv, yv)
    valid_idx = np.where(valid)[0][order]
    phi_reg[valid_idx] = y_iso

gate_diffs = np.diff(dist_ray)
gate_diffs = gate_diffs[np.isfinite(gate_diffs) & (gate_diffs > 0)]
if gate_diffs.size > 0:
    gate_len = np.nanmedian(gate_diffs)
else:
    gate_len = 0.075

kdp_lsf = np.full(phi_reg.shape, 0.0, dtype=float)

for irng in range(phi_reg.shape[0]):
    if not np.isfinite(phi_reg[irng]):
        kdp_lsf[irng] = 0.0
        continue
    if not np.isfinite(zh_qc[irng]):
        kdp_lsf[irng] = 0.0
        continue
    if not np.isfinite(dist_ray[irng]):
        kdp_lsf[irng] = 0.0
        continue

    if zh_qc[irng] > 45:
        half_win = max(1, int(round(0.75 / gate_len)))
    elif zh_qc[irng] > 35:
        half_win = max(1, int(round(1.5 / gate_len)))
    else:
        half_win = max(1, int(round(2.25 / gate_len)))

    i0 = max(0, irng - half_win)
    i1 = min(phi_reg.shape[0], irng + half_win + 1)

    xw = dist_ray[i0:i1]
    yw = phi_reg[i0:i1]

    ok = np.isfinite(xw) & np.isfinite(yw)
    if np.sum(ok) < 3:
        kdp_lsf[irng] = 0.0
        continue

    p = np.polyfit(xw[ok], yw[ok], 1)
    kdp_lsf[irng] = 0.5 * p[0]

file_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.basename(file_path)

if file_name.endswith(".bz2"):
    file_name = os.path.splitext(file_name)[0]

file_stem = os.path.splitext(file_name)[0]
out_txt = os.path.join(file_dir, file_stem + f"_radial_{radial_idx}_phi_kdp.txt")

out = np.column_stack([
    np.full(dist_ray.shape, radial_idx, dtype=float),
    dist_ray,
    zh_ray,
    rho_ray,
    zdr_ray,
    phi_ray,
    sd_zdr,
    sd_phi,
    zh_qc,
    rho_qc,
    zdr_qc,
    phi_qc,
    phi_reg,
    kdp_lsf
])

np.savetxt(
    out_txt,
    out,
    fmt="%.4f",
    delimiter="\t",
    header="radial_idx\tdistance\tzh\trho\tzdr\tphi\tsd_zdr\tsd_phi\tzh_qc\trho_qc\tzdr_qc\tphi_qc\tphi_reg\tkdp_lsf",
    comments=""
)

print(out_txt)
print("检查1: 仅对单条径向处理 =", radial_idx)
print("检查2: 原始zdr缺测已按前值填充 =", np.all(np.isfinite(zdr_ray[1:]) | ~np.isnan(zdr_ray[1:])))
print("检查3: 原始phi缺测已按前值填充 =", np.all(np.isfinite(phi_ray[1:]) | ~np.isnan(phi_ray[1:])))
print("检查4: 纹理缺测已按前值填充 =", np.all(np.isfinite(sd_zdr[1:])) and np.all(np.isfinite(sd_phi[1:])))
print("检查5: phi_reg缺测未直接删除而是前值延续 =", np.all(np.isfinite(phi_reg[np.where(np.isfinite(phi_qc))[0][0]:])) if np.sum(np.isfinite(phi_qc)) > 0 else True)
print("检查6: KDP拟合失败赋0而非NaN =", not np.any(np.isnan(kdp_lsf)))