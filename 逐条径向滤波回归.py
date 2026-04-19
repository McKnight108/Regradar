import os
import cinrad
import numpy as np
from sklearn.isotonic import IsotonicRegression

file_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"

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

azimuth = ds_phi["azimuth"].values.astype(float)
site_name = str(ds_ref.attrs["site_name"])
radar_lon = float(ds_ref.attrs["site_longitude"])
radar_lat = float(ds_ref.attrs["site_latitude"])

for iaz in range(zdr.shape[0]):
    for irng in range(zdr.shape[1]):
        if not np.isfinite(zdr[iaz, irng]):
            if irng > 0:
                zdr[iaz, irng] = zdr[iaz, irng - 1]

for iaz in range(phi.shape[0]):
    for irng in range(phi.shape[1]):
        if not np.isfinite(phi[iaz, irng]):
            if irng > 0:
                phi[iaz, irng] = phi[iaz, irng - 1]

window = 7
pad = window // 2

zdr_pad = np.pad(zdr, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)
phi_pad = np.pad(phi, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)

zdr_win = np.lib.stride_tricks.sliding_window_view(zdr_pad, window_shape=window, axis=1)
phi_win = np.lib.stride_tricks.sliding_window_view(phi_pad, window_shape=window, axis=1)

valid_zdr_count = np.sum(np.isfinite(zdr_win), axis=-1)
valid_phi_count = np.sum(np.isfinite(phi_win), axis=-1)

zdr_sum = np.nansum(zdr_win, axis=-1)
phi_sum = np.nansum(phi_win, axis=-1)

zdr_mean = np.full(zdr.shape, np.nan, dtype=float)
phi_mean = np.full(phi.shape, np.nan, dtype=float)

zdr_mean[valid_zdr_count > 0] = zdr_sum[valid_zdr_count > 0] / valid_zdr_count[valid_zdr_count > 0]
phi_mean[valid_phi_count > 0] = phi_sum[valid_phi_count > 0] / valid_phi_count[valid_phi_count > 0]

sd_zdr = np.full(zdr.shape, np.nan, dtype=float)
sd_phi = np.full(phi.shape, np.nan, dtype=float)

zdr_var_sum = np.nansum((zdr_win - zdr_mean[..., None]) ** 2, axis=-1)
phi_var_sum = np.nansum((phi_win - phi_mean[..., None]) ** 2, axis=-1)

sd_zdr[valid_zdr_count > 0] = np.sqrt(zdr_var_sum[valid_zdr_count > 0] / valid_zdr_count[valid_zdr_count > 0])
sd_phi[valid_phi_count > 0] = np.sqrt(phi_var_sum[valid_phi_count > 0] / valid_phi_count[valid_phi_count > 0])

for iaz in range(sd_zdr.shape[0]):
    for irng in range(sd_zdr.shape[1]):
        if not np.isfinite(sd_zdr[iaz, irng]):
            if irng > 0:
                sd_zdr[iaz, irng] = sd_zdr[iaz, irng - 1]

for iaz in range(sd_phi.shape[0]):
    for irng in range(sd_phi.shape[1]):
        if not np.isfinite(sd_phi[iaz, irng]):
            if irng > 0:
                sd_phi[iaz, irng] = sd_phi[iaz, irng - 1]

non_meteo_mask = (rho < 0.9) | (sd_zdr > 2.0) | (sd_phi > 25.0)

zh_qc = zh.copy()
rho_qc = rho.copy()
zdr_qc = zdr.copy()
phi_qc = phi.copy()

zh_qc[non_meteo_mask] = np.nan
rho_qc[non_meteo_mask] = np.nan
zdr_qc[non_meteo_mask] = np.nan
phi_qc[non_meteo_mask] = np.nan

phi_reg = np.full_like(phi_qc, np.nan, dtype=float)

for iaz in range(phi_qc.shape[0]):
    x = distance_2d[iaz, :]
    y = phi_qc[iaz, :]

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) >= 2:
        xv = x[valid]
        yv = y[valid]

        order = np.argsort(xv)
        xv = xv[order]
        yv = yv[order]

        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        y_iso = ir.fit_transform(xv, yv)

        valid_idx = np.where(valid)[0][order]
        phi_reg[iaz, valid_idx] = y_iso

phi_reg[~(np.isfinite(zh) & np.isfinite(phi))] = np.nan

kdp_lsf = np.full_like(phi_reg, np.nan, dtype=float)

if distance.ndim == 1:
    gate_diffs = np.diff(distance)
else:
    gate_diffs = np.diff(distance_2d, axis=1).reshape(-1)

gate_diffs = gate_diffs[np.isfinite(gate_diffs) & (gate_diffs > 0)]
if gate_diffs.size > 0:
    gate_len = np.nanmedian(gate_diffs)
else:
    gate_len = 0.075

for iaz in range(phi_reg.shape[0]):
    phi_ray = phi_reg[iaz, :]
    zh_ray = zh_qc[iaz, :]
    x_ray = distance_2d[iaz, :]

    for irng in range(phi_reg.shape[1]):
        if not np.isfinite(phi_ray[irng]):
            continue
        if not np.isfinite(zh_ray[irng]):
            continue
        if not np.isfinite(x_ray[irng]):
            continue

        if zh_ray[irng] > 45:
            half_win = max(1, int(round(0.75 / gate_len)))
        elif zh_ray[irng] > 35:
            half_win = max(1, int(round(1.5 / gate_len)))
        else:
            half_win = max(1, int(round(2.25 / gate_len)))

        i0 = max(0, irng - half_win)
        i1 = min(phi_reg.shape[1], irng + half_win + 1)

        xw = x_ray[i0:i1]
        yw = phi_ray[i0:i1]

        ok = np.isfinite(xw) & np.isfinite(yw)
        if np.sum(ok) < 3:
            kdp_lsf[iaz, irng] = 0.0
            continue

        p = np.polyfit(xw[ok], yw[ok], 1)
        kdp_lsf[iaz, irng] = 0.5 * p[0]
kdp_lsf[~(np.isfinite(zh) & np.isfinite(phi))] = np.nan

file_dir = os.path.dirname(file_path)
file_name = os.path.basename(file_path)

if file_name.endswith(".bz2"):
    file_name = os.path.splitext(file_name)[0]

file_stem = os.path.splitext(file_name)[0]
out_npz = os.path.join(file_dir, file_stem + "_滤除杂波回归.npz")

np.savez_compressed(
    out_npz,
    site_name=site_name,
    radar_lon=radar_lon,
    radar_lat=radar_lat,
    azimuth=azimuth,
    distance=distance_2d,
    zh=zh,
    rho=rho,
    zdr=zdr,
    phi=phi,
    sd_zdr=sd_zdr,
    sd_phi=sd_phi,
    zh_qc=zh_qc,
    rho_qc=rho_qc,
    zdr_qc=zdr_qc,
    phi_qc=phi_qc,
    phi_reg=phi_reg,
    kdp_lsf=kdp_lsf
)

print(out_npz)
