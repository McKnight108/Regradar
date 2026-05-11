import os
import numpy as np

npz_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT_滤波回归ZPHI衰减订正.npz"
ray_idx = 147

data = np.load(npz_path)

distance = data["distance"]
zh = data["zh"]
zh_attcorr = data["zh_attcorr"]
phi = data["phi"]
phi_reg = data["phi_reg"]

if "kdp" in data.files:
    kdp = data["kdp"]
else:
    kdp = np.full_like(phi, np.nan, dtype=float)

kdp_lsf = data["kdp_lsf"]

if distance.ndim == 2:
    dist_ray = distance[ray_idx, :]
else:
    dist_ray = distance[:]

zh_ray = zh[ray_idx, :]
zh_attcorr_ray = zh_attcorr[ray_idx, :]
phi_ray = phi[ray_idx, :]
phi_reg_ray = phi_reg[ray_idx, :]
kdp_ray = kdp[ray_idx, :]
kdp_lsf_ray = kdp_lsf[ray_idx, :]

out = np.column_stack([
    dist_ray,
    zh_ray,
    zh_attcorr_ray,
    phi_ray,
    phi_reg_ray,
    kdp_ray,
    kdp_lsf_ray
])

out_txt = os.path.splitext(npz_path)[0] + f"_ray{ray_idx:03d}.txt"

np.savetxt(
    out_txt,
    out,
    fmt="%.6f",
    delimiter="\t",
    header="distance\tzh\tzh_attcorr\tphi\tphi_reg\tkdp\tkdp_lsf",
    comments=""
)

print(out_txt)