import numpy as np
import pandas as pd

dat = pd.read_csv("./datatxt/phdp_300_reg.txt")

r = dat["dis"].values
phi = dat["phdp"].values
reg_phi = dat["reg_phi"].values
raw_kdp = dat["raw_kdp"].values

kdp_reg = np.full(len(r), np.nan)
kdp_lsf = np.full(len(r), np.nan)

valid_mask = ~np.isnan(reg_phi)
valid_indices = np.where(valid_mask)[0]

if len(valid_indices) > 0:
    r_valid = r[valid_mask]
    reg_phi_valid = reg_phi[valid_mask]

    kdp_reg_valid = 0.5 * np.gradient(reg_phi_valid, r_valid)
    kdp_reg[valid_mask] = kdp_reg_valid

    window = 15
    half_w = window // 2

    if len(valid_indices) > window:
        for i in range(half_w, len(valid_indices) - half_w):
            idx = valid_indices[i]
            r_win = r[valid_indices[i - half_w: i + half_w + 1]]
            phi_win = reg_phi[valid_indices[i - half_w: i + half_w + 1]]

            A = np.vstack([r_win, np.ones(len(r_win))]).T
            a, _ = np.linalg.lstsq(A, phi_win, rcond=None)[0]

            kdp_lsf[idx] = max(0.0, 0.5 * a)

out = pd.DataFrame({
    "dis": r,
    "phi": phi,
    "reg_phi": reg_phi,
    "raw_kdp": raw_kdp,
    "reg_kdp": kdp_reg,
    "kdp_lsf": kdp_lsf
})

save_path = "datatxt/kdp_lsf_300_15.txt"
out.to_csv(save_path, index=False, na_rep='NaN')
print("已保存:", save_path)