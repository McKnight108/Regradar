import numpy as np
import pandas as pd

dat = pd.read_csv(r'./datatxt/kdp_300.txt')

r   = dat['dis'].values
phi = dat['reg_phi'].values

window = 7          # ① 适当加大窗口，抑制噪声
half_w = window // 2
kdp_lsf = np.full(len(phi), np.nan)

for i in range(half_w, len(phi) - half_w):
    r_win   = r[i - half_w:i + half_w + 1]
    phi_win = phi[i - half_w:i + half_w + 1]

    # 跳过窗口内存在 NaN 的点
    mask = np.isfinite(phi_win)
    if mask.sum() < half_w:      # 有效点太少则跳过
        continue

    A = np.vstack([r_win[mask], np.ones(mask.sum())]).T
    a, _ = np.linalg.lstsq(A, phi_win[mask], rcond=None)[0]

    kdp_lsf[i] = 0.5 * a

# ② 将物理上不合理的极小负值截断为 0（可选）
kdp_lsf = np.where(kdp_lsf < 0, 0.0, kdp_lsf)

dat['kdp_lsf'] = kdp_lsf
dat.to_csv('kdp_lsf_300.txt', index=False)