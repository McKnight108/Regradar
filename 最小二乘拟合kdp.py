import numpy as np
import pandas as pd

dat = pd.read_csv(r'./datatxt/kdp_300.txt')

r = dat['dis'].values
phi = dat['reg_phi'].values

window = 7
half_w = window // 2

kdp_lsf = np.full(len(phi), np.nan)

for i in range(half_w, len(phi) - half_w):
    r_win = r[i - half_w:i + half_w + 1]
    phi_win = phi[i - half_w:i + half_w + 1]

    A = np.vstack([r_win, np.ones(len(r_win))]).T
    a, b = np.linalg.lstsq(A, phi_win, rcond=None)[0]

    kdp_lsf[i] = 0.5 * a

dat['kdp_lsf'] = kdp_lsf

dat.to_csv('kdp_lsf_300.txt', index=False)