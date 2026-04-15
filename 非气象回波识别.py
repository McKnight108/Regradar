import cinrad
import numpy as np

file_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f = cinrad.io.StandardData(file_path)

ds_rho = f.get_data(2, 140, "RHO")
ds_zdr = f.get_data(2, 140, "ZDR")
ds_phi = f.get_data(2, 140, "PHI")

rho_name = [i for i in ["RHO", "RHOHV", "CC", "RHV"] if i in ds_rho.data_vars][0]
zdr_name = [i for i in ["ZDR"] if i in ds_zdr.data_vars][0]
phi_name = [i for i in ["PHI", "PHIDP", "PDP"] if i in ds_phi.data_vars][0]

rho = ds_rho[rho_name].values.astype(float)
zdr = ds_zdr[zdr_name].values.astype(float)
phi = ds_phi[phi_name].values.astype(float)

window = 7
pad = window // 2

zdr_pad = np.pad(zdr, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)
phi_pad = np.pad(phi, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)

zdr_win = np.lib.stride_tricks.sliding_window_view(zdr_pad, window_shape=window, axis=1)
phi_win = np.lib.stride_tricks.sliding_window_view(phi_pad, window_shape=window, axis=1)

zdr_mean = np.nanmean(zdr_win, axis=-1)
phi_mean = np.nanmean(phi_win, axis=-1)

sd_zdr = np.sqrt(np.nanmean((zdr_win - zdr_mean[..., None]) ** 2, axis=-1))
sd_phi = np.sqrt(np.nanmean((phi_win - phi_mean[..., None]) ** 2, axis=-1))

non_meteo_mask = (rho < 0.9) | (sd_zdr > 1.0) | (sd_phi > 5.0)

zdr_qc = zdr.copy()
phi_qc = phi.copy()
rho_qc = rho.copy()

zdr_qc[non_meteo_mask] = np.nan
phi_qc[non_meteo_mask] = np.nan
rho_qc[non_meteo_mask] = np.nan