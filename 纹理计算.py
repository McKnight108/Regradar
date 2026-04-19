import cinrad
import numpy as np
import pandas as pd
from pathlib import Path

file_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
tilt = 2
drange = 140
window = 7
radial_ids = [17, 147]

out_dir = Path(r"C:\Users\Chan\PycharmProjects\Regradar\datatxt\texture_txt")
out_dir.mkdir(exist_ok=True)

f = cinrad.io.StandardData(file_path)

ds_zdr = f.get_data(tilt, drange, "ZDR")
ds_phi = f.get_data(tilt, drange, "PHI")

zdr = ds_zdr["ZDR"].values.astype(float)
phi = ds_phi["PHI"].values.astype(float)


distance = ds_zdr["distance"].values.astype(float)

if np.nanmax(distance) > 1000:
    distance_km = distance / 1000.0
else:
    distance_km = distance.copy()

bin_id = np.arange(zdr.shape[1])

pad = window // 2

zdr_pad = np.pad(zdr, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)
phi_pad = np.pad(phi, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)

zdr_win = np.lib.stride_tricks.sliding_window_view(zdr_pad, window_shape=window, axis=1)
phi_win = np.lib.stride_tricks.sliding_window_view(phi_pad, window_shape=window, axis=1)

valid_zdr = np.sum(~np.isnan(zdr_win), axis=-1)
valid_phi = np.sum(~np.isnan(phi_win), axis=-1)

zdr_mean = np.full(zdr_win.shape[:-1], np.nan, dtype=float)
phi_mean = np.full(phi_win.shape[:-1], np.nan, dtype=float)

zdr_ok = valid_zdr > 0  # strict -> 3
phi_ok = valid_phi > 0  # strict -> 3

zdr_mean[zdr_ok] = np.nansum(zdr_win[zdr_ok], axis=-1) / valid_zdr[zdr_ok]
phi_mean[phi_ok] = np.nansum(phi_win[phi_ok], axis=-1) / valid_phi[phi_ok]

sd_zdr = np.full(zdr_win.shape[:-1], np.nan, dtype=float)
sd_phi = np.full(phi_win.shape[:-1], np.nan, dtype=float)

sd_zdr[zdr_ok] = np.sqrt(
    np.nansum((zdr_win[zdr_ok] - zdr_mean[zdr_ok, None]) ** 2, axis=-1) / valid_zdr[zdr_ok]
)
sd_phi[phi_ok] = np.sqrt(
    np.nansum((phi_win[phi_ok] - phi_mean[phi_ok, None]) ** 2, axis=-1) / valid_phi[phi_ok]
)

file_stem = Path(file_path).stem

for radial_id in radial_ids:
    df = pd.DataFrame({
        "bin": bin_id,
        "dis": distance_km,
        "ZDR": zdr[radial_id, :],
        "SD_ZDR": sd_zdr[radial_id, :],
        "PHIDP": phi[radial_id, :],
        "SD_PHIDP": sd_phi[radial_id, :]
    })
    df.to_csv(out_dir / f"{file_stem}_tilt{tilt}_radial{radial_id}_texture.txt", sep="\t", index=False, float_format="%.6f")