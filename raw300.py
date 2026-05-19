import cinrad
import numpy as np
import pandas as pd

f = cinrad.io.StandardData(r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2")

ds_phi = f.get_data(2, 140, "PHI")
ds_kdp = f.get_data(2, 140, "KDP")

az_idx = 147
radial_phi = ds_phi.isel(azimuth=az_idx)
radial_kdp = ds_kdp.isel(azimuth=az_idx)

var_phi = list(radial_phi.data_vars)[0]
var_kdp = list(radial_kdp.data_vars)[0]

dat_phi = radial_phi[[var_phi]].to_dataframe().reset_index()
dat_kdp = radial_kdp[[var_kdp]].to_dataframe().reset_index()

dat = pd.merge(dat_phi[["distance", var_phi]], dat_kdp[["distance", var_kdp]], on="distance", how="outer")
dat = dat.sort_values("distance")

dat = dat.rename(columns={
    "distance": "dis",
    var_phi: "phdp",
    var_kdp: "raw_kdp"
})

dat['dis'] = np.round(dat['dis'], 3)

save_path = "datatxt/phdp_300.txt"
dat.to_csv(save_path, index=False, na_rep='NaN')

print("已保存:", save_path)
print(dat.head())