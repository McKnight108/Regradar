import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

dat = pd.read_csv("datatxt/phdp_300.txt", index_col=False)

out = dat.copy()
out["reg_phi"] = np.nan

valid_mask = dat["phdp"].notna() & (dat["phdp"] >= -900)
valid_dat = dat[valid_mask]

x = valid_dat["dis"].to_numpy()
y = valid_dat["phdp"].to_numpy()

if len(x) > 0:
    ir = IsotonicRegression(out_of_bounds="clip")
    y_reg = ir.fit_transform(x, y)
    out.loc[valid_mask, "reg_phi"] = y_reg

save_path = "datatxt/phdp_300_reg.txt"
out.to_csv(save_path, index=False, na_rep='NaN')

print("已保存:", save_path)
print(out.head())