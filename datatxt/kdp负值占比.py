import pandas as pd
import numpy as np

input_path = r"kdp_lsf_300_15.txt"

df = pd.read_csv(input_path)

cols = ["kdp", "reg_kdp", "kdp_lsf"]

bins = np.linspace(-16, 16, 7)
bin_center = (bins[:-1] + bins[1:]) / 2

hist_out = pd.DataFrame()
hist_out["bin_center"] = bin_center

for col in cols:
    x = df[col].to_numpy(dtype=float)
    x = x[~np.isnan(x)]

    count, _ = np.histogram(x, bins=bins)

    hist_out[col + "_count"] = count
    hist_out[col + "_frequency"] = count / len(x)

hist_out.to_csv("kdp_histogram_wide.csv", index=False, encoding="utf-8-sig")


grad_out = pd.DataFrame()
grad_out["dis_mid"] = (df["dis"].to_numpy(dtype=float)[:-1] + df["dis"].to_numpy(dtype=float)[1:]) / 2

for col in cols:
    x = df[col].to_numpy(dtype=float)
    grad_out[col + "_abs_diff"] = np.abs(np.diff(x))

grad_out.to_csv("kdp_abs_diff_wide.csv", index=False, encoding="utf-8-sig")






