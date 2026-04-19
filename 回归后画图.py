import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os
npz_path = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT_滤除杂波回归.npz"

data = np.load(npz_path)
site_name = str(data["site_name"])
radar_lon = float(data["radar_lon"])
radar_lat = float(data["radar_lat"])
site_text = f"{site_name}  ({radar_lat:.6f}, {radar_lon:.6f})"

azimuth = data["azimuth"].astype(float)
distance = data["distance"].astype(float)

if np.nanmax(np.abs(azimuth)) <= 2 * np.pi + 0.5:
    az = azimuth
else:
    az = np.deg2rad(azimuth)

order = np.argsort(az)
az = az[order]
distance = distance[order, :]

phi_reg = data["phi_reg"][order, :]
kdp_lsf = data["kdp_lsf"][order, :]
zh_qc = data["zh_qc"][order, :]

az_2d = np.broadcast_to(az[:, None], distance.shape)
r_2d = distance

x = r_2d * np.sin(az_2d)
y = r_2d * np.cos(az_2d)

plt.figure(figsize=(8, 8))
pcm = plt.pcolormesh(
    x, y, phi_reg,
    shading="auto",
    cmap="jet",
    vmin=0,
    vmax=140
)

plt.colorbar(pcm, label="phi_reg")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"PPI of phi_reg\n{site_text}")
plt.show()

kdp_levels = np.arange(-1, 11, 1)
kdp_norm = BoundaryNorm(kdp_levels, ncolors=plt.get_cmap("jet").N, clip=True)

plt.figure(figsize=(8, 8))
pcm = plt.pcolormesh(
    x, y, kdp_lsf,
    shading="auto",
    cmap="jet",
    norm=kdp_norm
)


plt.colorbar(pcm, label="kdp_lsf", ticks=kdp_levels)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"PPI of kdp_lsf\n{site_text}")
plt.show()

zh_levels = np.arange(0, 71, 10)
zh_cmap = plt.get_cmap("jet", len(zh_levels) - 1)
zh_norm = BoundaryNorm(zh_levels, ncolors=zh_cmap.N, clip=True)

plt.figure(figsize=(8, 8))
pcm = plt.pcolormesh(
    x, y, zh_qc,
    shading="auto",
    cmap=zh_cmap,
    norm=zh_norm
)


plt.colorbar(pcm, label="zh_qc", ticks=zh_levels)
plt.axis("equal")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"PPI of zh_qc\n{site_text}")
plt.show()