import cinrad
import matplotlib.colors as mcolors
import numpy as np
nFiles =  r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
datetime = f.scantime
print(datetime)
"""
ref = f.get_data(2,120,"REF") # 读取第2层的反射率
fig = cinrad.visualize.PPI(ref,style="black", dpi=300, label=True)
fig("dBZ_ppi.png")
"""
colors = [
    "#21B6E8", "#6A6EB3", "#2756A5", "#96CE93",
    "#6FBD44", "#438A45", "#E9E84E", "#DEE244",
    "#C0BE3C", "#F27A3E", "#ED4041", "#C03134",
    "#7E2225", "#EB3595", "#A55BA3"
]
cmap = mcolors.ListedColormap(colors)
bounds = np.arange(0, 160, 10)
phdp = f.get_data(2,120,"PHI") # 读取第2层的差分传播相移
norm = mcolors.BoundaryNorm(bounds, cmap.N)
fig = cinrad.visualize.PPI(
    phdp,
    style="black",
    dpi=300,
    cmap=cmap,   # 加这一行
    norm=norm,
    label=[0, 150]  # 或者 [0,150]
)
fig("PHDP_ppi.png")
"""
kdp = f.get_data(2,120,"KDP") # 读取第2层的相移率
fig = cinrad.visualize.PPI(kdp,style="black", dpi=300, label=True)
fig("KDP_ppi.png")
"""