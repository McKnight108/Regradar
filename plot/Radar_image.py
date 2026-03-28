import cinrad
import matplotlib.colors as mcolors

nFiles =  r"D:\Chan\Documents\radar_raw_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
datetime = f.scantime
print(datetime)
"""
ref = f.get_data(2,120,"REF") # 读取第2层的反射率
fig = cinrad.visualize.PPI(ref,style="black", dpi=300, label=True)
fig("dBZ_ppi.png")
"""

phdp = f.get_data(2,120,"PHI") # 读取第2层的差分传播相移
vmin=0
vmax=140
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
fig = cinrad.visualize.PPI(phdp,style="black", dpi=300, cmap="jet", norm=norm, label=[vmin,vmax])
fig("PHDP_ppi.png")

"""
kdp = f.get_data(2,120,"KDP") # 读取第2层的相移率
fig = cinrad.visualize.PPI(kdp,style="black", dpi=300, label=True)
fig("KDP_ppi.png")
"""