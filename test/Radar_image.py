import cinrad
import matplotlib.pyplot as plt

nFiles =  r"D:\Chan\Documents\radar_raw_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
ref = f.get_data(2,150,"RHO") # 读取第一层的反射率

datetime = f.scantime
print(datetime)

fig = cinrad.visualize.PPI(ref,style="black", dpi=300, label=True)
plt.gca().set_title(f"RHO PPI\n{datetime}", color="white")
fig("rho_ppi.png")
