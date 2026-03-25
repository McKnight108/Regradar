import cinrad


nFiles =  r"D:\Chan\Documents\radar_raw_data\ZA009\Z_RADR_I_ZA009_20230730040000_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
ref = f.get_data(2,230,"REF") # 读取第一层的反射率

datetime = f.scantime
print(datetime)

fig = cinrad.visualize.PPI(ref,style="white", label=True)
fig("ppi.png")
