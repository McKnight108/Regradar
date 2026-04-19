import cinrad

nFiles = r"C:\Users\Chan\Documents\raw_radar_data\ZA003\Z_RADR_I_ZA003_20230730212101_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
data = f.get_data(0,230,"REF") # 读取第一层的反射率
ref=f.available_tilt('REF') #REF产品有哪些仰角可以读取
print(data)
print(f.el)