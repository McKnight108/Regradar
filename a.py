import cinrad

nFiles = r"D:\Chan\Documents\radar_raw_data\ZA001\Z_RADR_I_ZA001_20230730000000_O_DOR_YLD2-D_CAP_FMT.bin.bz2"
f= cinrad.io.StandardData(nFiles)
data = f.get_data(0,230,"REF") # 读取第一层的反射率
ref=f.available_tilt('REF') #REF产品有哪些仰角可以读取
print(ref)
print(f.el)