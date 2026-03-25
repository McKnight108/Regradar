import cinrad
import matplotlib.pyplot as plt
import numpy as np

nFiles = r"D:\Chan\Documents\radar_raw_data\ZA001\Z_RADR_I_ZA001_20230730143000_O_DOR_YLD2-D_CAP_FMT.bin.bz2"

f = cinrad.io.StandardData(nFiles)

phi = f.get_data(2, 230, 'PHI')
ref = f.get_data(2, 230, 'REF')
rho = f.get_data(2, 230, 'RHO')
time = f.scantime

azimuth = phi['azimuth'].values
distance = phi['distance'].values

ref_arr = ref['REF'].values
rho_arr = rho['RHO'].values
phi_arr = phi['PHI'].values

candidates = []

for i in range(ref_arr.shape[0]):
    ref_line = ref_arr[i, :]
    rho_line = rho_arr[i, :]

    # -------- REF新条件：整条径向上至少10个点 REF > 45 --------
    if np.sum(ref_line > 45) < 10:
        continue

    # -------- RHO条件：保持你原来的“某一段距离上持续较大”逻辑 --------
    rho_mask = rho_line > 0.85

    current_start = None
    current_len = 0
    found = False

    for j in range(len(rho_mask)):
        if rho_mask[j]:
            if current_start is None:
                current_start = j
            current_len += 1
        else:
            if current_len >= 5:   # 这里的 5 按你原来的阈值改
                start_idx = current_start
                end_idx = j - 1
                found = True
                break
            current_start = None
            current_len = 0

    # 如果连续段一直延续到最后一个点
    if (not found) and (current_len >= 5):
        start_idx = current_start
        end_idx = len(rho_mask) - 1
        found = True

    if found:
        candidates.append({
            'radial_index': i,
            'start_idx': start_idx,
            'end_idx': end_idx
        })

print(f"共找到 {len(candidates)} 条 candidate 径向")


# 选中的径向
c = candidates[3]

i = c['radial_index']
start_idx = c['start_idx']
end_idx = c['end_idx']

# 数据
distance = phi['distance'].values
azimuth = phi['azimuth'].values

phi_line = phi_arr[i, :]
ref_line = ref_arr[i, :]
rho_line = rho_arr[i, :]

fig, ax = plt.subplots(3,1, figsize=(10,10), sharex=True)

# REF
ax[0].plot(distance, ref_line)
ax[0].set_ylabel("REF (dBZ)")
ax[0].grid()

# RHO
ax[1].plot(distance, rho_line)
#ax[1].axhline(0.85, linestyle='--')
ax[1].set_ylabel("RHO")
ax[1].grid()

# PHI
ax[2].plot(distance, phi_line)
ax[2].set_ylabel("PHI (deg)")
ax[2].set_xlabel("Distance (km)")
ax[2].grid()

for a in ax:
    a.axvspan(distance[start_idx], distance[end_idx], color='gray', alpha=0.3)
plt.ylim(40, 120)
plt.xlim(0, 40)
fig.suptitle(f"ZA001-{time}-(Azimuth={np.rad2deg(azimuth[i]):.1f}°)")
plt.show()
