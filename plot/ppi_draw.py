import matplotlib
matplotlib.use("Agg")

import cinrad
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool

folder = r"D:\Chan\Documents\radar_raw_data\ZA003"
save_dir = r"D:\Chan\Documents\radar_raw_data\图像\el2(1.5deg)\ZA003"

os.makedirs(save_dir, exist_ok=True)


def draw_one(file):
    if not file.endswith(".bz2"):
        return

    path = os.path.join(folder, file)
    save_path = os.path.join(save_dir, file + ".png")

    # 已存在就跳过
    if os.path.exists(save_path):
        print("已存在，跳过:", file)
        return

    try:
        f = cinrad.io.StandardData(path)
        ref = f.get_data(2, 230, "REF")

        fig = cinrad.visualize.PPI(
            ref,
            style="black",
            add_city_names=False
        )

        fig(save_path)
        plt.close("all")
        print("完成:", file)

    except Exception as e:
        plt.close("all")
        print("失败:", file, e)


if __name__ == "__main__":
    files = sorted(os.listdir(folder))

    # 进程数不要开满，留一点余量更稳
    nproc = 14

    with Pool(processes=nproc) as pool:
        pool.map(draw_one, files)