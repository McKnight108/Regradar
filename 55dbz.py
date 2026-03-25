import os
import numpy as np
import cinrad
from concurrent.futures import ProcessPoolExecutor, as_completed

data_dir = r"D:\Chan\Documents\radar_raw_data\ZA009"


def check_file(filepath):
    file = os.path.basename(filepath)
    try:
        f = cinrad.io.StandardData(filepath)

        tilt = int(f.available_tilt('REF')[2])
        ref = f.get_data(tilt, 230, 'REF')
        ref_arr = ref['REF'].values

        count_55 = np.sum(np.isfinite(ref_arr) & (ref_arr > 50))

        if count_55 >= 100:
            return file
        return None

    except Exception:
        return None


if __name__ == "__main__":
    files = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if file.endswith(".bz2")
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(check_file, fp) for fp in files]

        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                print(result)