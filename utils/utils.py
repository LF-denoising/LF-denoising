from glob import glob
import os


def test_img_list_generate(number_list, img_dir):
    all_img_list = [os.path.basename(p) for p in glob(img_dir + "/*.tif")]

    if "test_No0.tif" in all_img_list:
        return [f"test_No{i}.tif" for i in number_list]
    else:
        return [all_img_list[i] for i in number_list]
