import os
from glob import glob
import numpy as np
# from osgeo import gdal
from PIL import Image
from matplotlib import pyplot as plt


def load_data(path, mode, frac=.1):
    images_1_path = np.array(sorted(glob(os.path.join(path, r"im1/*.png"))))
    images_2_path = np.array(sorted(glob(os.path.join(path, r"im2/*.png"))))
    masks_path = np.array(sorted(glob(os.path.join(path, r"label1/*.png"))))

    np.random.seed(1)
    idx = np.arange(len(images_1_path))
    idx = np.random.permutation(idx)
    train_index = idx[:int(-2 * frac * len(idx))]
    valid_index = idx[int(-2 * frac * len(idx)): int(-1 * frac * len(idx))]
    test_index = idx[int(-1 * frac * len(idx)):]
    print(test_index)

    if mode == 'train':
        images_1_path, images_2_path, masks_path = images_1_path[train_index], \
                                                   images_2_path[train_index], masks_path[train_index]
    elif mode == 'valid':
        images_1_path, images_2_path, masks_path = images_1_path[valid_index], \
                                                   images_2_path[valid_index], masks_path[valid_index]
    else:
        images_1_path, images_2_path, masks_path = images_1_path[test_index], \
                                                   images_2_path[test_index], masks_path[test_index]
    return images_1_path, images_2_path, masks_path


def get_image(image_path):
    image = Image.open(image_path)
    image_arr = np.asarray(image, dtype=np.float32) / 255.0
    return image_arr


def get_mask(mask_path):
    mask = Image.open(mask_path)
    mask_arr = np.asarray(mask)
    mask_2d = np.zeros(shape=(mask_arr.shape[0], mask_arr.shape[1], 1), dtype=np.float32)
    m = mask_arr != (255, 255, 255)
    mask_2d[m[:, :, 0]] = 1
    return mask_2d


palette = {0: (255, 255, 255),  # White
           6: (0, 191, 255),  # DeepSkyBlue
           1: (34, 139, 34),  # ForestGreen
           3: (255, 0, 255),  # Magenta
           2: (0, 255, 0),  # Lime
           5: (255, 127, 80),  # Coral
           4: (255, 0, 0),  # Red
           7: (0, 255, 255),  # Cyan
           8: (0, 255, 0),  # Lime
           9: (0, 128, 128),
           10: (128, 128, 0),
           11: (255, 128, 128),
           12: (128, 128, 255),
           13: (128, 255, 128),
           14: (255, 128, 255),
           15: (165, 42, 42),
           16: (175, 238, 238)}

if __name__ == '__main__':
    mask = get_mask(mask_path='../SECOND_train_set/label1/00011.png')
    plt.imshow(mask)
    plt.show()
