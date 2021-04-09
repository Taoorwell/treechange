import os
from glob import glob
import numpy as np
# from osgeo import gdal
from PIL import Image
from matplotlib import pyplot as plt


def load_data(path, mode):
    images_1_path = sorted(glob(os.path.join(path, r"im1/*.png")))
    images_2_path = sorted(glob(os.path.join(path, r"im2/*.png")))
    masks_path = sorted(glob(os.path.join(path, r"label1/*.png")))
    if mode == 'train':
        images_1_path, images_2_path, masks_path = images_1_path[:2430], \
                                                   images_2_path[:2430], masks_path[0:2430]
    elif mode == 'valid':
        images_1_path, images_2_path, masks_path = images_1_path[2430:-268], \
                                                   images_2_path[2430:-268], masks_path[2430:-268]
    else:
        images_1_path, images_2_path, masks_path = images_1_path[-268:], \
                                                   images_2_path[-268:], masks_path[-268:]
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

# def norma_data(data, norma_methods="z-score"):
#     arr = np.empty(data.shape, dtype=np.float32)
#     for i in range(data.shape[-1]):
#         array = data.transpose(2, 0, 1)[i, :, :]
#         mins, maxs, mean, std = np.percentile(array, 1), np.percentile(array, 99), np.mean(array), np.std(array)
#         if norma_methods == "z-score":
#             new_array = (array-mean)/std
#         else:
#             new_array = np.clip(2*(array-mins)/(maxs-mins), 0, 1)
#         arr[:, :, i] = new_array
#     return arr


# def plot_mask(result):
#     arr_2d = result
#     arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
#     for c, i in palette.items():
#         m = arr_2d == c
#         arr_3d[m] = i
#     plt.imshow(arr_3d)


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
    _, _, images_path = load_data(path='../SECOND_train_set/', mode='test')
    m = get_mask(images_path[0])
    print(m.shape)
#     print(m)
#     plt.imshow(m)
#     plt.show()


