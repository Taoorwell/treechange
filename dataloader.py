import os
import random
from glob import glob
import numpy as np
from osgeo import gdal
from matplotlib import pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# TO DO: load images from file path
# Apply augmentation to these images including flipping, cropping and ect.
# load image and super pixel segmentation
# # satellite imagery loading using gdal
def get_image(image_path):
    ds = gdal.Open(image_path)
    image = np.empty((ds.RasterYSize, ds.RasterXSize, ds.RasterCount), dtype=np.float32)
    for b in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(b).ReadAsArray()
        image[:, :, b-1] = band
    image = norma_data(image, norma_methods='min-max')
    return image


def norma_data(data, norma_methods="z-score"):
    arr = np.empty(data.shape, dtype=np.float32)
    for i in range(data.shape[-1]):
        array = data[:, :, i]
        mi, ma, mean, std = np.percentile(array, 1), np.percentile(array, 99), array.mean(), array.std()
        if norma_methods == "z-score":
            new_array = (array-mean)/std
        else:
            new_array = (2*(array-mi)/(ma-mi)).clip(0, 1)
        arr[:, :, i] = new_array
    return arr


def plot_sample(image, segment):
    image = image.numpy()[0, [4, 3, 2], :, :]
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image.transpose((1, 2, 0)))
    plt.xlabel('Image')

    plt.subplot(122)
    plt.imshow(mark_boundaries(image.transpose((1, 2, 0)),
                               segment[0, :, :].numpy(),
                               color=(0, 1, 1)))
    plt.xlabel('Segment')

    plt.show()


def get_device(cuda_preference=True):
    print('cuda available:', torch.cuda.is_available(),
          ', cudnn available:', torch.backends.cudnn.is_available(),
          ', num devices:', torch.cuda.device_count())
    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, segment = sample['Image'], sample['Segment']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        segment = segment[top: top + new_h,
                          left: left + new_w]
        sample = {'Image': image, 'Segment': segment}
        return sample


class Flip(object):
    def __call__(self, sample):
        image, segment = sample['Image'], sample['Segment']
        a = random.choice((0, 1))
        image = np.flip(image, a).copy()
        segment = np.flip(segment, a).copy()

        sample = {'Image': image, 'Segment': segment}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, segment = sample['Image'], sample['Segment']
        image = image.transpose((2, 0, 1))
        sample = {'Image': torch.from_numpy(image),
                  'Segment': torch.from_numpy(segment)}
        return sample


# applying augmentation
class TreeCrownDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # self.segmentation = segmentation
        self.transform = transform
        self.image_path_list = sorted(glob(os.path.join(r'../../treecover_segmentation/quality/',
                                                        self.image_dir, '*.tif')))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        # if torch.is_tensor(item):
        #     item = item.tolist()
        image = get_image(self.image_path_list[item])
        segment = slic(image[:, :, [4, 3, 2]], n_segments=800, compactness=50, start_label=1)
        sample = {'Image': image, 'Segment': segment}
        if self.transform:
            sample = self.transform(sample)
        return sample


def dataloader(image_dir, batch_size, num_workers=2):
    dataset = TreeCrownDataset(image_dir=image_dir,
                               transform=transforms.Compose([RandomCrop(256),
                                                             Flip(),
                                                             ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    device = get_device()
    print(device)
    # # pass
    # dataset = TreeCrownDataset(image_dir='images/',
    #                            transform=transforms.Compose([RandomCrop(256),
    #                                                          Flip(),
    #                                                          ToTensor()]))
    # print(len(dataset))
    # for sample in dataset:
    #     # print(sample['Image'].dtype)
    #     print(sample['Image'].shape, sample['Segment'].shape)
    #     break
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    dataloader = dataloader(image_dir=r'images/', batch_size=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['Image'].shape, sample_batched['Segment'].shape)
        plot_sample(image=sample_batched['Image'], segment=sample_batched['Segment'])
        if i_batch == 3:
            break







