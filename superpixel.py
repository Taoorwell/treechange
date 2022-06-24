from skimage.segmentation import slic, mark_boundaries
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from osgeo import gdal
import numpy as np

# normal rgb image loading
# image_path = r'../SECOND_train_set/im2/00003.png'
# image = plt.imread(image_path)
# print(image.shape)
# print(image.dtype)


# satellite imagery loading using gdal
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


image_path = r'../../treecover_segmentation/quality/images/tile_100.tif'
image = get_image(image_path)
print(image.shape, image.dtype)

plt.imshow(image[:, :, [6, 5, 4]])
plt.show()

# applying k-means from sklearn package to cluster satellite imagery to tree cover and no tree cover
image_flat = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
km = KMeans(n_clusters=4)
km.fit(image_flat)
#
centroids = km.cluster_centers_
# # print(centroids)
# # print(centroids.squeeze())
#
# for center in centroids:
#     print(center)
#
labels = km.labels_
print(labels)

segment_image = np.zeros_like(image_flat, dtype=np.float32)

for i in range(image_flat.shape[0]):
    segment_image[i] = centroids[labels[i]]

segment_image = np.reshape(segment_image, image.shape)
plt.imshow(segment_image[:, :, [6, 5, 4]])
plt.show()

# spatial refinement using superpixel (segments and image labels)
# using SLIC to get superpixels
segments = slic(image[:, :, [6, 5, 4]], 1000, sigma=1)
print(segments)
# print(segments[segments == 0])
plt.imshow(mark_boundaries(segment_image[:, :, [4, 3, 2]], segments, color=(0, 1, 1)))
plt.show()

n_segments = np.unique(segments)
# print(n_segments)
image_labels = np.reshape(labels, (image.shape[0], image.shape[1]))
print(image_labels)

for segment in n_segments:
    segment_labels = image_labels[segments == segment]
    # print(segment_labels)
    u_segment_labels = np.unique(segment_labels)
    hist = np.zeros(len(u_segment_labels))
    for j in range(len(hist)):
        hist[j] = len(np.where(segment_labels == u_segment_labels[j])[0])
    image_labels[segments == segment] = u_segment_labels[np.argmax(hist)]
    # break
image_labels = np.reshape(image_labels, (image_flat.shape[0],))

re_segment_image = np.zeros_like(image_flat, dtype=np.float32)

for i in range(image_flat.shape[0]):
    re_segment_image[i] = centroids[image_labels[i]]

re_segment_image = np.reshape(re_segment_image, image.shape)
plt.imshow(re_segment_image[:, :, [4, 3, 2]])
plt.show()

#


