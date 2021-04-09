# import tensorflow as tf
from utility import *
from models import *
# from matplotlib import pyplot as plt


# Datasets construction
def image_dataset(path, mode, width, batch_size):
    # image path and mask path dataset
    images_1_path, images_2_path, masks_path = load_data(path, mode)
    datasets = tf.data.Dataset.from_tensor_slices((images_1_path, images_2_path, masks_path))

    # parse path into full image and then into patches
    # define parse function
    def parse_fun(x1, x2, y):
        def f(x1, x2, y):
            x1 = x1.decode()
            x2 = x2.decode()

            y = y.decode()

            x1 = get_image(x1)
            x2 = get_image(x2)
            y = get_mask(y)
            return x1, x2, y

        image1, image2, mask = tf.numpy_function(f, [x1, x2, y], [tf.float32, tf.float32, tf.float32])
        image1.set_shape([width, width, 3])
        image2.set_shape([width, width, 3])
        mask.set_shape([width, width, 1])
        return {'input_1': image1, 'input_2': image2}, mask
    datasets = datasets.map(parse_fun)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    return datasets


if __name__ == '__main__':
    width = 512
    batch_size = 2
    # image_path, mask_path = load_data(path='../', mode='test')
    train_dataset = image_dataset(path=r'../SECOND_train_set/', mode='train',
                                  width=width, batch_size=batch_size)
    # for input_dict, label in train_dataset:
    #     print(input_dict['input_1'].shape, label.shape)
    # for x1, x2, y in train_dataset:
    #     plt.subplot(131)
    #     plt.imshow(x1.numpy()[0])
    #     plt.subplot(132)
    #     plt.imshow(x2.numpy()[0])
    #     plt.subplot(133)
    #     plt.imshow(y.numpy()[0][:, :, 0])
    #     plt.show()

    # test_dataset = image_dataset(path='../', mode='test',
    #                              width=width, batch_size=1)
    # model restore
    # model = siamese_unet(input_shape=(width, width, 3))
    # model.load_weights('checkpoints/ckpt')

    # for i, (image, mask) in enumerate(test_dataset):
    #     mask_pred = model.predict(image)
    #     acc = dice(mask, mask_pred)
    #     mask_pred = (model.predict(image)[0] > 0.5) * 1
    #     image_id = image_path[i].split('_')[-1].split('.')[0]
    #
    #     plt.subplot(131)
    #     plt.imshow(image.numpy()[0][:, :, [4, 3, 2]])
    #     plt.xlabel('Image_{}'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(132)
    #     plot_mask(mask.numpy()[0][:, :, 0])
    #     plt.xlabel('mask_{}'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.subplot(133)
    #     plot_mask(mask_pred[:, :, 0])
    #     plt.xlabel('mask_{}_pre'.format(image_id))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    #     plt.title('Accuracy:{:.2%}'.format(acc))
    #     # plt.show()
    #     plt.savefig('pre/treecover/Image_{}_pre'.format(image_id))
    #     print('finish: {}'.format(i))
    #     if i == 34:
    #         break
    #
    # val_dataset = image_dataset(path='../..', mode='eval',
    #                             width=width, batch_size=batch_size)

    # model construction
    model = siamese_unet(input_shape=(width, width, 3))
    # model.summary()
    #
    # # model compile
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  loss=dice_loss, metrics=[dice])
    #
    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir',
                                                           histogram_freq=1)

    model.fit(train_dataset, steps_per_epoch=1350, epochs=25,
              callbacks=[tensorboard_callbacks])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt')

