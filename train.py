# import tensorflow as tf
from utility import *
from models import *
# from matplotlib import pyplot as plt


# Datasets construction
def image_dataset(path, mode, width, batch_size, dual=True):
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
        if dual is True:
            return {'input_1': image1, 'input_2': image2}, mask
        else:
            image = tf.concat([image1, image2], axis=2)
            return image, mask
    datasets = datasets.map(parse_fun)
    datasets = datasets.batch(batch_size)
    datasets = datasets.repeat()
    return datasets


if __name__ == '__main__':
    width = 512
    batch_size = 3
    train_steps = 2430 // batch_size
    valid_steps = 270 // batch_size
    # image_path, mask_path = load_data(path='../', mode='test')
    train_dataset = image_dataset(path=r'../SECOND_train_set/', mode='train',
                                  width=width, batch_size=batch_size, dual=False)
    valid_dataset = image_dataset(path=r'../SECOND_train_set/', mode='valid',
                                  width=width, batch_size=batch_size, dual=False)

    # for image, mask in train_dataset:
    #     # print(image['input_1'].shape, image['input_2'].shape, mask.shape)
    #     print(image.shape, mask.shape)
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
    # model = siamese_residual_unet(input_shape=(width, width, 3), mode='concat')
    # model = dual_residual_unet(input_shape=(width, width, 3), mode='concat')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = residual_unet(input_shape=(width, width, 6))
    # model.summary()
    # # model compile
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss=dice_loss, metrics=[dice])
    # tensorboard
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir='tb_callback_dir',
                                                           histogram_freq=1)
    #
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs=25, validation_data=valid_dataset,
              validation_steps=valid_steps,
              callbacks=[tensorboard_callbacks])
    # model.save('model.h5')
    model.save_weights('checkpoints/ckpt')

