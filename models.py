import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, concatenate


def conv_block(x, filters, strides):
    main_path = Conv2D(filters=filters[0], kernel_size=(3, 3),
                       strides=strides[0], padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters= filters[1], kernel_size=(3, 3),
                       strides=strides[1], padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    shortcut = Conv2D(filters=filters[1], kernel_size=(3, 3),
                      strides=strides[1], padding='same')(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = tf.keras.layers.add([shortcut, main_path])

    return main_path


def encoder(x):
    to_decoder = []
    main_path = conv_block(x, [32, 32], [(1, 1), (1, 1)])
    to_decoder.append(main_path)

    main_path = conv_block(main_path, [64, 64], [(1, 1), (2, 2)])
    to_decoder.append(main_path)

    main_path = conv_block(main_path, [128, 128], [(1, 1), (2, 2)])
    to_decoder.append(main_path)

    main_path = conv_block(main_path, [256, 256], [(1, 1), (2, 2)])
    to_decoder.append(main_path)

    return to_decoder


def slice_concatenate(x1, x2, x3):
    if x1.shape != x2.shape:
        x1 = x1[:, :x2.shape[1], :x2.shape[2], :]
    x = concatenate([x1, x2, x3], axis=3)
    return x


def decoder(x, from_decoder1, from_decoder2):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = slice_concatenate(main_path, from_decoder1[2], from_decoder2[2])
    main_path = conv_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = slice_concatenate(main_path, from_decoder1[1], from_decoder2[1])
    main_path = conv_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = slice_concatenate(main_path, from_decoder1[0], from_decoder2[0])
    main_path = conv_block(main_path, [32, 32], [(1, 1), (1, 1)])

    return main_path


def residual_unet(input_shape):
    inputs = Input(input_shape)
    to_decoder = encoder(inputs)
    main_path = UpSampling2D(size=(2, 2))(to_decoder[-1])
    main_path = concatenate([main_path, to_decoder[2]], axis=3)
    main_path = conv_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, to_decoder[1]], axis=3)
    main_path = conv_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, to_decoder[0]], axis=3)
    main_path = conv_block(main_path, [32, 32], [(1, 1), (1, 1)])
    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(main_path)

    return Model(inputs=inputs, outputs=output)


def siamese_residual_unet(input_shape, mode):
    # siamese residual Unet: two input in encoder part using weights sharing
    # and concatenate or diff both into the decoder part.
    inputs = Input(input_shape)
    encoder_part = Model(inputs=inputs, outputs=encoder(inputs))

    input1, input2 = Input(input_shape), Input(input_shape)
    to_decoder_1, to_decoder_2 = encoder_part([input1]), encoder_part([input2])
    if mode == 'concat':
        input12 = concatenate([to_decoder_1[-1], to_decoder_2[-1]], axis=-1)
    else:
        input12 = tf.math.abs(to_decoder_1[-1] - to_decoder_2[-1])

    output = decoder(input12, to_decoder_1, to_decoder_2)
    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(output)

    return Model(inputs={'input_1': input1, 'input_2': input2}, outputs=output)


def dual_residual_unet(input_shape, mode):
    # dual residual Unet: each input has its own encoder part weights (no sharing),
    # and then concatenate or differ each other to the decoder part.
    input1, input2 = Input(input_shape), Input(input_shape)
    to_decoder_1, to_decoder_2 = encoder(input1), encoder(input2)
    if mode == 'concat':
        input12 = concatenate([to_decoder_1[-1], to_decoder_2[-1]], axis=-1)
    else:
        input12 = tf.math.abs(to_decoder_1[-1] - to_decoder_2[-1])

    output = decoder(input12, to_decoder_1, to_decoder_2)
    output = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(output)

    return Model(inputs={'input_1': input1, 'input_2': input2}, outputs=output)


def dice(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / denominator


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator / denominator)


# if __name__ == '__main__':
    # model = residual_unet(input_shape=(512, 512, 6))
    # model = siamese_residual_unet(input_shape=(512, 512, 3), mode='concat')
    # model = dual_residual_unet(input_shape=(512, 512, 3), mode='diff')
    # model.summary()

