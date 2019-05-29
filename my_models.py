from keras import Model
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Conv1D, Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.layers.core import Reshape, Permute
from keras.regularizers import l1_l2
from keras.optimizers import Adam
import keras.backend as K
from keras.constraints import max_norm
from keras.utils.generic_utils import get_custom_objects

def EEGNet_old(nb_classes,params, Chans=64, Samples=128, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2

    with a few modifications: we use striding instead of max-pooling as this
    helped slightly in classification performance while also providing a
    computational speed-up.

    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.

    Inputs:

        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        params        : dict with regRate, filtNumLayerX, filtNumLayerX, lr
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)

    """

    # start the model
    input_main = Input((1, Chans, Samples))
    layer1 = Conv2D(int(params['filtNumLayer1']), (Chans, 1), input_shape=(1, Chans, Samples),
                    kernel_regularizer=l1_l2(l1=params['regRate'], l2=params['regRate']))(input_main)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = Dropout(params['dropoutRate1'])(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(int(params['filtNumLayer2']), kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=params['regRate']),
                    strides=strides)(permute1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = Dropout(params['dropoutRate2'])(layer2)

    layer3 = Conv2D(int(params['filtNumLayer3']), kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=params['regRate']),
                    strides=strides)(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = Dropout(params['dropoutRate3'])(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    classification_model = Model(inputs=input_main, outputs=softmax)
    opt = Adam(lr=params['lr'])
    classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classification_model

def my_EEGnet(time_samples_num, channels_num,params):
    """
    This implementation need channels_last (tf) in keras configuration file (unlike Vernon's implementation)
    :param time_samples_num:
    :param channels_num:
    :param params:
    :return:
    """
    num_of_filt = 16

    input = Input(shape=(time_samples_num, channels_num,))
    convolved = Conv1D(num_of_filt, kernel_size=(1), activation='elu', padding='same',
                       kernel_regularizer=l1_l2(0.0001))(input)
    convolved = Reshape((1, num_of_filt, time_samples_num))(convolved)
    #
    #
    b_normed = BatchNormalization(axis=2)(convolved)
    dropouted = Dropout(params['dropouts0'])(b_normed)
    #
    # #second
    num_of_filt = 4
    convolved = Conv2D(num_of_filt, kernel_size=(2, 32),
                       activation='elu',
                       # kernel_regularizer=l1_l2(0.0000),
                       data_format='channels_first',
                       padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2, 4), data_format='channels_first')(b_normed)
    dropouted = Dropout(params['dropouts1'])(pooled)
    #
    # #Third
    num_of_filt = 4
    convolved = Conv2D(num_of_filt, kernel_size=(8, 4),
                       activation='elu',
                       # kernel_regularizer=l1_l2(0.0000),
                       data_format='channels_first',
                       padding='same')(dropouted)
    b_normed = BatchNormalization(axis=1)(convolved)
    pooled = MaxPooling2D(pool_size=(2, 4), data_format='channels_first')(
        b_normed)  # 41 time sample point affects this feature
    dropouted = Dropout(params['dropouts2'], seed=1)(pooled)

    # Fourth
    flatten = Flatten()(dropouted)
    out = Dense(2, activation=None)(flatten)
    out = Activation(activation='softmax')(out)
    classification_model = Model(inputs=input, outputs=out)
    opt = Adam(lr=0.005)
    classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return classification_model






