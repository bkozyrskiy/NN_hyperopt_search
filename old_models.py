from keras import Model
from keras.layers import Input, MaxPooling2D, AveragePooling2D, Conv1D, Conv2D, BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.layers.core import Reshape, Permute
from keras.regularizers import l1_l2
from keras.optimizers import Adam
import keras.backend as K
from keras.constraints import max_norm
from keras.utils.generic_utils import get_custom_objects

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






