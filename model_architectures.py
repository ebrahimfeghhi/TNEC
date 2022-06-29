import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, BatchNormalization, UpSampling1D, Reshape, LSTM, GaussianNoise
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

def convnetLFP(size, dims, filters, kernel_size, cnn_nonlin, nonlin, num_classes, drop, dense_num=10):
    
    data = Input(shape=(size,dims))
    
    # Conv block 1 
    conv_1 = Conv1D(filters=filters[0], kernel_size=kernel_size[0], 
                    padding='same', activation=cnn_nonlin)(data)
    batch_norm_1 = BatchNormalization()(conv_1)
    max_pool_1 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_1)
    drop_1 = Dropout(drop)(max_pool_1)
    
    # Conv block 2
    conv_2 = Conv1D(filters=filters[1], kernel_size=kernel_size[0], 
                    padding='same', activation=cnn_nonlin)(drop_1)
    batch_norm_2 = BatchNormalization()(conv_2)
    max_pool_2 = AveragePooling1D(pool_size=2, strides=2, padding='valid')(batch_norm_2)
    drop_2 = Dropout(drop)(max_pool_2)
    
    # flatten for dense layers 
    flatten_output = Flatten()(drop_2)
    
    dense_1 = Dense(dense_num, activation=cnn_nonlin)(flatten_output)
    batch_norm_3 = BatchNormalization()(dense_1)
    dense_2 = Dense(num_classes, activation=nonlin)(batch_norm_3)
    
    model = Model(inputs=data, outputs=dense_2)

    return model 

