# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:40:24 2020

@author: alfah
"""
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv3D, ZeroPadding2D, UpSampling3D, Dense, concatenate, Conv3DTranspose
from keras.layers.pooling import GlobalAveragePooling3D, MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
import numpy as np

smooth = 1.
dropout_rate = 0.5

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'relu'

    x = Conv3D(nb_filter, (kernel_size, kernel_size, kernel_size), name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    
    #x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    #x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    #x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    bn_axis = -1
    img_input = Input(shape=(512, 512, None, 1), name='main_input')


    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling3D((2, 2, 1), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D((2, 2, 1), name='pool2')(conv2_1)

    up1_2 = Conv3DTranspose(nb_filter[0], (3,3,3), strides=(2, 2, 1), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D((2, 2, 1), name='pool3')(conv3_1)

    up2_2 = Conv3DTranspose(nb_filter[1], (3,3,3), strides=(2, 2, 1), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3DTranspose(nb_filter[0], (3,3,3), strides=(2, 2, 1), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling3D((2, 2, 1), name='pool4')(conv4_1)

    up3_2 = Conv3DTranspose(nb_filter[2], (3,3,3), strides=(2, 2, 1), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv3DTranspose(nb_filter[1], (3,3,3), strides=(2, 2, 1), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3DTranspose(nb_filter[0], (3,3,3), strides=(2, 2, 1), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv3DTranspose(nb_filter[3], (3,3,3), strides=(2, 2, 1), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv3DTranspose(nb_filter[2], (3,3,3), strides=(2, 2, 1), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv3DTranspose(nb_filter[1], (3,3,3), strides=(2, 2, 1), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv3DTranspose(nb_filter[0], (3,3,3), strides=(2, 2, 1), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv3D(num_class, (1, 1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(img_input, [nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(img_input, [nestnet_output_4])

    return model

model = Nest_Net(96,96,1)
model.summary(positions=[.33, .7, .8, 1.])
#%%
import pandas as pd
import numpy as np

df = pd.read_csv('hemorrhage_diagnosis_raw_ct.csv')
f = df.to_numpy()
f = f[:,:-1]
f[:,0] = f[:,0] - 49
s = np.unique(f[:,0])
y=[] #no hemorrhage
n=[] #have hemorrhage
size=[]
for i in s:
    d = f[f[:,0]==i]
    size.append(d.shape[0])#max-58, min-31
    if np.sum(d[:,-1]) == d.shape[0]:
        y.append(i)
    else:
        n.append(i)
n, y = np.asarray(n), np.asarray(y)

#(['PatientNumber', 'SliceNumber', 'Intraventricular', 'Intraparenchymal', 'Subarchanoid', 'Epidural', 'Subdural', 'NoHemorrhage'])
