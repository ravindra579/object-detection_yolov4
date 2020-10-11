NUM_CLASS=80
from keras.layers import ZeroPadding2D,Conv2D,BatchNormalization,LeakyReLU,UpSampling2D,Input,Activation,MaxPool2D,Reshape,concatenate
from keras.models import Model
import tensorflow as tf
import os
from cv2 import cv2 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.utils import get_custom_objects
def conv_(input_layer,filters_shape,downsample=False,bn=True,activate=True,activate_type="leaky"):
    if downsample:
        input_layer=ZeroPadding2D(((1,0),(1,0)))(input_layer)
        padding='valid'
        strides=2
    else:
        strides=1
        padding='same'
    conv=Conv2D(filters=filters_shape[1],kernel_size=filters_shape[0],strides=strides,padding=padding)(input_layer)
    if bn:
        conv=BatchNormalization()(conv)
    if activate:
        if activate_type=='leaky':
            conv=tf.nn.leaky_relu(conv,alpha=0.1)
        else:
            conv=Activation('mish')(conv)
    return conv
class Mish(Activation): 
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'
def mysoftplus(x):
    mask_min = tf.cast((x<-20.0),tf.float32)
    ymin = mask_min*tf.math.exp(x)
    mask_max = tf.cast((x>20.0),tf.float32)
    ymax = mask_max*x
    mask= tf.cast((abs(x)<=20.0),tf.float32)
    y = mask*tf.math.log(tf.math.exp(x) + 1.0)
    return(ymin+ymax+y)
def mish(x):
    return (x*tf.math.tanh(mysoftplus(x)))
get_custom_objects().update({'mish':Mish(mish)})
def residual_block_a(input_layer,filter_1,filter_2,activate_type="leaky"):
    input_=input_layer
    conv=conv_(input_layer,filters_shape=(1,filter_1),activate_type=activate_type)
    conv=conv_(conv,filters_shape=(3,filter_2),activate_type=activate_type)
    resid_out=conv+input_
    return resid_out
def yolov4_model():
    input_image=Input(shape=(416,416,3))
    #--------------------cspdarknet begins---------------------------------------

    input_data=conv_(input_image,(3,32),activate_type="mish")# filters 32 size 3 x 3 stride=1 padding=same no zeropadding
    input_data=conv_(input_data,(3,64),downsample=True,activate_type="mish")#64 3 x 3 2 valid zero padding
    route=input_data
    route=conv_(route,(1,64),activate_type="mish")
    input_data=conv_(input_data,(1,64),activate_type="mish")
    for i in range(1):
        input_data=residual_block_a(input_data,32,64,activate_type="mish")
    input_data=conv_(input_data,(1,64),activate_type="mish")
    input_data=concatenate([input_data,route],axis=-1)
    input_data=conv_(input_data,(1,64),activate_type="mish")
    input_data=conv_(input_data,(3,128),downsample=True,activate_type="mish")
    route=input_data
    route=conv_(route,(1,64),activate_type="mish")
    input_data=conv_(input_data,(1,64),activate_type="mish")
    for i in range(2):
        input_data=residual_block_a(input_data,64,64,activate_type="mish")
    input_data=conv_(input_data,(1,64),activate_type="mish")
    input_data=concatenate([input_data,route],axis=-1)
    input_data=conv_(input_data,(1,128),activate_type="mish")
    input_data=conv_(input_data,(3,256),downsample=True,activate_type="mish")
    route=input_data
    route=conv_(route,(1,128),activate_type="mish")
    input_data=conv_(input_data,(1,128),activate_type="mish")
    for i in range(8):
        input_data=residual_block_a(input_data,128,128,activate_type="mish")
    input_data=conv_(input_data,(1,128),activate_type="mish")
    input_data=concatenate([input_data,route],axis=-1)
    input_data=conv_(input_data,(1,256),activate_type="mish")
    route_1=input_data
    input_data=conv_(input_data,(3,512),downsample=True,activate_type="mish")
    route=input_data
    route=conv_(route,(1,256),activate_type="mish")
    input_data=conv_(input_data,(1,256),activate_type="mish")
    for i in range(8):
        input_data=residual_block_a(input_data,256,256,activate_type="mish")
    input_data=conv_(input_data,(1,256),activate_type="mish")
    input_data=concatenate([input_data,route],axis=-1)
    input_data=conv_(input_data,(1,512),activate_type="mish")
    route_2=input_data
    input_data=conv_(input_data,(3,1024),downsample=True,activate_type="mish")
    route=input_data
    route=conv_(route,(1,512),activate_type="mish")
    input_data=conv_(input_data,(1,512),activate_type="mish")
    for i in range(4):
         input_data=residual_block_a(input_data,512,512,activate_type="mish")
    input_data=conv_(input_data,(1,512),activate_type="mish")
    input_data=concatenate([input_data,route],axis=-1)
    input_data=conv_(input_data,(1,1024),activate_type="mish")
    input_data=conv_(input_data,(1,512))
    input_data=conv_(input_data,(3,1024))
    input_data=conv_(input_data,(1,512))
    #----------------------cspdarknet53 end------------------------------------
    #---------------------------spp begins(spatial pyramid polling)--------------
    input_data=concatenate([tf.nn.max_pool(input_data,ksize=13,padding='SAME',strides=1),tf.nn.max_pool(input_data,ksize=9,padding='SAME',strides=1),tf.nn.max_pool(input_data,ksize=5,padding='SAME',strides=1),input_data],axis=-1)
     #---------------------- spp begins(spatial pyramid polling)-----------------------
    input_data=conv_(input_data,(1,512))
    input_data=conv_(input_data,(3,1024))
    input_data=conv_(input_data,(1,512))
    conv=input_data
    route = conv
    conv =conv_(conv, (1,256))
    conv =UpSampling2D(size=(2,2))(conv)
    route_2 = conv_(route_2, (1, 256))
    conv = concatenate([route_2, conv], axis=-1)
    conv = conv_(conv, (1,256))
    conv = conv_(conv, (3, 512))
    conv =conv_(conv, (1,256))
    conv = conv_(conv, (3,512))
    conv = conv_(conv, (1,256))
    route_2 = conv
    conv = conv_(conv, (1,128))
    conv = UpSampling2D(size=(2,2))(conv)
    #-------------------head 1 begins---------------------------------------
    route_1 = conv_(route_1, (1,128))
    conv = concatenate([route_1, conv], axis=-1)
    conv = conv_(conv, (1,128))
    conv = conv_(conv, (3,256))
    conv = conv_(conv, (1,128))
    conv = conv_(conv, (3,256))
    conv =conv_(conv, (1,128))
    route_1 = conv
    conv = conv_(conv, (3,256))
    conv_1 =conv_(conv, (1,255), activate=False, bn=False)# 255 is based for 80 classes if classes chages we have to chage 255
    #----------------------head 1 ends ,conv_1 is first output-----------------------------------
    #-------------------------head 2 begins------------------------------------------------------
    conv =conv_(route_1, (3,256), downsample=True)
    conv =concatenate([conv, route_2], axis=-1)
    conv = conv_(conv, (1,256))
    conv =conv_(conv, (3,512))
    conv = conv_(conv, (1,256))
    conv =conv_(conv, (3,512))
    conv = conv_(conv, (1,256))
    route_2 = conv
    conv = conv_(conv, (3,512))
    conv_2 = conv_(conv, (1, 255), activate=False, bn=False)
    #--------------------------head 2 ends conv_2 is 2nd output------------------------------------
    #--------------------------head 3 begins-------------------------------------------------------
    conv = conv_(route_2, (3,512), downsample=True)
    conv = concatenate([conv, route], axis=-1)
    conv = conv_(conv, (1,512))
    conv =conv_(conv, (3,1024))
    conv = conv_(conv, (1,512))
    conv = conv_(conv, (3,1024))
    conv = conv_(conv, (1, 512))
    conv = conv_(conv, (3,1024))
    conv_3 = conv_(conv, (1,255), activate=False, bn=False)
    model=Model(input_image,[conv_1,conv_2,conv_3])
    return model