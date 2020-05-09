import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl
import cv2
import tqdm
import h5py
import sys

def quantize_weights(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s

def batch_norm(x, mean, variance, offset=None, scale=None):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-3)


def maxpool_2d(x, k=2, s=2, padding='VALID'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],padding=padding)

def avgpool_2d(x, k=2, s=1, padding='VALID'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],padding=padding)

def get_dense_weights(weights, weight_name, bias_name='bbb', quant=True):
    w = weights[weight_name]
    if quant:
        w, s = quantize_weights(w)
        w = tf.constant(w, dtype=tf.float32)
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 1.0
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s
    
    
def denselayer(x,w,b,quant,calibrate,x_max=[],x_min=[],weight_scale=1.0,activation_scale=1., activation=''):
    
    if calibrate:
        x_max.append(tf.reduce_max(x))
        x_min.append(tf.reduce_min(x))
        x = tf.matmul(x, w)
        
    if quant:
        x, sx = quantize_tensor(x,activation_scale)
        x = tf.cast(x, dtype=tf.float32)
        x = tf.matmul(x, w)
        x = x*weight_scale
        x = x*sx
    
    x = tf.add(x, b)
    if activation == "relu":
        x = tf.nn.relu(x)
    
    return x, x_max, x_min





def quantize_conv_weights(weights,conv_quant):
    
    if conv_quant == 'per_channel':
        s = []
        for i in range(weights.shape[-1]):
            abs_weights = np.abs(weights[:,:,:,i])
            vmax = np.max(abs_weights)
            scale = vmax/127.
            s.append(scale)
        
        scales = np.array(s)
        qweights = np.divide(weights,scales)
        qweights = np.round(qweights)
        qweights = qweights.astype(np.int8)
        return qweights,scales 
    
    if conv_quant == 'per_layer':
        abs_weights = np.abs(weights)
        vmax = np.max(abs_weights)
        s = vmax / 127.
        qweights = weights / s
        qweights = np.round(qweights)
        qweights = qweights.astype(np.int8)
        return qweights, s


def get_conv_weights_biases(weights, conv_quant,weight_name, bias_name='bbb', quant=True):
    w = weights[weight_name]
    if quant:
        w, s = quantize_conv_weights(w,conv_quant)
        w = tf.constant(w, dtype=tf.float32)
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 1.0
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s




def get_bn_param(weights, mean, std, beta):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    return mean, std, beta


def quantize_tensor(x,s):
    x = tf.divide(x, s)
    x = tf.rint(x)
    x = tf.clip_by_value(x,-128.0,127.0)
    return x,s



def conv_2d(x, w, b, quant, calibrate, x_max, x_min, activation_scale,
            weight_scale=1.0, strides=1, padding='SAME', dilations=[1,1,1,1], activation=''):
    
    if calibrate:
        x_max.append(tf.reduce_max(x))
        x_min.append(tf.reduce_min(x))
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
        
    if quant:
        x, sx = quantize_tensor(x,activation_scale)
        x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
        x = x * weight_scale
        x = x*sx
    if b is not None:
        x = tf.nn.bias_add(x, b)
    
    if activation == 'relu':
        x = tf.nn.relu(x)
        
    return x, x_max, x_min


def conv2d_bn(x, quant, calibrate, x_max, x_min, activation_scale,conv_quant,layer_count, 
              weights, strides=1, padding='SAME'):
    bn_beta = 'batch_normalization_' + str(layer_count) + '/beta:0'
    bn_mean = 'batch_normalization_' + str(layer_count) + '/moving_mean:0'
    bn_var = 'batch_normalization_' + str(layer_count) + '/moving_variance:0'
    conv_name = 'conv2d_' + str(layer_count) + '/kernel:0'
    bias_name = 'conv2d_' + str(layer_count) + '/bias:0'

    layer_count += 1
            
    w, b, s = get_conv_weights_biases(weights, conv_quant,conv_name, bias_name,quant)
    
    x, x_max, x_min = conv_2d(x, w, b, quant, calibrate, x_max, x_min, activation_scale, s, strides=strides, padding=padding)

    mean, std, beta = get_bn_param(weights, bn_mean, bn_var, bn_beta)
    x = batch_norm(x, mean, std, beta)
    x = tf.nn.relu(x)
    return x, layer_count, x_max, x_min


def InceptionV3(img_input, weights, quant, calibrate, activation_scales,conv_quant):
    
    x_max,x_min = [], []
    
    layer_count = 1

    x = tf.reshape(img_input, shape=[-1, 299, 299, 3])

    x,layer_count,x_max,x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,activation_scales[layer_count-1] ,conv_quant,
                                          layer_count,weights,strides=2,padding='VALID')
    
    x,layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],conv_quant,
                                            layer_count, weights, strides=1, padding='VALID')
    
    x, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,activation_scales[layer_count-1], conv_quant,
                                             layer_count, weights)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    x, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],conv_quant, 
                                             layer_count, weights, strides=1, padding='VALID')
    x, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],conv_quant,
                                             layer_count, weights, strides=1, padding='VALID')
    
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch5x5, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],
                                                     conv_quant,layer_count, weights)
    branch5x5, layer_count, x_max, x_min = conv2d_bn(branch5x5, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)
    
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 1: 35 x 35 x 256
    branch1x1, layer_count , x_max, x_min= conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    branch5x5, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1] ,conv_quant,layer_count, weights)
    
    branch5x5, layer_count, x_max, x_min = conv2d_bn(branch5x5, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1] ,conv_quant,layer_count, weights)

    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 2: 35 x 35 x 256
    branch1x1, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch5x5, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)
    branch5x5, layer_count, x_max, x_min = conv2d_bn(branch5x5, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1],conv_quant,layer_count, weights)
    x = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # mixed 3: 17 x 17 x 768
    branch3x3, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],
                                                     conv_quant,layer_count, weights, strides=2, padding='VALID')

    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant, layer_count, weights)
    branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count,
                                                        weights, strides=2, padding='VALID')

    branch_pool = maxpool_2d(x, k=3, s=2, padding='VALID')
    x = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis=3)

    # mixed 4: 17 x 17 x 768
    branch1x1, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch7x7, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant, layer_count, weights)
    branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)
    branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    
    branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1],conv_quant,layer_count, weights)
    x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1], conv_quant,layer_count, weights)

        branch7x7, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1], conv_quant,layer_count, weights)
        
        branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1],conv_quant,layer_count, weights)

        branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1], conv_quant,layer_count, weights)
        
        branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1], conv_quant,layer_count, weights)
        
        branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1], conv_quant,layer_count, weights)
        
        branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1], conv_quant,layer_count, weights)

        branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
        
        branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                           activation_scales[layer_count-1],conv_quant,layer_count, weights)
        x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 7: 17 x 17 x 768
    branch1x1, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1], conv_quant,layer_count, weights)

    branch7x7, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch7x7, layer_count, x_max, x_min = conv2d_bn(branch7x7, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)

    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant, layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7dbl, layer_count, x_max, x_min = conv2d_bn(branch7x7dbl,quant,calibrate,x_max,x_min,
                                                        activation_scales[layer_count-1],conv_quant, layer_count, weights)

    branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
    branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1],conv_quant,layer_count, weights)
    x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3)

    # mixed 8: 8 x 8 x 1280
    branch3x3, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                     activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch3x3, layer_count, x_max, x_min = conv2d_bn(branch3x3, quant,calibrate,x_max,x_min,activation_scales[layer_count-1],
                                                     conv_quant,layer_count, weights, strides=2, padding='VALID')

    branch7x7x3, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1],conv_quant,layer_count, weights)
    
    branch7x7x3, layer_count, x_max, x_min = conv2d_bn(branch7x7x3,quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7x3, layer_count, x_max, x_min = conv2d_bn(branch7x7x3,quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1], conv_quant,layer_count, weights)
    
    branch7x7x3, layer_count, x_max, x_min = conv2d_bn(branch7x7x3,quant,calibrate,x_max,x_min,
                                                       activation_scales[layer_count-1], conv_quant,
                                                       layer_count, weights, strides=2, padding='VALID')

    branch_pool = maxpool_2d(x, k=3, s=2, padding='VALID')
    x = tf.concat([branch3x3, branch7x7x3, branch_pool], axis=3)

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1, layer_count , x_max, x_min= conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1],conv_quant,layer_count, weights)

        branch3x3, layer_count, x_max, x_min = conv2d_bn(x, quant,calibrate,x_max,x_min,
                                                         activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3_1, layer_count, x_max, x_min = conv2d_bn(branch3x3, quant,calibrate,x_max,x_min,
                                                           activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3_2, layer_count, x_max, x_min = conv2d_bn(branch3x3, quant,calibrate,x_max,x_min,
                                                           activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3 = tf.concat([branch3x3_1, branch3x3_2], axis=3)

        branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(x,quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1], conv_quant,layer_count, weights)
        
        branch3x3dbl, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                            activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3dbl_1, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl,quant,calibrate,x_max,x_min,
                                                              activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3dbl_2, layer_count, x_max, x_min = conv2d_bn(branch3x3dbl, quant,calibrate,x_max,x_min,
                                                              activation_scales[layer_count-1],conv_quant,layer_count, weights)
        
        branch3x3dbl = tf.concat([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = avgpool_2d(x, k=3, s=1, padding='SAME')
        branch_pool, layer_count, x_max, x_min = conv2d_bn(branch_pool, quant,calibrate,x_max,x_min,
                                                           activation_scales[layer_count-1],conv_quant,layer_count, weights)
        x = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3)

    x = avgpool_2d(x, k=8)
    
    
    w, b, s = get_dense_weights(weights, 'predictions/kernel:0', 'predictions/bias:0', quant)
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    
    x, x_max, x_min = denselayer(x, w, b, quant,calibrate,x_max, x_min, s, activation_scales[layer_count-1])
    
    print(layer_count)
    
    if calibrate:
        return x_max,x_min
    
    else:
        return x

   
    
def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


def weight_loader(weight_file):
    weights = {}
    f = h5py.File(weight_file, mode='r')
    try:
        layers = f.attrs['layer_names']
    except:
        raise ValueError("weights file must contain attribution: 'layer_names'")
    for layer_name in layers:
        g = f[layer_name]
        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name].value
            name = str(weight_name).split("'")[1]
            weights[name] = weight_value
    return weights

def generate_global_max_min():
    global_max, global_min = [],[]
    for i in range(500):
        global_max.append(float("-inf"))
        global_min.append(float("inf"))
    return global_max,global_min 

def collect_stats(all_global_max,all_global_min,max_x, min_x):
    
    all_global_max.append(max_x)
    all_global_min.append(min_x)
    
    return all_global_max, all_global_min
    

def get_final_max_and_min(all_global_max, all_global_min, method= 'absolute'):
    
    if method == 'absolute':
        global_max, global_min = [], []

        d_max = np.array([float("-inf") for i in all_global_max[0]])
        d_min = np.array([float("inf")  for i in all_global_min[0]])

        for j in range(len(all_global_max[0])):
            for i in range(len(all_global_max)):
                if d_max[j]<all_global_max[i][j]:
                    d_max[j]=all_global_max[i][j]
                if d_min[j]>all_global_min[i][j]:
                    d_min[j]=all_global_min[i][j]
    
        return d_max, d_min
 

    if method == 'average':
        
        max_sum = np.array([0 for i in all_global_max[0]])
        min_sum = np.array([0 for i in all_global_min[0]])
        
        for i in range(len(all_global_max)):
            max_sum = max_sum + np.array(all_global_max[i])
            min_sum = min_sum + np.array(all_global_min[i])
        
        global_max, global_min = max_sum/(i+1), min_sum/(i+1) 
            
        return global_max,global_min


def get_scales(global_max, global_min, threshold):
    scales = []
    for i in range(global_max.size):
        abs_value = max(threshold*(np.abs(global_max[i])),threshold*(np.abs(global_min[i])))
        s = np.divide(abs_value, 127.)
        scales.append(s)
    return scales

parse = argparse.ArgumentParser(description='Command for quantization models')
parse.add_argument('--samples',type=int,default=100,help='No. of calibration data samples')
parse.add_argument('--calib_method',type=str,default='average',help='Method to find max/min')
parse.add_argument('--conv_quant_method', type=str, default = 'per_layer', help='conv quant method')
parse.add_argument('--threshold', type=float, default = 1.00, help='conv quant method')
args = parse.parse_args()

weights = {'inception': 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'}
weights = weight_loader('/Weights/{}'.format(weights['inception']))

global_max, global_min = generate_global_max_min()


X = tf.placeholder(tf.float32, [None, 299, 299, 3])
Y = tf.placeholder(tf.float32, [None, 1000])


def image_loader(pkl_file, model='vgg', dtype='float32'):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    f.close()
    for im, target in tqdm.tqdm(zip(data['data'], data['target']), total=50000):
        im = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im, (299, 299))
        im = im.astype(dtype)
        im = np.expand_dims(im, axis=0)
        im /= 255.
        im -= 0.5
        im *= 2.
        label = int(target)
        yield im, label



acc = 0.
acc_top5 = 0.



with tf.device('/gpu:0'):
    max_x1, min_x1 = InceptionV3(X, weights, False, True,global_max,args.conv_quant_method)

with tf.Session() as sess:
    print('start calibrating')
    
    all_global_max, all_global_min = [], [] 
    
    i=0
    
    for im, label in image_loader('/Data/val224_compressed.pkl'):
        max_x, min_x = sess.run([max_x1, min_x1],feed_dict={X: im})
        all_global_max, all_global_min = collect_stats(all_global_max,all_global_min,max_x, min_x)
        i = i+1
        if i==args.samples:
            break
    
    print('done calibrating')
    
    all_global_max, all_global_min = get_final_max_and_min(all_global_max, all_global_min,args.calib_method)
    scales = get_scales(all_global_max, all_global_min, args.threshold)
    
    with tf.device('/gpu:0'):
        logits = InceptionV3(X, weights, True, False, scales,args.conv_quant_method)
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)

    for im, label in image_loader('/Data/val224_compressed.pkl'):   
        t1, t5 = sess.run([pred, prediction], feed_dict={X: im})
        if t1[0] == label:
            acc += 1
        if label in top5_acc(t5[0].tolist()):
            acc_top5 += 1
             
    print('Top1 accuracy of Inception_calibration_127: {}'.format(acc / 50000))
    print('Top5 accuracy of Inception_calibration_127: {}'.format(acc_top5 / 50000))
    
    write_list = ["conv quant method= ","accuracy_top_1= ","accuracy_top_5= ", 'No. of samples= ',
                  "Calibration method= ", "Threshold = "]
    write_values = [args.conv_quant_method,str(acc/(50000)),str(acc_top5 /(50000)),str(args.samples),
                        args.calib_method, str(args.threshold)]
        
        
    with open("samplvsacc_cpu.txt", "a") as myfile:
        for items in range(len(write_list)):
            myfile.write(write_list[items])
            myfile.write(write_values[items])
            myfile.write("\n")
            print(write_list[items],write_values[items])
        myfile.write("----------------------------------------------------------------------------------------------")
        myfile.write("\n")
            

