import argparse
import tensorflow as tf
import numpy as np
import pickle as pkl
import cv2
import tqdm
import h5py, sys

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.33

def quantize_tensor(x,s):
    x = tf.divide(x, s)
    x = tf.rint(x)
    x = tf.clip_by_value(x,-128.0,127.0)
    return x,s

def quantize_dense_weights(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s

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

def get_dense_weights_biases(weights, quant,weight_name, bias_name='bbb'):
    w = weights[weight_name]
    if quant:
        w, s = quantize_dense_weights(w)
        w = tf.constant(w, dtype=tf.float32)
        
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 1.0
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s


def get_conv_weights_biases(weights,conv_quant,quant,weight_name, bias_name='bbb'):
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


def separable_conv2d(x, dw, pw, quant,calibrate,act_scale,x_max=[],x_min=[], 
                     dw_scale=1.0, pw_scale=1.0, strides=1, padding='SAME', activation=''):
    
    if calibrate:
        x_max.append(tf.reduce_max(x))
        x_min.append(tf.reduce_min(x))
        x = tf.nn.separable_conv2d(x, dw, pw, strides=[1, strides, strides, 1], padding=padding)
        if activation == 'relu':
            x = tf.nn.relu(x)
        
    if quant:
        x, sx = quantize_tensor(x,act_scale)
        x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.separable_conv2d(x, dw, pw, strides=[1, strides, strides, 1], padding=padding)
        x = x * sx * dw_scale * pw_scale
        if activation == 'relu':
            x = tf.nn.relu(x)
    
    return x, x_max, x_min

def conv_2d(x, w, b, quant, calibrate, act_scale, x_max, x_min,weight_scale=1.0, strides=1, 
            padding='SAME', dilations=[1,1,1,1], activation=''):
    
    print("Inside conv_2d")
    if calibrate:
        x_max.append(tf.reduce_max(x))
        x_min.append(tf.reduce_min(x))
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
        if b is not None:
            x = tf.nn.bias_add(x, b)
        if activation == 'relu':
            x = tf.nn.relu(x)
        
    if quant:
        x, sx = quantize_tensor(x,act_scale)
        x = tf.cast(x, dtype=tf.float32)
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
        s = sx * weight_scale
        x = x * s
        if b is not None:
            x = tf.nn.bias_add(x, b)
        if activation == 'relu':
            x = tf.nn.relu(x)
            
    return x,  x_max, x_min


def denselayer(x, w, b, weight_scale, x_max,x_min,layer,quant,calibrate,act_scales, activation=''):
    
    if calibrate:
        x_max.append(tf.reduce_max(x))
        x_min.append(tf.reduce_min(x))
        x = tf.matmul(x, w)
        x = tf.add(x, b)
        if activation == "relu":
            x = tf.nn.relu(x)
    
    if quant:
        x, sx = quantize_tensor(x, act_scales)
        x = tf.cast(x, dtype=tf.float32)
        x = tf.matmul(x, w)
        s = sx * weight_scale
        x = x * s
        x = tf.add(x, b)
        if activation == "relu":
            x = tf.nn.relu(x)
    layer = layer + 1
    return x,layer, x_max, x_min


def batch_norm(x, mean, variance, offset=None, scale=None):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-3)

def maxpool_2d(x, k=2, s=2, padding='VALID'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


def avgpool_2d(x, k=2, s=1, padding='VALID'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],
                          padding=padding)

def get_bn_param(weights, layer_num):
    mean = 'batchnormalization_' + str(layer_num) + '_running_mean:0'
    std = 'batchnormalization_' + str(layer_num) + '_running_std:0'
    beta = 'batchnormalization_' + str(layer_num) + '_beta:0'
    gamma = 'batchnormalization_' + str(layer_num) + '_gamma:0'

    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    gamma = tf.constant(weights[gamma], dtype=tf.float32)
    return mean, std, beta, gamma


def conv_block(x, weights, conv_num, bn_num,x_max, x_min,layer, quant,calibrate,conv_quant,act_scale,strides=1, 
               padding='SAME',activation=True):
    
    print("Inside Conv block")
    conv_name = 'convolution2d_{}_W:0'.format(conv_num)
    bias_name = 'convolution2d_{}_b:0'.format(conv_num)
    
    w, b, s = get_conv_weights_biases(weights, conv_quant, quant, conv_name, bias_name)
    x,x_max, x_min = conv_2d(x,w,None,quant,calibrate,act_scale,x_max,x_min,s,strides=strides,padding=padding)
    mean, std, beta, gamma = get_bn_param(weights, bn_num)
    x = batch_norm(x, mean, std, beta, gamma)
    if activation:
        x = tf.nn.relu(x)
    conv_num += 1
    bn_num += 1
    layer = layer + 1
    
    print("Coming out of Conv block")
    
    
    return x, conv_num, bn_num,layer, x_max, x_min


    

def separable_conv_block(x, weights, sep_num, bn_num, x_max, x_min, layer, quant, calibrate, conv_quant, act_scale,
                         strides=1, padding='SAME', activation=True):
    
    print("Inside separable block")
    
    
    dw_name = 'separableconvolution2d_{}_depthwise_kernel:0'.format(sep_num)
    pw_name = 'separableconvolution2d_{}_pointwise_kernel:0'.format(sep_num)

    dw, b, ds = get_conv_weights_biases(weights, conv_quant, quant, dw_name)
    pw, b, ps = get_conv_weights_biases(weights, conv_quant, quant, pw_name)

    x,x_max, x_min = separable_conv2d(x,dw,pw,quant,calibrate,act_scale,x_max,x_min,ds,ps,strides=strides, padding=padding)
    
    mean, std, beta, gamma = get_bn_param(weights, bn_num)
    x = batch_norm(x, mean, std, beta, gamma)
    if activation:
        x = tf.nn.relu(x)
    sep_num += 1
    bn_num += 1
    layer = layer + 1
    
    print("coming out of separable block")
    
    return x, sep_num, bn_num,layer, x_max, x_min 


def Xception(x,weights,quant,calibrate, conv_quant,scales):
    bn_count = 1
    conv_count = 1
    sepconv_count = 1
    x = tf.reshape(x, shape=[-1, 299, 299, 3])
    layer = 0
    x_max,x_min = [],[]
    
    print("Inside Xception")
    
    x, conv_count, bn_count ,layer, x_max, x_min= conv_block(x, weights, conv_count, bn_count, x_max, x_min,layer,quant,
                                                             calibrate,conv_quant, scales[layer],strides=2, padding='VALID')
    
    x, conv_count, bn_count ,layer, x_max, x_min = conv_block(x, weights, conv_count, bn_count, x_max,
                                                              x_min,layer,quant,calibrate,
                                                              conv_quant, scales[layer],strides=1, padding='VALID')
    
    
    residual, conv_count, bn_count ,layer, x_max, x_min= conv_block(x, weights, conv_count, bn_count,x_max, x_min,layer,quant,
                                                                    calibrate,conv_quant,scales[layer],
                                                                    strides=2, activation=False)

    x, sepconv_count, bn_count ,layer, x_max, x_min = separable_conv_block(x, weights, sepconv_count, 
                                                                           bn_count,x_max, x_min,layer,
                                                                           quant,calibrate,conv_quant,scales[layer])
    
    
    x, sepconv_count, bn_count ,layer, x_max, x_min = separable_conv_block(x, weights, sepconv_count, bn_count,x_max,
                                                                           x_min,layer,quant,calibrate,conv_quant,
                                                                           scales[layer],activation=False)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')
    x = tf.add(x, residual)

    residual, conv_count, bn_count,layer, x_max, x_min = conv_block(x, weights, conv_count, bn_count,x_max, x_min,layer,quant,
                                                                    calibrate,conv_quant,scales[layer],
                                                                    strides=2, activation=False)

    x = tf.nn.relu(x)
    x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x, weights, sepconv_count, bn_count,
                                                                          x_max, x_min,layer,
                                                                          quant,calibrate,conv_quant,scales[layer])
    
    x, sepconv_count, bn_count ,layer, x_max, x_min = separable_conv_block(x, weights, sepconv_count, bn_count,
                                                                           x_max, x_min,layer,quant,calibrate,conv_quant,
                                                                           scales[layer],activation=False)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')
    x = tf.add(x, residual)

    residual, conv_count, bn_count ,layer, x_max, x_min= conv_block(x, weights, conv_count, bn_count,x_max, x_min,layer,
                                                                    quant, calibrate,conv_quant,scales[layer],strides=2,
                                                                    activation=False)

    x = tf.nn.relu(x)
    x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x, weights, sepconv_count, bn_count,x_max, x_min,
                                                                          layer,quant,calibrate,
                                                                          conv_quant,scales[layer])
    
    x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x, weights, sepconv_count, bn_count, x_max, x_min,
                                                                          layer,quant,calibrate,conv_quant,
                                                                          scales[layer],activation=False)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')
    x = tf.add(x, residual)

    for i in range(8):
        residual = x
        x = tf.nn.relu(x)
        x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x, weights, sepconv_count,bn_count,x_max, x_min,
                                                                              layer,quant,calibrate,
                                                                              conv_quant,scales[layer])
        
        x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x,weights,sepconv_count,bn_count,x_max,
                                                                              x_min,layer,
                                                                              quant,calibrate,
                                                                              conv_quant,scales[layer])
        
        x,sepconv_count,bn_count,layer,x_max,x_min=separable_conv_block(x,weights,
                                                                        sepconv_count,bn_count,x_max,x_min,
                                                                        layer,quant,
                                                                        calibrate,conv_quant,scales[layer],activation=False)

        x = tf.add(x, residual)

    residual, conv_count, bn_count ,layer, x_max, x_min= conv_block(x,weights,conv_count,bn_count,x_max,x_min,layer,quant,
                                                                    calibrate,conv_quant,scales[layer],strides=2,
                                                                    activation=False)

    x = tf.nn.relu(x)
    x, sepconv_count, bn_count,layer, x_max, x_min = separable_conv_block(x,weights,sepconv_count,bn_count,x_max, x_min,layer,
                                                                          quant,calibrate,conv_quant,
                                                                          scales[layer])
    
    x, sepconv_count, bn_count ,layer, x_max, x_min= separable_conv_block(x,weights,sepconv_count,bn_count,x_max,x_min,
                                                                          layer, quant,calibrate,conv_quant,scales[layer],
                                                                          activation=False)
    
    x = maxpool_2d(x, k=3, s=2, padding='SAME')
    x = tf.add(x, residual)

    x, sepconv_count, bn_count ,layer, x_max, x_min = separable_conv_block(x, weights, sepconv_count, bn_count,x_max,
                                                                           x_min,layer,
                                                                           quant,calibrate,conv_quant,scales[layer])
    
    x, sepconv_count, bn_count ,layer, x_max, x_min = separable_conv_block(x, weights, sepconv_count,bn_count,x_max,
                                                                           x_min,layer,
                                                                           quant,calibrate,conv_quant,scales[layer])

    x = avgpool_2d(x, k=8)

    
    w, b, s = get_dense_weights_biases(weights,quant, 'dense_2_W:0', 'dense_2_b:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    
    print(w.shape)
    print(tf.shape(x))
    x ,layer, x_max, x_min= denselayer(x, w, b, s,x_max, x_min,layer,quant,calibrate,scales[layer])
    
    if calibrate:
        return x_max,x_min
    
    else:
        return x


def weight_loader(weight_file):
    weights = {}
    f = h5py.File(weight_file, mode='r')
    # f = f['model_weights']
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



weights = {'xception': 'xception_weights_tf_dim_ordering_tf_kernels.h5'}


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

def collect_stats(all_global_max,all_global_min,max_x, min_x):
    
    all_global_max.append(max_x)
    all_global_min.append(min_x)
    
    return all_global_max, all_global_min

def generate_global_max_min():
    global_max, global_min = [],[]
    for i in range(54):
        global_max.append(float("-inf"))
        global_min.append(float("inf"))
    return global_max,global_min


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


def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


if __name__ == '__main__':
    
    parse = argparse.ArgumentParser(description='Command for quantization models')
    parse.add_argument('--samples',type=int,default=3,help='No. of calibration data samples')
    parse.add_argument('--calib_method',type=str,default='average',help='Method to find max/min')
    parse.add_argument('--conv_quant_method', type=str, default = 'per_layer', help='conv quant method')
    parse.add_argument('--threshold', type=float, default = 1.00, help='conv quant method')
    args = parse.parse_args()

    weights = weight_loader('/work/arun/krishnat/AMD_Quantization/weights/{}'.format(weights['xception']))
    
    X = tf.placeholder(tf.float32, [None, 299, 299, 3])
    Y = tf.placeholder(tf.float32, [None, 1000])

    data_file = '/work/arun/krishnat/AMD_Quantization/data/val224_compressed.pkl' 
    global_max, global_min = generate_global_max_min()
    
    
    with tf.device('/gpu:0'):
        max_x1, min_x1 = Xception(X, weights, False, True,args.conv_quant_method,global_max)
    
    with tf.Session() as sess:
        print("Start calibarting")
        
        all_global_max, all_global_min = [], [] 
        
        i=0
        
        for im, label in image_loader(data_file):
            max_x, min_x = sess.run([max_x1, min_x1],feed_dict={X: im})
            all_global_max, all_global_min = collect_stats(all_global_max,all_global_min,max_x, min_x)
            i = i+1
            if i==args.samples:
                break
        
        print('Done calibrating')
        
        all_global_max, all_global_min = get_final_max_and_min(all_global_max, all_global_min,args.calib_method)
        scales = get_scales(all_global_max, all_global_min, args.threshold)
        
        with tf.device('/gpu:0'):
            logits = Xception(X, weights, True, False,args.conv_quant_method, scales)
            prediction = tf.nn.softmax(logits)
            pred = tf.argmax(prediction, 1)
        
        print("Start Evaluating")
        acc = 0.
        acc_top5 = 0.
        for im, label in image_loader(data_file):   
            t1, t5 = sess.run([pred, prediction], feed_dict={X: im})
            if t1[0] == label:
                acc += 1
            if label in top5_acc(t5[0].tolist()):
                acc_top5 += 1
             
        print('Top1 accuracy of vgg16_calibration_127: {}'.format(acc / 50000))
        print('Top5 accuracy of vgg16_calibration_127: {}'.format(acc_top5 / 50000))
    
        write_list = ["conv quant method= ","accuracy_top_1= ","accuracy_top_5= ", 'No. of samples= ',
                  "Calibration method= ", "Threshold = "]

        write_values = [args.conv_quant_method,str(acc/(50000)),str(acc_top5 /(50000)),str(args.samples),
                        args.calib_method, str(args.threshold)]

        with open("avs_abs.txt", "a") as myfile:
            for items in range(len(write_list)):
                myfile.write(write_list[items])
                myfile.write(write_values[items])
                myfile.write("\n")
                print(write_list[items],write_values[items])
            myfile.write("----------------------------------------------------------------------------------------------")
            myfile.write("\n")
        
    
    
