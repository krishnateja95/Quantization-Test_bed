import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl
import cv2
import tqdm
import h5py
import sys

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"

def maxpool_2d(x, k=2, s=2, padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],padding=padding)

def avgpool_2d(x, k=2, s=1, padding='VALID'):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],padding=padding)


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





def quantize_tensor(x,s):
    x = tf.divide(x, s)
    x = tf.rint(x)
    x = tf.clip_by_value(x,-128.0,127.0)
    return x,s


def conv_2d(x, w, b, quant, calibrate, x_max, x_min, activation_scale,layer_count,
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
    
    layer_count += 1
    
    return x, x_max, x_min, layer_count


def fire_module(x, weights, quant, calibrate, scales, conv_quant_method,x_max, x_min, layer_count,fire_id):
    s_id = 'fire' + str(fire_id) + '/'

    w, b, s = get_conv_weights_biases(weights, conv_quant_method, s_id + sq1x1 + '_W:0', s_id + sq1x1 + '_b:0',quant)
    x, x_max, x_min, layer_count = conv_2d(x, w, b, quant,calibrate,x_max,x_min,scales[layer_count],
                                           layer_count,s, strides=1, padding='VALID', activation='relu')

    w, b, s = get_conv_weights_biases(weights, conv_quant_method,s_id + exp1x1 + '_W:0', s_id + exp1x1 + '_b:0',quant)
    left, x_max, x_min, layer_count = conv_2d(x, w, b, quant,calibrate,x_max,x_min,scales[layer_count],
                                              layer_count,s, strides=1, padding='VALID', activation='relu')

    w, b, s = get_conv_weights_biases(weights,conv_quant_method, s_id + exp3x3 + '_W:0', s_id + exp3x3 + '_b:0',quant)
    right, x_max, x_min, layer_count = conv_2d(x, w, b, quant,calibrate,x_max,x_min,scales[layer_count],
                                               layer_count,s, strides=1, padding='SAME', activation='relu')

    x = tf.concat([left, right], axis=3)
    return x, x_max, x_min, layer_count



def SqueezeNet(x, weights, quant, calibrate, scales, conv_quant_method):
    
    x_max,x_min = [], []
    
    layer_count = 0
    
    
    x = tf.reshape(x, shape=[-1, 227, 227, 3])

    w, b, s = get_conv_weights_biases(weights,conv_quant_method, 'conv1_W:0', 'conv1_b:0',quant)
    x, x_max, x_min, layer_count = conv_2d(x,w,b,quant,calibrate,x_max,x_min,scales[layer_count],
                                           layer_count,s,strides=2, padding='VALID', activation='relu')
    x = maxpool_2d(x, k=3, s=2)

    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=2)
    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=3)
    x = maxpool_2d(x, k=3, s=2)

    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=4)
    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=5)
    x = maxpool_2d(x, k=3, s=2)

    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=6)
    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=7)
    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=8)
    x,x_max,x_min,layer_count=fire_module(x,weights,quant,calibrate,scales,conv_quant_method,x_max,x_min,layer_count,fire_id=9)

    w, b, s = get_conv_weights_biases(weights,conv_quant_method, 'conv10_W:0', 'conv10_b:0',quant)
    x, x_max, x_min, layer_count = conv_2d(x, w, b, quant,calibrate,x_max,x_min,scales[layer_count],
                                           layer_count,s, strides=1, padding='VALID', activation='relu')

    x = avgpool_2d(x, k=13)
    x = tf.reshape(x, shape=[-1, 1000])
    
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

weights = {'squeezenet': 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5'}
weights = weight_loader('/Weights/{}'.format(weights['squeezenet']))

global_max, global_min = generate_global_max_min()


X = tf.placeholder(tf.float32, [None, 227, 227, 3])
Y = tf.placeholder(tf.float32, [None, 1000])


def image_loader(pkl_file, model='vgg', dtype='float32'):
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
    f.close()
    for im, target in tqdm.tqdm(zip(data['data'], data['target']), total=50000):
        im = cv2.imdecode(np.fromstring(im, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (227, 227))
        im = im.astype(dtype)
        im = np.expand_dims(im, axis=0)
        im[..., 0] -= 103.939
        im[..., 1] -= 116.779
        im[..., 2] -= 123.68
        label = int(target)
        yield im, label


acc = 0.
acc_top5 = 0.


with tf.device('/gpu:0'):
    max_x1, min_x1 = SqueezeNet(X, weights, False, True,global_max,args.conv_quant_method)

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
        logits = SqueezeNet(X, weights, True, False, scales,args.conv_quant_method)
        prediction = tf.nn.softmax(logits)
        pred = tf.argmax(prediction, 1)
    
    for im, label in image_loader('/Data/val224_compressed.pkl'):   
        t1, t5 = sess.run([pred, prediction], feed_dict={X: im})
        if t1[0] == label:
            acc += 1
        if label in top5_acc(t5[0].tolist()):
            acc_top5 += 1
             
    print('Top1 accuracy of Squeezenet_calibration_127: {}'.format(acc / 50000))
    print('Top5 accuracy of Squeezenet_calibration_127: {}'.format(acc_top5 / 50000))
    
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
            

