# Using the filter sizes found here: 
# https://github.com/rharish101/DLGeneralization/blob/master/Mini%20Alexnet/cifar10_alexnet.py

import tensorflow as tf
import numpy as np
from .. import sn


def alexnet_sn(NUM_CLASSES, wd=0):
    """AlexNet architecture with spectral normalization on all layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', tighter_sn=True)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    conv = sn.conv2d(pool1, [5, 5, 96, 256], scope_name='conv2', tighter_sn=True)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    
    reshape = tf.reshape(pool2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1')
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2')
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, NUM_CLASSES, scope_name='fc')
        
    return input_data, input_labels, fc


def alexnet_sar(NUM_CLASSES, wd=0):
    """AlexNet architecture with spectral adversarial regularization
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', tighter_sn=True)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    conv = sn.conv2d(pool1, [5, 5, 96, 256], scope_name='conv2', tighter_sn=True)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    
    reshape = tf.reshape(pool2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1')
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2')
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, NUM_CLASSES, scope_name='fc', spectral_norm=False, wd=wd, l2_norm=True)
        
    return input_data, input_labels, fc
