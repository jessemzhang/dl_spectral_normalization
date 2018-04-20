# Can perfectly fit random labels for 20k CIFAR10 samples, but not 50k

import tensorflow as tf
import numpy as np
from .. import sn

def alexnet(NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """AlexNet architecture
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    conv = sn.conv2d(input_data, [5, 5, 3, 64], scope_name='conv1', spectral_norm=False)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    conv = sn.conv2d(norm1, [5, 5, 64, 64], scope_name='conv2', spectral_norm=False)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*64])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, NUM_CLASSES, scope_name='fc', spectral_norm=False)
        
    return input_data, input_labels, fc


def alexnet_sn(NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """AlexNet architecture with spectral normalization on all layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    conv = sn.conv2d(input_data, [5, 5, 3, 64], scope_name='conv1', update_collection=update_collection, beta=beta)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    conv = sn.conv2d(pool1, [5, 5, 64, 64], scope_name='conv2', update_collection=update_collection, beta=beta)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    
    reshape = tf.reshape(pool2, [-1, 6*6*64])
    lin = sn.linear(reshape, 384, scope_name='linear1', update_collection=update_collection, beta=beta)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', update_collection=update_collection, beta=beta)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, NUM_CLASSES, scope_name='fc', update_collection=update_collection, beta=beta)
        
    return input_data, input_labels, fc


def alexnet_sar(NUM_CLASSES, wd=0, update_collection=None, beta=1.):
    """AlexNet architecture with spectral adversarial regularization
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    conv = sn.conv2d(input_data, [5, 5, 3, 64], scope_name='conv1', update_collection=update_collection, beta=beta)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')

    conv = sn.conv2d(pool1, [5, 5, 64, 64], scope_name='conv2', update_collection=update_collection, beta=beta)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    
    reshape = tf.reshape(pool2, [-1, 6*6*64])
    lin = sn.linear(reshape, 384, scope_name='linear1', update_collection=update_collection, beta=beta)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', update_collection=update_collection, beta=beta)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, NUM_CLASSES, scope_name='fc', spectral_norm=False, wd=wd, l2_norm=True)
        
    return input_data, input_labels, fc
