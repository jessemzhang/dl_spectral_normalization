# Basing code off the code here: 
# https://github.com/huyng/tensorflow-vgg/blob/master/vgg.py
#
# Using filter sizes from "Exploring Generalization in Deep Learning"
# https://arxiv.org/abs/1706.08947

import tensorflow as tf
import numpy as np
from .. import sn

def vgg(NUM_CLASSES, wd=0):
    """VGG architecture"""

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1', bn=True, xavier=True, spectral_norm=False))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', bn=True, xavier=True, spectral_norm=False))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4', bn=True, xavier=True, spectral_norm=False))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5', bn=True, xavier=True, spectral_norm=False))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7', bn=True, xavier=True, spectral_norm=False))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8', bn=True, xavier=True, spectral_norm=False))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10', bn=True, xavier=True, spectral_norm=False))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11', bn=True, xavier=True, spectral_norm=False))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True, spectral_norm=False)
    fc = sn.linear(layer14, NUM_CLASSES, scope_name='fc', xavier=True, spectral_norm=False)

    return input_data, input_labels, fc


def vgg_sn(NUM_CLASSES, wd=0):
    """VGG architecture with spectral normalization on all layers"""
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1', xavier=True, tighter_sn=True))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', xavier=True, tighter_sn=True))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4', xavier=True, tighter_sn=True))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5', xavier=True, tighter_sn=True))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7', xavier=True, tighter_sn=True))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8', xavier=True, tighter_sn=True))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10', xavier=True, tighter_sn=True))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11', xavier=True, tighter_sn=True))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True)
    fc = sn.linear(layer14, NUM_CLASSES, scope_name='fc', xavier=True)

    return input_data, input_labels, fc


def vgg_sar(NUM_CLASSES, wd=0):
    """VGG architecture with spectral adversarial regularization"""
    
    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')
    
    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1', xavier=True, tighter_sn=True))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', xavier=True, tighter_sn=True))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4', xavier=True, tighter_sn=True))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5', xavier=True, tighter_sn=True))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7', xavier=True, tighter_sn=True))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8', xavier=True, tighter_sn=True))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10', xavier=True, tighter_sn=True))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11', xavier=True, tighter_sn=True))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True)
    fc = sn.linear(layer14, NUM_CLASSES, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True)

    return input_data, input_labels, fc
