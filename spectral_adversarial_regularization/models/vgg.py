# Basing code off the code here: 
# https://github.com/huyng/tensorflow-vgg/blob/master/vgg.py
#
# Using filter sizes from "Exploring Generalization in Deep Learning"
# https://arxiv.org/abs/1706.08947

import tensorflow as tf
import numpy as np
from .. import sn

def vgg(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """VGG architecture"""

    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1',
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', 
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4',
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5',
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7',
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8',
                                  bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10',
                                   bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11',
                                   bn=True, xavier=True, spectral_norm=False, reuse=reuse))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True, spectral_norm=False, reuse=reuse)
    fc = sn.linear(layer14, num_classes, scope_name='fc', xavier=True, spectral_norm=False, reuse=reuse)

    return fc


def vgg_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """VGG architecture with spectral normalization on all layers"""
    
    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10', bn=True, xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11', bn=True, xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True,
                        update_collection=update_collection, beta=beta, reuse=reuse)
    fc = sn.linear(layer14, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse)

    return fc


def vgg_sar(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None):
    """VGG architecture with spectral adversarial regularization"""
    
    layer1 = tf.nn.relu(sn.conv2d(input_data, [3, 3, 3, 64], scope_name='conv1', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer2 = tf.nn.relu(sn.conv2d(layer1, [3, 3, 64, 64], scope_name='conv2', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    # layer3 = tf.nn.dropout(layer3, 0.5, name='dropout3')
    
    layer4 = tf.nn.relu(sn.conv2d(layer3, [3, 3, 64, 128], scope_name='conv4', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer5 = tf.nn.relu(sn.conv2d(layer4, [3, 3, 128, 128], scope_name='conv5', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer6 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool6')
    # layer6 = tf.nn.dropout(layer6, 0.5, name='dropout6')
    
    layer7 = tf.nn.relu(sn.conv2d(layer6, [3, 3, 128, 256], scope_name='conv7', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer8 = tf.nn.relu(sn.conv2d(layer7, [3, 3, 256, 256], scope_name='conv8', bn=True, xavier=True,
                                  update_collection=update_collection, beta=beta, reuse=reuse))
    layer9 = tf.nn.max_pool(layer8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool9')
    # layer9 = tf.nn.dropout(layer9, 0.5, name='dropout9')
    
    layer10 = tf.nn.relu(sn.conv2d(layer9, [3, 3, 256, 512], scope_name='conv10', bn=True, xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    layer11 = tf.nn.relu(sn.conv2d(layer10, [3, 3, 512, 512], scope_name='conv11', bn=True, xavier=True,
                                   update_collection=update_collection, beta=beta, reuse=reuse))
    layer12 = tf.nn.max_pool(layer11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool12')
    # layer12 = tf.nn.dropout(layer12, 0.5, name='dropout12')
    
    layer13 = tf.nn.pool(layer12, window_shape=[4, 4], pooling_type='AVG', 
                         padding='SAME', strides=[1, 1], name='mean_pool13')
    
    layer14 = sn.linear(layer13, 512, scope_name='linear14', xavier=True,
                        update_collection=update_collection, beta=beta, reuse=reuse)
    fc = sn.linear(layer14, num_classes, scope_name='fc', spectral_norm=False,
                   xavier=True, wd=wd, l2_norm=True, reuse=reuse)

    return fc
