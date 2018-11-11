# Using the filter sizes found here: 
# https://github.com/rharish101/DLGeneralization/blob/master/Mini%20Alexnet/cifar10_alexnet.py

import tensorflow as tf
import numpy as np
from .. import sn

def alexnet(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """

    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
        
    return fc


def alexnet_bn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with batch normalization
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """

    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse,
                     bn=True, training=training)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse,
                     bn=True, training=training)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
        
    return fc


def alexnet_dropout(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with dropout layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    dropout = 0.8

    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    norm1 = tf.nn.dropout(norm1, dropout)
    
    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    norm2 = tf.nn.dropout(norm2, dropout)
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
        
    return fc


def alexnet_wd(input_data, num_classes=10, wd=5e-4, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with weight decay
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """

    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse,
                     l2_norm=True, wd=wd)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse,
                     l2_norm=True, wd=wd)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse,
                    l2_norm=True, wd=wd)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse,
                    l2_norm=True, wd=wd)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse,
                   l2_norm=True, wd=wd)
        
    return fc


def alexnet_nolrn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """

    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    
    conv = sn.conv2d(pool1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')

    reshape = tf.reshape(pool2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
    
    return fc


def alexnet_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with spectral normalization on all layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc',
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def alexnet_miyato_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with spectral normalization on all layers
        SN is performed using Miyato's strategy (normalizing the convolutional kernel only)
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', tighter_sn=False,
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', tighter_sn=False,
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc',
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def alexnet_elu(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with spectral normalization on all layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1', spectral_norm=False, reuse=reuse)
    conv1 = tf.nn.elu(conv, name='conv1_elu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2', spectral_norm=False, reuse=reuse)
    conv2 = tf.nn.elu(conv, name='conv2_elu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1', spectral_norm=False, reuse=reuse)
    lin1 = tf.nn.elu(lin, name='linear1_elu')

    lin = sn.linear(lin1, 192, scope_name='linear2', spectral_norm=False, reuse=reuse)
    lin2 = tf.nn.elu(lin, name='linear2_elu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, reuse=reuse)
        
    return fc


def alexnet_elu_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with spectral normalization on all layers
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv1 = tf.nn.elu(conv, name='conv1_elu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv2 = tf.nn.elu(conv, name='conv2_elu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin1 = tf.nn.elu(lin, name='linear1_elu')

    lin = sn.linear(lin1, 192, scope_name='linear2',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin2 = tf.nn.elu(lin, name='linear2_elu')

    fc = sn.linear(lin2, num_classes, scope_name='fc',
                   update_collection=update_collection, beta=beta, reuse=reuse)
        
    return fc


def alexnet_snl2(input_data, num_classes, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    """AlexNet architecture with spectral normalization on all layers except last one, which
       can be L2 regularized
        two [convolution 5x5 -> max-pool 3x3 -> local-response-normalization] modules 
        followed by two fully connected layers with 384 and 192 hidden units, respectively. 
        Finally a NUM_CLASSES-way linear layer is used for prediction
    """
    
    conv = sn.conv2d(input_data, [5, 5, 3, 96], scope_name='conv1',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv1 = tf.nn.relu(conv, name='conv1_relu')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool1')
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    conv = sn.conv2d(norm1, [5, 5, 96, 256], scope_name='conv2',
                     update_collection=update_collection, beta=beta, reuse=reuse)
    conv2 = tf.nn.relu(conv, name='conv2_relu')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='VALID', name='pool2')
    norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    reshape = tf.reshape(norm2, [-1, 6*6*256])
    lin = sn.linear(reshape, 384, scope_name='linear1',
                    update_collection=update_collection, beta=beta, reuse=reuse)
    lin1 = tf.nn.relu(lin, name='linear1_relu')

    lin = sn.linear(lin1, 192, scope_name='linear2', update_collection=update_collection, beta=beta)
    lin2 = tf.nn.relu(lin, name='linear2_relu')

    fc = sn.linear(lin2, num_classes, scope_name='fc', spectral_norm=False, wd=wd, l2_norm=True, reuse=reuse)
        
    return fc
