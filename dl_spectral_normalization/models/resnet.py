# Based off the code found here (using the same architecture): 
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py

import tensorflow as tf
import numpy as np
from .. import sn


def batch_norm(inputs, training):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
      inputs=inputs, axis=3, momentum=0.997, epsilon=1e-5, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name='conv2d', **kwargs):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    in_size = inputs.get_shape().as_list()[-1]
    return sn.conv2d(inputs, [kernel_size, kernel_size, in_size, filters], stride=strides,
                     padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                     xavier=False, variance_scaling=True, bn=False,
                     scope_name=name, **kwargs)


def block_fn(inputs, filters, training, projection_shortcut, strides, name, bn=True, **kwargs):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shortcut = inputs
        if bn:
            inputs = tf.layers.batch_normalization(inputs, training=training)
        inputs = tf.nn.relu(inputs)

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = conv2d_fixed_padding(inputs, filters, 3, strides, name='conv1', **kwargs)
        if bn:
            inputs = batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = conv2d_fixed_padding(inputs, filters, 3, 1, name='conv2', **kwargs)

        return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides,
                training, name, bn=False, **kwargs):
    """Creates one layer of blocks for the ResNet model."""

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        def projection_shortcut(inputs):
            return conv2d_fixed_padding(inputs, filters, 1, strides,
                                        name='projection_shortcut', **kwargs)

        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, training, projection_shortcut, strides, 'block0', bn=bn, **kwargs)
        
        for i in range(1, blocks):
            inputs = block_fn(inputs, filters, training, None, 1, 'block{}'.format(i), bn=bn, **kwargs)

        return inputs
    

def resnet(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    
    snconv_kwargs = dict(beta=beta, spectral_norm=False,
                         update_collection=update_collection, reuse=reuse)
    
    num_filters = 16
    kernel_size = 3
    conv_stride = 1
    resnet_size = 32
    num_blocks = (resnet_size - 2) // 6
    block_sizes = [num_blocks] * 3
    block_strides = [1, 2, 2]

    # inputs: A Tensor representing a batch of input images.
    inputs = conv2d_fixed_padding(input_data, num_filters, kernel_size, conv_stride, **snconv_kwargs)
    inputs = tf.identity(inputs, 'initial_conv')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs = block_layer(inputs, num_filters, block_fn, num_blocks, block_strides[i], training,
                             name='block_layer{}'.format(i + 1), bn=False, **snconv_kwargs)

    inputs = tf.nn.relu(inputs)

    axes = [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    
    tf.add_to_collection('debug', inputs)
    
    fc = sn.linear(inputs, num_classes, scope_name='fc', xavier=True,
                   spectral_norm=False, reuse=reuse) 

    return fc
    
    
def resnet_bn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    
    snconv_kwargs = dict(beta=beta, spectral_norm=False,
                         update_collection=update_collection, reuse=reuse)
    
    num_filters = 16
    kernel_size = 3
    conv_stride = 1
    resnet_size = 32
    num_blocks = (resnet_size - 2) // 6
    block_sizes = [num_blocks] * 3
    block_strides = [1, 2, 2]

    # inputs: A Tensor representing a batch of input images.
    inputs = conv2d_fixed_padding(input_data, num_filters, kernel_size, conv_stride, **snconv_kwargs)
    inputs = tf.identity(inputs, 'initial_conv')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs = block_layer(inputs, num_filters, block_fn, num_blocks, block_strides[i], training,
                             name='block_layer{}'.format(i + 1), bn=True, **snconv_kwargs)

    inputs = batch_norm(inputs, training)
    inputs = tf.nn.relu(inputs)

    axes = [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    
    tf.add_to_collection('debug', inputs)
    
    fc = sn.linear(inputs, num_classes, scope_name='fc', xavier=True,
                   spectral_norm=False, reuse=reuse) 

    return fc


def resnet_sn(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    
    snconv_kwargs = dict(beta=beta, spectral_norm=True,
                         update_collection=update_collection, reuse=reuse)
    
    num_filters = 16
    kernel_size = 3
    conv_stride = 1
    resnet_size = 32
    num_blocks = (resnet_size - 2) // 6
    block_sizes = [num_blocks] * 3
    block_strides = [1, 2, 2]

    # inputs: A Tensor representing a batch of input images.
    inputs = conv2d_fixed_padding(input_data, num_filters, kernel_size, conv_stride, **snconv_kwargs)
    inputs = tf.identity(inputs, 'initial_conv')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs = block_layer(inputs, num_filters, block_fn, num_blocks, block_strides[i], training,
                             name='block_layer{}'.format(i + 1), bn=False, **snconv_kwargs)

    inputs = tf.nn.relu(inputs)

    axes = [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    
    tf.add_to_collection('debug', inputs)
    
    fc = sn.linear(inputs, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse) 

    return fc


def resnet_sn_large(input_data, num_classes=10, wd=0, update_collection=None, beta=1., reuse=None, training=False):
    
    # For imagenet. Architecture based on 
    # https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_main.py
    
    snconv_kwargs = dict(beta=beta, spectral_norm=True,
                         update_collection=update_collection, reuse=reuse)
    
    num_filters = 64
    kernel_size = 7
    conv_stride = 2
    resnet_size = 50
    num_blocks = (resnet_size - 2) // 6
    block_sizes = [num_blocks] * 3
    block_strides = [1, 2, 2, 2]

    # inputs: A Tensor representing a batch of input images.
    inputs = conv2d_fixed_padding(input_data, num_filters, kernel_size, conv_stride, **snconv_kwargs)
    inputs = tf.identity(inputs, 'initial_conv')

    for i, num_blocks in enumerate(block_sizes):
        num_filters = num_filters * (2**i)
        inputs = block_layer(inputs, num_filters, block_fn, num_blocks, block_strides[i], training,
                             name='block_layer{}'.format(i + 1), bn=False, **snconv_kwargs)

    inputs = tf.nn.relu(inputs)

    axes = [1, 2]
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    inputs = tf.identity(inputs, 'final_reduce_mean')
    
    tf.add_to_collection('debug', inputs)
    
    fc = sn.linear(inputs, num_classes, scope_name='fc', xavier=True,
                   update_collection=update_collection, beta=beta, reuse=reuse) 

    return fc