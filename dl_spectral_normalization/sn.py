### Code adapted from https://github.com/nnUyi/SNGAN, which is similar to https://github.com/minhnhat93/tf-SNDCGAN
### Note that if update_collection is not None, then need to run the update ops during training:
# for iter in range(max_iters):
#    # Training goes here
#    ...
#    # Update ops at the end
#    for update_op in spectral_norm_update_ops:
#        sess.run(update_op)
#
# Setting update_collection is important for grabbing weights from the graph!! 

import tensorflow as tf
from operator import mul

def l2_norm(input_x, epsilon=1e-12):
    """normalize input to unit norm"""
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def conv2d(input_x, kernel_size, scope_name='conv2d',
           xavier=True, variance_scaling=False, stride=1, padding='SAME', use_bias=True,
           beta=1., spectral_norm=True, tighter_sn=True,
           update_collection=None, reuse=None,
           l2_norm=False, wd=0, 
           bn=False, training=False):
    """2D convolution layer with spectral normalization option"""
    
    shape = input_x.get_shape().as_list()
    assert shape[1] == shape[2]
    u_width = shape[1]
    
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name, reuse=reuse):
        if xavier:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        elif variance_scaling:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.variance_scaling_initializer())
        else:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection,
                                            tighter_sn=tighter_sn, u_width=u_width, beta=beta,
                                            u_depth=kernel_size[-2], stride=stride, padding=padding)
        elif l2_norm:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(input_x, weights, strides=[1, stride, stride, 1], padding=padding)
        if use_bias:
            bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
            conv = tf.nn.bias_add(conv, bias)
        if bn:
            conv = tf.layers.batch_normalization(conv, training=training)
        return conv


def linear(input_x, output_size, scope_name='linear', spectral_norm=True, 
           update_collection=None, l2_norm=False, wd=0, xavier=True, beta=1., reuse=None):
    """Fully connected linear layer with spectral normalization and weight decay options"""
        
    shape = input_x.get_shape().as_list()
    
    if len(shape) > 2:
        flat_x = tf.reshape(input_x, [-1, reduce(mul, shape[1:])])
    else:
        flat_x = input_x
        
    shape = flat_x.get_shape()
    input_size = shape[1]
    
    with tf.variable_scope(scope_name, reuse=reuse):
        if xavier:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection, beta=beta)
        elif l2_norm:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        output = tf.matmul(flat_x, weights) + bias
        return output


def weights_spectral_norm(weights, u=None, Ip=1, update_collection=None,
                          reuse=False, name='weights_SN', beta=1.,
                          tighter_sn=False, u_width=28, u_depth=3, stride=1, padding='SAME'):
    """Perform spectral normalization"""

    def power_iteration(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.matmul(u_, tf.transpose(w_mat)))
            u_ = l2_norm(tf.matmul(v_, w_mat))
        return u_, v_

    def power_iteration_conv(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.nn.conv2d(u_, w_mat, strides=[1, stride, stride, 1], padding=padding))
            u_ = l2_norm(tf.nn.conv2d_transpose(v_, w_mat, [1, u_width, u_width, u_depth],
                                                strides=[1, stride, stride, 1], padding=padding))
        return u_, v_

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        
        # The tighter spectral normalization approach breaks the [f_in, f_out, d_in, d_out] filters
        # into a set of f_in*f_out subfilters each of size d_in*d_out.
        # ONLY USE THIS FOR conv2d LAYERS. Original sn works better for fully-connected layers
        if tighter_sn:
            if u is None:
                # Initialize u (our "eigenimage")
                u = tf.get_variable('u', shape=[1, u_width, u_width, u_depth], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)

            u_hat, v_hat = power_iteration_conv(u, weights, Ip)
            z = tf.nn.conv2d(u_hat, weights, strides=[1, stride, stride, 1], padding=padding)
            sigma = tf.maximum(tf.reduce_sum(tf.multiply(z, v_hat))/beta, 1)
            
            if update_collection is None:
                with tf.control_dependencies([u.assign(u_hat)]):
                    w_norm = weights/sigma
            else:
                tf.add_to_collection(update_collection, u.assign(u_hat))
                w_norm = weights/sigma

        # Use the spectral normalization proposed in SN-GAN paper
        else:
            if u is None:
                u = tf.get_variable('u', shape=[1, w_shape[-1]], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)

            w_mat = tf.reshape(weights, [-1, w_shape[-1]])
            u_hat, v_hat = power_iteration(u, w_mat, Ip)
            sigma = tf.maximum(tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))/beta, 1)
            
            w_mat = w_mat/sigma

            if update_collection is None:
                with tf.control_dependencies([u.assign(u_hat)]):
                    w_norm = tf.reshape(w_mat, w_shape)
            else:
                tf.add_to_collection(update_collection, u.assign(u_hat))
                w_norm = tf.reshape(w_mat, w_shape)

        tf.add_to_collection('w_after_sn', w_norm)

        return w_norm