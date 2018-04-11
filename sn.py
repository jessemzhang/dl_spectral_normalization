### Code adapted from https://github.com/nnUyi/SNGAN, which is similar to https://github.com/minhnhat93/tf-SNDCGAN
### Note that if update_collection is not None, then need to run the update ops during training:
# for iter in range(max_iters):
#    # Training goes here
#    ...
#    # Update ops at the end
#    for update_op in spectral_norm_update_ops:
#        sess.run(update_op)

import tensorflow as tf
from operator import mul

def l2_norm(input_x, epsilon=1e-12):
    """normalize input to unit norm"""
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def conv2d(input_x, kernel_size, scope_name='conv2d', stride=1, tighter_sn=False,
           padding='SAME', spectral_norm=True, update_collection=None, xavier=False, bn=False):
    """2D convolution layer with spectral normalization option"""
    
    output_len = kernel_size[3]
    with tf.variable_scope(scope_name):
        if xavier:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights = tf.get_variable('weights', kernel_size, tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', output_len, tf.float32, initializer=tf.constant_initializer(0))
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection, tighter_sn=tighter_sn)
        conv = tf.nn.conv2d(input_x, weights, strides=[1, stride, stride, 1], padding=padding)
        conv = tf.nn.bias_add(conv, bias)
        if bn:
            conv = tf.layers.batch_normalization(conv)
        return conv


def linear(input_x, output_size, scope_name='linear', spectral_norm=True, 
           update_collection=None, l2_norm=False, wd=0, xavier=False):
    """Fully connected linear layer with spectral normalization and weight decay options"""
        
    shape = input_x.get_shape().as_list()
    
    if len(shape) > 2:
        flat_x = tf.reshape(input_x, [-1, reduce(mul, shape[1:])])
    else:
        flat_x = input_x
        
    shape = flat_x.get_shape()
    input_size = shape[1]
    
    with tf.variable_scope(scope_name):
        if xavier:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights = tf.get_variable('weights', [input_size, output_size], tf.float32, 
                                      initializer=tf.random_normal_initializer(stddev=0.02))
        if l2_norm:
            weight_decay = tf.multiply(tf.nn.l2_loss(weights), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        bias = tf.get_variable('bias', output_size, tf.float32, initializer=tf.constant_initializer(0))        
        if spectral_norm:
            weights = weights_spectral_norm(weights, update_collection=update_collection)
        output = tf.matmul(flat_x, weights) + bias
        return output


def weights_spectral_norm(weights, u=None, Ip=1, update_collection=None,
                          reuse=False, name='weights_SN', tighter_sn=False):
    """Perform spectral normalization"""

    def power_iteration(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.matmul(u_, tf.transpose(w_mat)))
            u_ = l2_norm(tf.matmul(v_, w_mat))
        return u_, v_

    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        w_shape = weights.get_shape().as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])
        
        # The tighter spectral normalization approach breaks the [f_in, f_out, d_in, d_out] filters
        # into a set of f_in*f_out subfilters each of size d_in*d_out. We want to get the spectral 
        # norm of each of these subfilters and normalize the original w_mat by the sum of these
        # spectral norms (this will guarantee that the linear transformation due to conv2d will 
        # remain spectral norm <= 1). 
        # ONLY USE THIS FOR conv2d LAYERS. Original sn works better for fully-connected layers
        if tighter_sn:
            
            if u is None:
                u = tf.get_variable('u', shape=[w_shape[0]*w_shape[1], w_shape[-1]], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)
            
            w_mat_list = tf.split(w_mat, w_shape[0]*w_shape[1], axis=0)
            u_list = tf.split(u, w_shape[0]*w_shape[1], axis=0)
            
            sigma_list = []
            u_hat_list = []

            for i, w in enumerate(w_mat_list):
                u_hat_, v_hat_ = power_iteration(u_list[i], w, Ip)
                u_hat_list.append(u_hat_)
                sigma_list.append(tf.matmul(tf.matmul(v_hat_, w), tf.transpose(u_hat_)))
                
            u_hat = tf.concat(u_hat_list, 0)
            sigma = tf.add_n(sigma_list)
        
        else:
            if u is None:
                u = tf.get_variable('u', shape=[1, w_shape[-1]], 
                                    initializer=tf.truncated_normal_initializer(), trainable=False)

            u_hat, v_hat = power_iteration(u, w_mat, Ip)
            sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
            
        w_mat = w_mat/sigma

        if update_collection is None:
            with tf.control_dependencies([u.assign(u_hat)]):
                w_norm = tf.reshape(w_mat, w_shape)
        else:
            tf.add_to_collection(update_collection, u.assign(u_hat))
            w_norm = tf.reshape(w_mat, w_shape)

        tf.add_to_collection('w_after_sn', w_norm)

        return w_norm