# Code adapted and modified from https://github.com/openai/cleverhans

import time
import os
import numpy as np
import tensorflow as tf
import dl_utils

def l1_norm_keepdims(x, epsilon=1e-24):
    """normalize input to unit L1 norm"""
    reduc_ind = tuple(xrange(1, len(x.shape)))
    return np.sum(np.abs(x), axis=reduc_ind, keepdims=True) + epsilon


def l2_norm_keepdims(x, epsilon=1e-24):
    """normalize input to unit L2 norm"""
    reduc_ind = tuple(xrange(1, len(x.shape)))
    return np.sqrt(np.sum(np.square(x), axis=reduc_ind, keepdims=True)) + epsilon


def fgm(x, graph, sess, eps=0.3, order=2):
    """
    Fast Gradient Method for computing adv perturbation of x
    """

    g = sess.run(graph['x_grad'], feed_dict={graph['input_data']: x,
                                             graph['delta']: np.zeros(np.shape(x))})
    
    if order == np.inf:
        adv_x = x + eps*np.sign(g)
        
    elif order == 1:
        reduc_ind = tuple(xrange(1, len(x.shape)))
        adv_x = x + eps*l1_norm_keepdims(g)
        
    elif order == 2:
        reduc_ind = tuple(xrange(1, len(x.shape)))
        adv_x = x + eps*l2_norm_keepdims(g)

    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                  "currently implemented.")

    return adv_x


def pgm(x, graph, sess, eps=0.3, k=15, a=None, order=2, num_stability=1e-24):
    """
    Projected Gradient Method
    (referencing: https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py)
    """
    
    if a is None:
        a = 2*eps/k
    
    adv_x = np.copy(x)
    for i in range(k):
        
        g = sess.run(graph['x_grad'], feed_dict={graph['input_data']: adv_x,
                                                 graph['delta']: np.zeros(np.shape(x))})
        
        if order == np.inf:
            adv_x += a*np.sign(g)
            adv_x = np.clip(adv_x, x-eps, x+eps) 
            
        elif order == 1:
            adv_x += a*g/l1_norm_keepdims(g)
            d = adv_x-x
            u = l1_norm_keepdims(d)
            u_ind = u.reshape(-1) > eps
            adv_x[u_ind] = x[u_ind] + eps*d[u_ind]/u[u_ind]
            
        elif order == 2:
            adv_x += a*g/l2_norm_keepdims(g)
            d = adv_x-x
            u = l2_norm_keepdims(d)
            u_ind = u.reshape(-1) > eps
            adv_x[u_ind] = x[u_ind] + eps*d[u_ind]/u[u_ind]
            
        else:
            raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                      "currently implemented.")

    return adv_x


def wrm(x, graph, sess, eps=0.3, Ip=15, step_adv=None):
    """
    Wasserstein Distributionally Adversarial Training Method
    (referencing: https://github.com/duchi-lab/certifiable-distributional-robustness/blob/master/attacks_tf.py)
    """
    
    d = np.zeros(np.shape(x))

    # Optimization using gradient descent as described in paper
    sess.run(graph['eps'].assign(eps))

    for ip in range(Ip):
        if step_adv is None:
            step_adv = 1/np.sqrt(2+ip)

        d -= step_adv * sess.run(graph['delta_grad'],
                                 feed_dict={graph['input_data']: x,
                                            graph['delta']: d})
        
    return x-d
    
    
def gen_adv_examples_in_sess(X, graph, sess, batch_size=100, method=fgm, **kwargs):
    """Use trained model to generate adversarial examples from X within a session"""
    
    adv_x = np.zeros(np.shape(X))
    for i in range(0, len(X), batch_size):
        adv_x[i:i+batch_size] = method(X[i:i+batch_size], graph, sess, **kwargs)
    return adv_x


def test_net_against_adv_examples(X, Y, load_dir, arch, d=None, beta=1., num_channels=3,
                                  verbose=True, gpu_id=0, gpu_prop=0.2,
                                  method=fgm, **kwargs):
    """For a trained network, generate and get accuracy for adversarially-perturbed samples"""
    
    num_classes = len(np.unique(Y))
    start = time.time()
        
    # Use previously fitted network which had achieved 100% training accuracy
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = dl_utils.graph_builder_wrapper(num_classes, load_dir, beta=beta, arch=arch,
                                               num_channels=num_channels, update_collection='_')

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
            if d is None:
                load_epoch = dl_utils.latest_epoch(load_dir)
                graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

            else:
                for v in tf.trainable_variables():
                    sess.run(v.assign(d[v.name]))
            
            # Generate adversarial samples and predict
            X_adv = gen_adv_examples_in_sess(X, graph, sess, method=method, **kwargs)
            Yhat_adv = dl_utils.predict_labels_in_sess(X_adv, graph, sess)

    accs_adv = np.sum(Yhat_adv == Y)/float(len(Y))

    if verbose:
        print('Acc on adv examples: %.2f (%.3f s elapsed)' \
              %(accs_adv, time.time()-start))

    return accs_adv
            
