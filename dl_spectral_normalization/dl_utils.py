from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import adversarial as ad
from sklearn.utils import shuffle


def loss(g, Y, mean=True, add_other_losses=True):
    """Cross-entropy loss between labels and output of linear activation function"""
    out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=g)
    
    if mean:
        out = tf.reduce_mean(out)
        
        if add_other_losses:
            tf.add_to_collection('losses', out)
            return tf.add_n(tf.get_collection('losses'))
            
    return out


def acc(g, Y):
    """Accuracy"""
    correct_prediction = tf.equal(Y, tf.argmax(g, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def graph_builder_wrapper(arch,
                          num_classes=10,
                          adv='erm',
                          eps=0.3,
                          save_dir=None,
                          wd=0,
                          update_collection=None,
                          beta=1.,
                          save_histograms=False,
                          num_channels=3,
                          max_save=200,
                          training=False,
                          loss=loss,
                          order=2,
                          opt='momentum'):
    """Wrapper for building graph and accessing all relevant ops/placeholders"""
    
    assert isinstance(adv, str)

    input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, num_channels], name='in_data')
    input_labels = tf.placeholder(tf.int64, shape=[None], name='in_labels')

    fc_out = arch(input_data, num_classes=num_classes, wd=wd, training=training,
                  beta=beta, update_collection=update_collection)

    # Loss and optimizer (with adversarial training options)
    learning_rate = tf.Variable(0.01, name='learning_rate', trainable=False)

    if adv in ['wrm', 'fgm', 'pgm']:
        if adv == 'wrm':
            adv_x = ad.wrm(input_data, fc_out, eps=eps, order=order, model=arch, k=15,
                           num_classes=num_classes, graph_beta=beta, training=training)
        elif adv == 'fgm':
            adv_x = ad.fgm(input_data, fc_out, eps=eps, order=order, training=training)
            
        elif adv == 'pgm':
            adv_x = ad.pgm(input_data, fc_out, eps=eps, order=order, model=arch, k=15,
                           num_classes=num_classes, graph_beta=beta, training=training)
            
        fc_out_adv = arch(adv_x, num_classes=num_classes, wd=wd,
                          beta=beta, update_collection=update_collection, reuse=True, training=training)
        
    else:
        fc_out_adv = fc_out
        
    total_loss = loss(fc_out_adv, input_labels)
    total_acc = acc(fc_out_adv, input_labels)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        if num_channels == 1 or opt == 'adam': # For MNIST dataset
            opt_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)
        else:
            opt_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)
    
    # Output dictionary to useful tf ops in the graph
    graph = dict(
        input_data = input_data,
        input_labels = input_labels,
        total_loss = total_loss,
        total_acc = total_acc,
        fc_out = fc_out,
        fc_out_adv = fc_out_adv,
        opt_step = opt_step,
        learning_rate = learning_rate
    )

    # Saving weights and useful information to tensorboard
    if save_dir is not None:
        saver = tf.train.Saver(max_to_keep=max_save)
        graph['saver'] = saver
        
        if not os.path.isdir(save_dir):
            tf.summary.scalar('loss', total_loss)
            tf.summary.scalar('accuracy', total_acc)

            # Add histograms for trainable variables (really slows down training though)
            if save_histograms:
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)

            # Merge all the summaries and write them out to save_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'))
            graph_writer = tf.summary.FileWriter(os.path.join(save_dir, 'graph'), graph=tf.get_default_graph())
            valid_writer = tf.summary.FileWriter(os.path.join(save_dir, 'validation'))

            graph['merged'] = merged
            graph['train_writer'] = train_writer
            graph['graph_writer'] = graph_writer
            graph['valid_writer'] = valid_writer
    
    return graph


def train(Xtr, Ytr, graph, save_dir,
          val_set=None,
          lr_initial=0.01,
          seed=0,
          num_epochs=100,
          batch_size=100,
          write_every=1,
          save_every=None,
          verbose=True,
          load_epoch=-1,
          early_stop_acc=None,
          early_stop_acc_num=10,
          gpu_prop=0.2,
          shuffle_data=True):
    """Train the graph"""
    
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    if save_every is None:
        if num_epochs > 100:
            save_every = num_epochs/100
        else:
            save_every = 1
    
    start = time.time()
    training_losses, training_accs = [], []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:

        sess.run(tf.global_variables_initializer())

        if load_epoch > -1:
            if verbose:
                print('Continuing training starting at epoch %s+1'%(load_epoch))
            if save_dir is not None:
                restore_weights_file = os.path.join(save_dir, 'checkpoints', 'epoch%s'%(load_epoch))
            if 'saver' in graph:
                graph['saver'].restore(sess, restore_weights_file)

        else:
            if save_dir is not None and not os.path.exists(os.path.join(save_dir, 'checkpoints')):
                os.mkdir(os.path.join(save_dir, 'checkpoints'))
            if 'saver' in graph and save_dir is not None:
                graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch0'))
        
        for epoch in range(load_epoch+2, load_epoch+num_epochs+2):

            lr = lr_initial*0.95**(epoch/390.) # initial lr * decay rate ^(step/decay_steps)
            sess.run(graph['learning_rate'].assign(lr))

            t = time.time()
            training_loss = 0.
            training_acc = 0.
            steps = 0
            if shuffle_data:
                Xtr_, Ytr_ = shuffle(Xtr, Ytr)
            else:
                Xtr_, Ytr_ = Xtr, Ytr

            if len(Xtr_)%batch_size == 0:
                end = len(Xtr_)
            else:
                end = len(Xtr_)-batch_size
            for i in range(0, end, batch_size):

                x, y = Xtr_[i:i+batch_size], Ytr_[i:i+batch_size]

                feed_dict = {graph['input_data']: x, graph['input_labels']: y}
                training_loss_, training_acc_, _ = \
                    sess.run([graph['total_loss'], graph['total_acc'], graph['opt_step']],
                             feed_dict=feed_dict)
                training_loss += training_loss_
                training_acc += training_acc_
                steps += 1

                if verbose:
                    print('\rEpoch %s/%s (%.3f s), batch %s/%s (%.3f s): loss %.3f, acc %.3f'
                          %(epoch, load_epoch+num_epochs+1, time.time()-start, steps,
                            len(Xtr_)/batch_size, time.time()-t, training_loss_, training_acc_),
                          end='')            

            if 'saver' in graph and epoch%write_every == 0: # writing to tensorboard
                summary = sess.run(graph['merged'], feed_dict=feed_dict)
                graph['train_writer'].add_summary(summary, epoch)

                if val_set is not None: # make sure to keep the val_set small
                    feed_dict = {graph['input_data']: val_set['X'],
                                 graph['input_labels']: val_set['Y']}
                    summary = sess.run(graph['merged'], feed_dict=feed_dict)
                    graph['valid_writer'].add_summary(summary, epoch)

            if 'saver' in graph and save_dir is not None and epoch%save_every == 0:
                graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))

            training_losses.append(training_loss/float(steps))
            training_accs.append(training_acc/float(steps))

            if early_stop_acc is not None and np.mean(training_accs[-early_stop_acc_num:]) >= early_stop_acc:
                if verbose:
                    print('\rMean acc >= %s for last %s epochs. Stopping training after epoch %s/%s.'
                          %(early_stop_acc, early_stop_acc_num, epoch, load_epoch+num_epochs+1), end='')
                break

        if verbose: print('\nDONE: Trained for %s epochs.'%(epoch))
        if 'saver' in graph and save_dir is not None and not os.path.exists(os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch))):
            graph['saver'].save(sess, os.path.join(save_dir, 'checkpoints', 'epoch%s'%(epoch)))

    return training_losses, training_accs


def build_graph_and_train(Xtr, Ytr, save_dir, arch,
                          num_classes=10,
                          num_channels=3,
                          adv=None,
                          eps=0.3,
                          wd=0,
                          gpu_id=0,
                          verbose=True,
                          beta=1.,
                          order=2,
                          opt='momentum',
                          get_train_time=False,
                          **kwargs):
    """Build tensorflow graph and train"""
    
    tf.reset_default_graph()

    if verbose: start = time.time()
    with tf.device("/gpu:%s"%(gpu_id)):
        if save_dir is None or not os.path.exists(save_dir) or 'checkpoints' not in os.listdir(save_dir):
            graph = graph_builder_wrapper(arch, adv=adv, eps=eps, 
                                          num_classes=num_classes, save_dir=save_dir,
                                          wd=wd, beta=beta, num_channels=num_channels, 
                                          order=order, training=True, opt=opt)
            if get_train_time:
                start = time.time()
            tr_losses, tr_accs = train(Xtr, Ytr, graph, save_dir, **kwargs)
            if get_train_time:
                train_time = time.time()-start
        else:
            
            graph = graph_builder_wrapper(arch, num_classes=num_classes, save_dir=save_dir,
                                          wd=wd, beta=beta, num_channels=num_channels,
                                          order=order, update_collection='_', opt=opt)
            if verbose:
                print('Model already exists.. loading trained model..')
                
        if 'gpu_prop' in kwargs:
            gpu_prop = kwargs.get('gpu_prop', "default value")
        if save_dir is None:
            train_acc = np.nan
            if verbose:
                print('save_dir set to None.. returning NaN since weights not saved')
        else:
            Ytrhat = predict_labels(Xtr, graph, save_dir, gpu_prop=gpu_prop)
            train_acc = np.sum(Ytrhat == Ytr)/float(len(Ytr))

    if verbose:
        print('Train acc: %.2f (%.1f s elapsed)'%(train_acc, time.time()-start))

    if get_train_time:
        return train_acc, train_time
        
    return train_acc


def predict_labels_in_sess(X, graph, sess, batch_size=100):
    """Predict labels within a session"""
    
    labels = np.zeros(len(X))
    for i in range(0, len(X), batch_size):
        g_ = sess.run(graph['fc_out'], feed_dict = {graph['input_data']:X[i:i+batch_size]})
        labels[i:i+batch_size] = np.argmax(g_, 1)
    return labels


def latest_epoch(save_dir):
    """Grabs int corresponding to last epoch of weights saved in save_dir"""
    
    return max([int(f.split('epoch')[1].split('.')[0]) 
                for f in os.listdir(os.path.join(save_dir, 'checkpoints')) if 'epoch' in f])


def predict_labels(X, graph, load_dir, 
                   batch_size=100,
                   load_epoch=None,
                   gpu_prop=0.2):
    """Use trained model to predict"""
    
    # Load from checkpoint corresponding to latest epoch if none given
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min((latest_epoch(load_dir), load_epoch))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))        
        return predict_labels_in_sess(X, graph, sess, batch_size=batch_size)


def build_graph_and_predict(X, load_dir, arch,
                            Y=None,
                            num_classes=10,
                            gpu_id=0,
                            beta=1.,
                            num_channels=3,
                            load_epoch=None,
                            gpu_prop=0.2,
                            order=2,
                            opt='momentum'):
    """Build a tensorflow graph and predict labels"""
    
    tf.reset_default_graph()
    with tf.device("/gpu:%s"%(gpu_id)):
        graph = graph_builder_wrapper(arch, num_classes=num_classes, save_dir=load_dir, 
                                      order=order, beta=beta, opt=opt,
                                      update_collection='_', num_channels=num_channels)
        Yhat = predict_labels(X, graph, load_dir, load_epoch=load_epoch, gpu_prop=gpu_prop)
    if Y is None: 
        return Yhat
    return np.sum(Yhat == Y)/float(len(Y))
        

def build_graph_and_get_acc(X, Y, arch, adv='erm', eps=0.3, save_dir=None, beta=1., order=2,
                            batch_size=100, gpu_prop=0.2, load_epoch=None, num_channels=3, opt='momentum'):
    """Build a tensorflow graph and gets accuracy"""

    tf.reset_default_graph()
    graph = graph_builder_wrapper(arch, adv=adv, eps=eps, save_dir=save_dir,
                                  update_collection='_', beta=beta, opt=opt,
                                  order=order, num_channels=num_channels)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        load_file = tf.train.latest_checkpoint(os.path.join(save_dir, 'checkpoints'))
        if load_epoch is not None:
            load_file = load_file.replace(load_file.split('epoch')[1], str(load_epoch))
        graph['saver'].restore(sess, load_file)
        
        num_correct = 0
        num_total_samples = 0
        for i in range(0, len(X), batch_size):
            x, y = X[i:i+batch_size], Y[i:i+batch_size]
            num_batch_samples = len(x)
            feed_dict = {graph['input_data']: x, graph['input_labels']: y}
            num_correct += sess.run(graph['total_acc'], feed_dict=feed_dict)*num_batch_samples
            num_total_samples += num_batch_samples
        
        return num_correct/num_total_samples


def recover_curve(X, Y, load_dir,
                  num_classes=10,
                  gpu_id=0,
                  verbose=True,
                  keyword='epoch'):
    """Evaluate performance on a dataset during training"""
    
    list_epochs = np.unique([int(f.split(keyword)[1].split('.')[0]) \
                             for f in os.listdir(os.path.join(load_dir, 'checkpoints')) if keyword in f])
    accs = np.zeros(len(list_epochs))
    
    if verbose: start = time.time()
    for i, epoch in enumerate(list_epochs):
        accs[i] = build_graph_and_predict(X, load_dir,
                                          Y=Y,
                                          num_classes=num_classes,
                                          gpu_id=gpu_id,
                                          load_epoch=epoch)
        if verbose:
            print('\rRecovered accuracy for %s %s/%s: %.2f (%.2f s elapsed)'
                  %(keyword, i+1, len(list_epochs), accs[i], time.time()-start), end='')
    if verbose:
        print('')
    return accs


def recover_train_and_test_curves(Xtr, Ytr, Xtt, Ytt, load_dir,
                                  num_classes=10,
                                  gpu_id=0,
                                  verbose=True):
    """Recover training and test curves"""
    
    train_accs = recover_curve(Xtr, Ytr, load_dir,
                               num_classes=num_classes,
                               gpu_id=gpu_id,
                               verbose=verbose)
    test_accs = recover_curve(Xtt, Ytt, load_dir,
                              num_classes=num_classes,
                              gpu_id=gpu_id,
                              verbose=verbose)
    return train_accs,test_accs


def get_embedding_in_sess(X, graph, sess, batch_size=100):
    """Gets embedding (last layer output) within a session"""
    
    num_classes = graph['fc_out_adv'].shape.as_list()[1]
    embedding = np.zeros((len(X), num_classes))
    for i in range(0, len(X), batch_size):
        embedding_ = sess.run(graph['fc_out_adv'], feed_dict = {graph['input_data']:X[i:i+batch_size]})
        embedding[i:i+batch_size] = embedding_
    return embedding


def get_embedding(X, load_dir, arch, num_classes=10, num_channels=3, beta=1., 
                  adv='erm', eps=0.3, order=2,
                  batch_size=100, sn_fc=False, load_epoch=None, gpu_prop=0.2):
    """recovers the representation of the data at the layer before the softmax layer
       Use sn_fc to indicate that last layer (should be named 'fc/weights:0') needs to be
         spectrally normalized.
    """
    
    tf.reset_default_graph()
    graph = graph_builder_wrapper(arch, num_classes=num_classes, num_channels=num_channels,
                                  save_dir=load_dir, beta=beta, update_collection='_',
                                  order=order, adv=adv, eps=eps)
    
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min((latest_epoch(load_dir), load_epoch))

    if sn_fc:
        assert 'fc/weights:0' in [v.name for v in tf.global_variables()]
        W_fc_tensor = [v for v in tf.global_variables() if v.name == 'fc/weights:0'][0]
        b_fc_tensor = [v for v in tf.global_variables() if v.name == 'fc/bias:0'][0]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

        # spectral normalization on last layer (fully connected)
        if sn_fc:
            W_fc, b_fc = sess.run([W_fc_tensor, b_fc_tensor])
            sigma = np.linalg.svd(W_fc.T)[1][0]
            sess.run([W_fc_tensor.assign(W_fc/sigma), b_fc_tensor.assign(b_fc/sigma)])

        return get_embedding_in_sess(X, graph, sess, batch_size=batch_size)
    
    
def get_grads_wrt_samples(X, Y, load_dir, arch, num_classes=10, num_channels=3, beta=1., 
                          batch_size=100, load_epoch=None, gpu_prop=0.2):
    """Computes gradients with respect to samples"""
    
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min((latest_epoch(load_dir), load_epoch))

    tf.reset_default_graph()
    graph = graph_builder_wrapper(arch, num_classes=num_classes, num_channels=num_channels,
                                  save_dir=load_dir, beta=beta, update_collection='_')

    grad, = tf.gradients(graph['total_loss'], graph['input_data'])

    g = np.zeros(np.shape(X))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:

        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))

        for i in range(0, len(X), batch_size):
            g_ = sess.run(grad, feed_dict={graph['input_data']: X[i:i+batch_size],
                                           graph['input_labels']: Y[i:i+batch_size]})
            g[i:i+batch_size] = g_
            
        return g

    
def check_weights_svs(load_dir, arch, num_classes=10, n=2, load_epoch=None, beta=1.):    
    """Check singular value of all weights"""
    
    tf.reset_default_graph()
    graph = graph_builder_wrapper(arch, num_classes=num_classes, save_dir=load_dir,
                                  update_collection='_', beta=beta)
    
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min((latest_epoch(load_dir), load_epoch))
        
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # Grab all weights
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        
        for tfvar in tf.get_collection('w_after_sn'):
            if 'weights' in tfvar.name:
                W = tfvar.eval(session=sess)
                print('%30s with shape %15s and top %s sv(s): %s' \
                      %(tfvar.name, np.shape(W), n, 
                        ', '.join(['%.2f'%(i) for i in np.linalg.svd(W.reshape(-1, np.shape(W)[-1]))[1][:n]])))
                

def print_total_number_of_trainable_params():
    """prints total number of trainable parameters according to default graph"""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

    
def extract_curve_tensorboard(tb_log_file, curve='loss'):
    """Given the name of a tensorboard event file, returns the desired curve"""
    
    values = []
    for e in tf.train.summary_iterator(tb_log_file):
        for v in e.summary.value:
            if v.tag == curve:
                values.append(v.simple_value)
    return np.array(values)


def extract_train_valid_tensorboard(load_dir, curve='accuracy', show_plot=False, only_final_value=False):
    """For a particular model, grab the tfevents training and validation curves"""

    # get train
    event_file = sorted(os.listdir(os.path.join(load_dir, 'train')))[0]
    tb_log_file = os.path.join(load_dir, 'train', event_file)
    train_values = extract_curve_tensorboard(tb_log_file, curve=curve)

    # get validation
    event_file = sorted(os.listdir(os.path.join(load_dir, 'validation')))[0]
    tb_log_file = os.path.join(load_dir, 'validation', event_file)
    valid_values = extract_curve_tensorboard(tb_log_file, curve=curve)

    if show_plot:
        plt.figure()
        plt.plot(train_values, label='training %s'%(curve))
        plt.plot(valid_values, label='validation %s'%(curve))
        plt.grid()
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(curve)
        plt.show()
        
    if only_final_value:
        return train_values[-1], valid_values[-1]

    return train_values, valid_values


def plot_stacked_hist(v0, v1, labels=None, bins=20):
    """Plots two histograms on top of one another"""
    if labels is None:
        labels = ['0', '1']
    bins = np.histogram(np.hstack((v0, v1)), bins=bins)[1]
    data = [v0, v1]
    plt.hist(data, bins, label=labels, alpha=0.8, color=['r','g'],
             normed=True, edgecolor='none')
    plt.legend()


def get_margins(X, Y, load_dir, arch, sn_fc=True, beta=1.):
    """Compute margins for X (margin = last layer difference between true label and 
       highest value that's not the true label)
    """
    
    num_classes = len(np.unique(Y))
    embeddings = get_embedding(X, load_dir, arch, num_classes=10, beta=beta, sn_fc=sn_fc)
#    embeddings = np.exp(embeddings)
#    embeddings /= np.sum(embeddings, 1).reshape(-1, 1)
    margins = np.zeros(len(embeddings))
    
    print('Sanity check: accuracy is %.5f.'
          %(np.sum(np.argmax(embeddings, 1) == Y)/float(len(Y))))
    
    for i in range(len(embeddings)):
        if Y[i] == 0:
            margins[i] = np.max(embeddings[i][1:])
        elif Y[i] == len(embeddings[0])-1:
            margins[i] = np.max(embeddings[i][:-1])
        else:
            margins[i] = np.max([np.max(embeddings[i][:int(Y[i])]),
                                np.max(embeddings[i][int(Y[i])+1:])])
            
    return margins


def get_weights(load_dir, arch, num_classes=10, beta=1., num_channels=3,
                   load_epoch=None, verbose=False, gpu_prop=0.2):
    """Grab all weights from graph (also works for spectrally-normalized models)"""
    
    if load_epoch is None:
        load_epoch = latest_epoch(load_dir)
    else:
        load_epoch = np.min((latest_epoch(load_dir), load_epoch))

    tf.reset_default_graph()
    graph = graph_builder_wrapper(arch, save_dir=load_dir, num_classes=num_classes, beta=beta,
                                  update_collection='_', num_channels=num_channels)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        graph['saver'].restore(sess, os.path.join(load_dir, 'checkpoints', 'epoch%s'%(load_epoch)))
        d = {v.name:sess.run(v) for v in tf.trainable_variables()}
        for v in tf.get_collection('w_after_sn'):
            key = v.name.split('_SN')[0]+':0'
            d[key] = sess.run(v)
            if verbose:
                dim = d[key].shape[-1]
                print('%30s with shape %15s and top 2 sv(s): %s' \
                      %(key, np.shape(d[key]),
                        ', '.join(['%.2f'%(i) for i in np.linalg.svd(d[key].reshape(-1, dim))[1][:2]])))

    return d


def l2_norm(input_x, epsilon=1e-12):
    """normalize input to unit norm"""
    input_x_norm = input_x/(tf.reduce_sum(input_x**2)**0.5 + epsilon)
    return input_x_norm


def power_iteration_tf(W, Ip=20, seed=0):
    """Power method for computing top singular value of a matrix W
       NOTE: resets tensorflow graph
    """
    
    def power_iteration(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.matmul(u_, tf.transpose(w_mat)))
            u_ = l2_norm(tf.matmul(v_, w_mat))
        return u_, v_
    
    tf.reset_default_graph()
    if seed is not None: 
        tf.set_random_seed(seed)
        
    u = tf.get_variable('u', shape=[1, W.shape[-1]],
                        initializer=tf.truncated_normal_initializer(), trainable=False)

    w_mat = tf.Variable(W)
    u_hat, v_hat = power_iteration(u, w_mat, Ip)
    sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(sigma).reshape(-1)


def power_iteration_conv_tf(W, length=28, width=28, stride=1, Ip=20, seed=0, padding='SAME'):
    """Power method for computing top singular value of a convolution operation using W.
       NOTE: resets tensorflow graph
       Also, note that if you set stride to 1 when the network is trained with stride = 2, 
       the output will be twice as large as expected
    """
    
    u_dims = [1, length, width, W.shape[-2]]
    
    def power_iteration_conv(u, w_mat, Ip):
        u_ = u
        for _ in range(Ip):
            v_ = l2_norm(tf.nn.conv2d(u_, w_mat, strides=[1, stride, stride, 1], padding=padding))
            u_ = l2_norm(tf.nn.conv2d_transpose(v_, w_mat, u_dims,
                                                strides=[1, stride, stride, 1], padding=padding))
        return u_, v_
    
    tf.reset_default_graph()
    if seed is not None:
        tf.set_random_seed(seed)
    
    # Initialize u (our "eigenimage")
    u = tf.get_variable('u', shape=u_dims, 
                        initializer=tf.truncated_normal_initializer(), trainable=False)

    w_mat = tf.Variable(W)
    u_hat, v_hat = power_iteration_conv(u, w_mat, Ip)
    z = tf.nn.conv2d(u_hat, w_mat, strides=[1, stride, stride, 1], padding=padding)
    sigma = tf.reduce_sum(tf.multiply(z, v_hat))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(sigma).reshape(-1)
    
    
def get_overall_sn(load_dir, arch, num_classes=10, verbose=True, return_snorms=False,
                   num_channels=3, seed=0, load_epoch=None, beta=1., gpu_prop=0.2):
    """Gets the overall spectral norm of a network with specified weights"""

    d = get_weights(load_dir, arch, num_classes=num_classes, gpu_prop=gpu_prop,
                    num_channels=num_channels, load_epoch=load_epoch, beta=beta)
        
    s_norms = {}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_prop))) as sess:
        
        conv_ops_dict = {'/'.join(i.name.split('/')[:-1]): {'stride':int(i.get_attr('strides')[1]), 
                                                            'padding':i.get_attr('padding'), 
                                                            'length':i.inputs[0].get_shape().as_list()[1],
                                                            'width':i.inputs[0].get_shape().as_list()[2],
                                                            'seed':seed} 
                         for i in sess.graph.get_operations()
                         if 'Conv2D' in i.name and 'gradients' not in i.name}

    for i in sorted(d.keys()):
        if 'weights' in i:
            if 'conv' in i:
                key = '/'.join(i.split('/')[:-1])
                s_norms[i] = power_iteration_conv_tf(d[i], **conv_ops_dict[key])[0]
            else:
                s_norms[i] = power_iteration_tf(d[i], seed=seed)[0]

            if verbose:
                print('%20s with spectral norm %.4f'%(i, s_norms[i]))
                
    if return_snorms:
        return s_norms

    return(np.prod(s_norms.values()))


from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))