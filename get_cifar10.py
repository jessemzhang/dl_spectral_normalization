# Requires tflearn master version:
#   pip install git+https://github.com/tflearn/tflearn.git
#   much of the cifar10 fetching code is from Chiyuan Zhang at MIT
  
import os
import re
import numpy as np
import tensorflow as tf
import tflearn
  
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation  
from scipy.io import loadmat

def unpickle(file):
  import cPickle
  with open(file, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict


class dataset():

  def __init__(self,FLAGS):
    self.FLAGS = FLAGS
  
  def per_image_whitening(self,images):
    "Mimic tensorflow per_image_whitening"
    orig_shape = images.shape
    images = images.reshape((images.shape[0], -1))
    img_means = np.mean(images, axis=1, keepdims=True)
    img_stds = np.std(images, axis=1, keepdims=True)
    adj_stds = np.maximum(img_stds, 1.0 / np.sqrt(images.shape[1]))
    whiten_imgs = (images - img_means) / adj_stds
    return whiten_imgs.reshape(orig_shape)
  
  
  def crop_datasets(self, datasets, do_whitening=True):
    datasets_cropped = []
    for dset_x, dset_y in datasets:
      new_x = dset_x.reshape((-1, 32, 32, 3))
      if do_whitening:
        new_x = self.per_image_whitening(new_x)
      new_x = new_x[:, 2:30, 2:30, :]
      datasets_cropped.append((new_x, dset_y))
    return datasets_cropped
  
  
  def corrupt_dataset(self, dataset, percent_corrupt):
    # set random seed so that we get the same corrupted dataset
    if not self.FLAGS.rand_seed >= 0:
      np.random.seed(percent_corrupt)
  
    p_corrupt = percent_corrupt / 100.0
    dset_x, dset_y = dataset
  
    b_corrupt = np.random.rand(len(dset_y)) <= p_corrupt
    rand_y = np.random.choice(self.FLAGS.n_classes, len(dset_y))
    new_y = np.copy(dset_y)
    new_y[b_corrupt] = rand_y[b_corrupt]
  
    return dset_x, new_y
  
  
  def get_datasets(self):
    commands = self.FLAGS.dataset.split('|')
    name = commands[0]
    commands = commands[1:]
    if name == 'cifar10':
      datasets = list(tflearn.datasets.cifar10.load_data())
    elif name == 'cifar100':
      dataset_tr = unpickle('./cifar-100/train')
      dataset_tt = unpickle('./cifar-100/test')
      datasets = [(np.transpose(dataset_tr['data'].reshape(-1,3,32,32).astype(float)/255,(0,2,3,1)),
                   np.array(dataset_tr['fine_labels'])),
                  (np.transpose(dataset_tt['data'].reshape(-1,3,32,32).astype(float)/255,(0,2,3,1)),
                   np.array(dataset_tt['fine_labels']))]
    elif name == 'cifar20':
      dataset_tr = unpickle('./cifar-100/train')
      dataset_tt = unpickle('./cifar-100/test')
      datasets = [(np.transpose(dataset_tr['data'].reshape(-1,3,32,32).astype(float)/255,(0,2,3,1)),
                   np.array(dataset_tr['coarse_labels'])),
                  (np.transpose(dataset_tt['data'].reshape(-1,3,32,32).astype(float)/255,(0,2,3,1)),
                   np.array(dataset_tt['coarse_labels']))]
    elif name == 'svhn':
      if not os.path.isdir('./svhn'):
        os.system('mkdir svhn')
      if not os.path.isfile('./svhn/train_32x32.mat'):
        print('Downloading SVHN train set..')
        os.system('wget -O ./svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
      if not os.path.isfile('./svhn/test_32x32.mat'):
        print('Downloading SVHN test set..')
        os.system('wget -O ./svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
      dataset_tr = loadmat('./svhn/train_32x32.mat')
      dataset_tt = loadmat('./svhn/test_32x32.mat')
      y_tr = dataset_tr['y'].reshape(-1).astype(int)
      y_tr[y_tr == 10] = 0
      y_tt = dataset_tt['y'].reshape(-1).astype(int)
      y_tt[y_tt == 10] = 0
      datasets = [(np.transpose(dataset_tr['X'].astype(float)/255, axes=[3, 0, 1, 2]), y_tr),
                  (np.transpose(dataset_tt['X'].astype(float)/255, axes=[3, 0, 1, 2]), y_tt)]
    elif name == 'mnist':
      def rgb_and_pad_mnist(im):
        im_ = np.zeros((32,32))
        im_[2:30,2:30] = im
        return np.repeat(im_[:, :, np.newaxis],3,axis=2)
      a = list(tflearn.datasets.mnist.load_data())
      datasets = [(np.array(map(rgb_and_pad_mnist,a[0].reshape(-1,28,28))),a[1]),
                  (np.array(map(rgb_and_pad_mnist,a[2].reshape(-1,28,28))),a[3])]
    elif name.startswith('data-gen:'):
      ret = re.match(r'data-gen:(.*)', name)
      dset_fn = os.path.join(self.FLAGS.logdir, 'data-gen', DATAGEN_MAPPING[ret.group(1)])
      data = np.load(dset_fn)
      datasets = [(data['x_tr'], data['y_tr']), (data['x_tt'], data['y_tt'])]
  
    def get_dset_idx(pattern):
      if pattern == 'tr':
        return [0]
      if pattern == 'tt':
        return [1]
      if pattern == 'trtt':
        return [0, 1]
  
    for cmd in commands:
      if cmd.startswith('SubS:'):
        # take subset, e.g. take a random subset of 5000 samples for training
        #   SubS:tr:5000
        ret = re.match(r'SubS:([^:]*):(.*)', cmd)
  
        subset_count = int(ret.group(2))
        # set random seed for reproducibility
        if not self.FLAGS.rand_seed >= 0:
          np.random.seed(subset_count)

        for ds_idx in get_dset_idx(ret.group(1)):
          dset_x, dset_y = datasets[ds_idx]
          dset_count = dset_x.shape[0]
          assert dset_count >= subset_count
          subset_idx = np.random.choice(dset_count, subset_count, replace=False)
          datasets[ds_idx] = (dset_x[subset_idx], dset_y[subset_idx])
  
      if cmd.startswith('RndL:'):
        # randomly corrupt labels, e.g. randomly corrupt the train and test set labels with 20% probability
        #   RndL:trtt:20
        ret = re.match(r'RndL:([^:]*):(.*)', cmd)
        p_corrupt = int(ret.group(2))
  
        for ds_idx in get_dset_idx(ret.group(1)):
          datasets[ds_idx] = self.corrupt_dataset(datasets[ds_idx], p_corrupt)
  
      if cmd.startswith('Cls:'):
        # take only samples belong to the specificed classes here
        ret = re.match(r'Cls:([^:]*):(.*)', cmd)
        classes = [int(x) for x in ret.group(2)]
  
        for ds_idx in get_dset_idx(ret.group(1)):
          dset_x, dset_y = datasets[ds_idx]
          dset_y = np.array(dset_y)
          idx_sel = np.zeros(dset_y.shape, dtype=bool)
          for c in classes:
            idx_sel += np.equal(dset_y, c)
  
          datasets[ds_idx] = (dset_x[idx_sel], dset_y[idx_sel])
  
    return datasets
  
  
  def prepare_inputs(self):
    datasets = self.crop_datasets(self.get_datasets(), do_whitening=self.FLAGS.per_image_whitening)
    return datasets
  

# default parameters
class cifar10_parameters():
  # Name of the dataset
  dataset = 'cifar10'
  # Whether to perform tf style per image whitening
  per_image_whitening = True
  # Number of classes
  n_classes = 10
  # Use this random seed if non-negative
  rand_seed = -1

class cifar20_parameters():
  dataset = 'cifar20'
  per_image_whitening = True
  n_classes = 20
  rand_seed = -1

class cifar100_parameters():
  dataset = 'cifar100'
  per_image_whitening = True
  n_classes = 100
  rand_seed = -1

class svhn_parameters():
  dataset = 'svhn'
  per_image_whitening = True
  n_classes = 10
  rand_seed = -1

class mnist_parameters():
  dataset = 'mnist'
  per_image_whitening = True
  n_classes = 10
  rand_seed = -1

def cifar10_one_hot(i):
  v = np.zeros(10)
  v[i] = 1
  return v


def get_cifar10_dataset(p_corrupt_label,n_samps=50000,rand_seed=None,onehot=False):
  class params(cifar10_parameters):
    def __init__(self,p,rand_seed,n_samp=50000):
      self.dataset = 'cifar10|SubS:tr:%s|RndL:trtt:%s'%(int(n_samp),int(p))
      self.rand_seed = rand_seed

  p = params(p_corrupt_label,rand_seed,n_samp=n_samps)
  c = dataset(p)
  datasets = c.prepare_inputs()

  if onehot:
    return datasets[0][0],np.array(map(cifar10_one_hot,datasets[0][1])), \
           datasets[1][0],np.array(map(cifar10_one_hot,datasets[1][1]))
  
  return datasets[0][0],datasets[0][1],datasets[1][0],datasets[1][1]


def get_cifar100_dataset(p_corrupt_label,n_samps=50000,rand_seed=None):
  class params(cifar100_parameters):
    def __init__(self,p,rand_seed,n_samp=50000):
      self.dataset = 'cifar100|SubS:tr:%s|RndL:trtt:%s'%(int(n_samp),int(p))
      self.rand_seed = rand_seed
      
  p = params(p_corrupt_label,rand_seed,n_samp=n_samps)
  c = dataset(p)
  datasets = c.prepare_inputs()

  return datasets[0][0],datasets[0][1],datasets[1][0],datasets[1][1]


# Same as cifar100 except with coarser labels
def get_cifar20_dataset(p_corrupt_label,n_samps=50000,rand_seed=None):
  class params(cifar20_parameters):
    def __init__(self,p,rand_seed,n_samp=50000):
      self.dataset = 'cifar20|SubS:tr:%s|RndL:trtt:%s'%(int(n_samp),int(p))
      self.rand_seed = rand_seed

  p = params(p_corrupt_label,rand_seed,n_samp=n_samps)
  c = dataset(p)
  datasets = c.prepare_inputs()

  return datasets[0][0],datasets[0][1],datasets[1][0],datasets[1][1]


def get_svhn_dataset(p_corrupt_label,n_samps=73257,rand_seed=None):
  class params(svhn_parameters):
    def __init__(self,p,rand_seed,n_samp=73257):
      self.dataset = 'svhn|SubS:tr:%s|RndL:trtt:%s'%(int(n_samp),int(p))
      self.rand_seed = rand_seed

  p = params(p_corrupt_label,rand_seed,n_samp=n_samps)
  c = dataset(p)
  datasets = c.prepare_inputs()

  return datasets[0][0],datasets[0][1],datasets[1][0],datasets[1][1]


def get_mnist_dataset(p_corrupt_label,n_samps=50000,rand_seed=None):
  class params(mnist_parameters):
    def __init__(self,p,rand_seed,n_samp=50000):
      self.dataset = 'mnist|SubS:tr:%s|RndL:trtt:%s'%(int(n_samp),int(p))
      self.rand_seed = rand_seed

  p = params(p_corrupt_label,rand_seed,n_samp=n_samps)
  c = dataset(p)
  datasets = c.prepare_inputs()

  return datasets[0][0],datasets[0][1],datasets[1][0],datasets[1][1]
