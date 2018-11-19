# Spectral Normalization for Deep Convolutional Networks

The code in this repository accompanies the experiments performed in the paper [Generalizable Adversarial Training via Spectral Normalization](https://arxiv.org/) by Farnia*, Zhang*, and Tse (*equal contributors).

The repository contains the following:
- [dl_spectral_normalization](https://github.com/jessemzhang/dl_spectral_normalization/tree/master/dl_spectral_normalization): Python deep learning module with spectral normalization code, code for building and training neural networks using TensorFlow, code for adversarially training networks, and example neural network architectures
- [notebooks_figures](https://github.com/jessemzhang/dl_spectral_normalization/tree/master/notebooks_figures): Contains scripts for generating all figures in the main text of the paper
- [get_cifar10.py](https://github.com/jessemzhang/dl_spectral_normalization/blob/master/get_cifar10.py): Code for downloading and preprocessing datasets as described by [Zhang et al. 2017](https://arxiv.org/pdf/1611.03530.pdf)
- [train_network_template.ipynb](https://github.com/jessemzhang/dl_spectral_normalization/blob/master/train_network_template.ipynb): Example notebook for training a neural network using the [dl_spectral_normalization](https://github.com/jessemzhang/dl_spectral_normalization/tree/master/dl_spectral_normalization) module

## Installation

The dl_spectral_normalization package can be installed via pip:

```
pip install dl_spectral_normalization
```

An example approach for accessing package contents is as follows:

```python
# Imports utilities for building and training networks
from dl_spectral_normalization import dl_utils

# Import one of the provided neural network architectures: AlexNet
from dl_spectral_normalization.models import alexnet

# Import adversarial training methods
from dl_spectral_normalization import adversarial as ad
```

For a more detailed tutorial, please refer to [train_network_template.ipynb](https://github.com/jessemzhang/dl_spectral_normalization/blob/master/train_network_template.ipynb). For references on visualizing results, we provide several examples in [notebooks_figures](https://github.com/jessemzhang/dl_spectral_normalization/tree/master/notebooks_figures). We were able to run all of our experiments in an [nvidia-docker image](https://github.com/NVIDIA/nvidia-docker) (tensorflow/tensorflow:latest-gpu running TensorFlow version 1.10.1).
