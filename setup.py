# Following http://devarea.com/deploying-a-new-python-package-to-pypi/#.W_IJBJNKjOR
from distutils.core import setup
setup(
    name = 'dl_spectral_normalization',
    packages = ['dl_spectral_normalization'], # this must be the same as the name above
    version = '0.1',
    description = 'Library for building neural networks in TensorFlow with spectrally normalized layers',
    author = 'Jesse Zhang, Farzan Farnia',
    author_email = 'jessez@stanford.edu',
    url = 'https://github.com/dev-area/dl_spectral_normalization',
    download_url = 'https://github.com/dev-area/dl_spectral_normalization/tarball/0.1',
    keywords = ['deep-learning', 'neural-network', 'adversarial-attacks', 'spectral-normalization', 'regularization'], 
    classifiers = [],
)
