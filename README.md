Validating density estimation models with decoy distributions
============================================

This repository provides 1) Code for loading pre-trained decoy distribution models 2) Code to train decoy distributions on either CIFAR10 or MNIST.

To download the pretrained model data, please download the "release" version of the repository, which includes a .zip file (~1 Gb) containing  the pretrained models.

To train a decoy distribution, use the command

`python TrainDecoyDistribution.py -w Width -d Dataset -g GPU -b BatchSize`

Width can be any positive number and determines the width of the uniform distribution of log-likelihoods. Dataset can be either MNIST or CIFAR10. GPU specifies which GPU to use (default 0) and BatchSize defaults to 50.
For a complete list of supported commands, type

`python TrainDecoyDistribution.py -h`






## Requirements

All code was written in Python 3.6 and tensorflow 1.15, and requires the packages numpy, scipy, matplotlib and pickle


