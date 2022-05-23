Validating density estimation models with decoy distributions
============================================

This repository provides 1) Code for loading pre-trained decoy distribution models 2) Code to train decoy distributions on either CIFAR10 or MNIST.

To download the pretrained model data, please download the "release" version of the repository, which includes a .zip file (~1 Gb) containing  the pretrained models.

To train a decoy distribution, use the command,

`python TrainDecoyDistribution.py -w Width -d Dataset -g GPU -b BatchSize`

Width can be any positive number and determines the width of the uniform distribution of log-likelihoods. Dataset can be either MNIST or CIFAR10. GPU specifies which GPU to use (default 0) and BatchSize defaults to 50.
For a complete list of supported commands, type

`python TrainDecoyDistribution.py -h`

To load an existing model, use the DecoyDistribution class. For example,

```python
import DecoyDistributionModel.DecoyDistribution`
filename = 'my_trained_decoy_distribution.obj'
sess = tf.InteractiveSession()
decoy = DecoyDistribution(filename)
tf.compat.v1.global_variables_initializer().run()
decoy.initialize(sess)
```

We have provided an example script that loads a decoy distribution given a filename,

`python ExampleDecoyDistributionLoad -f FILENAME`

This script plots samples from the loaded decoy distribtion as well as a histogram of log-likelihoods.

## Requirements

All code was written in Python 3.6 and tensorflow 1.15, and requires the packages numpy, scipy, matplotlib and pickle


