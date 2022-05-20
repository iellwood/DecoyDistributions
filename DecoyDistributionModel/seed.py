"""
Code from "Validating density estimation models with decoy distributions"

These functions generate seeds with a uniform distribution of log-likelihoods and compute the log-likelihood of a sample
from that distribution.

"""


import tensorflow as tf
import scipy.special
import numpy as np

def generate_seed(prior_info, image_shape):
    """
    Generates a seed for a normalizing flow with the given prior
    :param prior_info: A dictionary with parameters for the prior. Must include keys 'prior_type' (with values 'gaussian' or 'uniform_distribution_of_log_likelihoods')
    and 'W' if the latter.
    :param image_shape: The shape of the returned seed [batch_size, w, h, c]
    :return: A seed
    """

    w = image_shape[1]
    h = image_shape[2]
    c = image_shape[3]

    n = w*h*c

    batch_size = image_shape[0]

    if prior_info['prior_type'] == 'gaussian':
        return tf.random_normal(image_shape)
    elif prior_info['prior_type'] == 'uniform_distribution_of_log_likelihoods':
        return generate_uniform_distribution_of_log_likelihood_seed(batch_size, n=n, W=prior_info['W'], reshape_shape=image_shape)
    else:
        raise Exception('Unknown seed type')


def generate_uniform_distribution_of_log_likelihood_seed(batch_size, n, W, B=1, Lambda=0.999, reshape_shape=None):
    '''
    Generates a batch of seeds with an (approximately) uniform distribution of log-likelihoods.

    :param batch_size: the batch size
    :param n: number of dimensions of the space
    :param W: Width of the uniform distribution of log-liklihoods
    :param B: Determines the scale of the distribution as a function of radius. Set to one to compare with the paper
    :param Lambda: probability in [0, 1] that a sample is part of the uniform distribution of log-likelihoods
    :param reshape_shape: Optional shape to reshape the [batch_size, n] dimensional tensor
    :return: a tensor of shape [batch_size, n] of batch_size seeds with a uniform distribution of log-likelihoods
    '''

    A = tf.constant(Lambda/W*n, dtype=tf.float32) # WARNING: This definition of A is off by a factor of n from the paper
    B = tf.constant(B, dtype=tf.float32)
    y0 = tf.constant(Lambda, dtype=tf.float32)

    x = tf.random_normal([batch_size, n])

    norm = tf.linalg.norm(x, axis=1)
    direction = x/tf.expand_dims(norm, 1)

    y = tf.random_uniform([batch_size])


    r_long_tail = B*tf.exp(y/A)*tf.pow(1 - tf.exp(-n*y/A), 1/n)

    r0 = B*tf.exp(y0/A)*tf.pow(1 - tf.exp(-n*y0/A), 1/n) # Radius beyond which an exponential distribution is used
    N = A/r0*1/(1 + tf.pow(r0/B, -n))/(1 - y0)

    y_bar = tf.nn.relu(y - y0 - 1e-9)/(1 - y0)

    r_exp = -tf.log(1 - y_bar)/N + r0

    r = tf.where(y >= y0, r_exp, r_long_tail)

    x = tf.expand_dims(r, 1)*direction

    if reshape_shape is not None:
        x = tf.reshape(x, reshape_shape)

    return x

def log_sphere_area(n):
    return np.log(2) + (n/2)*np.log(np.pi) - scipy.special.loggamma(n/2)

def log_likelihood_of_uniform_distribution_of_log_likelihood_prior(x, W, B=1, Lambda=0.999):
    '''
    Computes the likelihood in distribution with a uniform distribution of log-likelihoods

    :param x: a batch of values to evaluate the log-likelihood
    :param W: The width of the uniform distribution of log-likelihoods
    :param B: Determines the scale of the distribution as a function of radius. Set to one to compare with the paper
    :param Lambda: probability in [0, 1] that a sample is part of the uniform distribution of log likelihoods
    :return: log_likehood of the samples, x
    '''
    x = tf.reshape(x, [x.shape.as_list()[0], -1] )
    n = x.shape.as_list()[1]

    A = Lambda/W*n # WARNING: This definition of A is off by a factor of n from the paper


    r0 = B * tf.exp(Lambda / A) * tf.pow(1 - tf.exp(-n * Lambda / A), 1 / n)
    N = A/r0*1/(1 + tf.pow(r0/B, -n))/(1 - Lambda)

    r = tf.linalg.norm(x, axis=1) + 1e-10

    r_r = tf.maximum(r, B)
    r_l = tf.minimum(r, B)

    log_p_long_tail_right = tf.log(A/B) - tf.log(r_r/B) - tf.log(1 + tf.pow(r_r/B, -n))
    log_p_long_tail_left = tf.log(A/B) + (n - 1)*tf.log(r_l/B) - tf.log(1 + tf.pow(r_l/B, n))
    log_p_long_tail = tf.where(r >= B, log_p_long_tail_right, log_p_long_tail_left)
    log_p_exp = tf.log(N * (1 - Lambda)) - tf.nn.relu(r - r0) * N

    return tf.where(r >= r0, log_p_exp, log_p_long_tail) - log_sphere_area(n) - (n - 1)*tf.log(r)
