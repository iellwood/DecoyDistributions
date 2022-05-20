"""
Code from "Validating density estimation models with decoy distributions"

Written in tensorflow 1.15

The code in this section is essentially the Real-NVP Model as implemented in the FlowGAN paper, but modified to produce
a constant Jacobian. See (https://github.com/ermongroup/flow-gan) for the original implementation. This code also
implements NICE, but that model was never used in this study.

Changes to their code:

1) Eliminating the sigmoid layer (set final_sigmoid = False)

2) Using nn_rigid.py, instead of nn.py from FlowGAN in order produce a normalizing flow with a constant Jacobian

3) Minor changes have been made to interface this code with our training algorithm

It is important to note that the FlowGAN model must always be run first "backwards" on an image, before it
is run "forwards" on a seed, due to the way that the model is initialized.


"""


"""
The core Real-NVP model
"""

import tensorflow as tf
import DecoyDistributionModel.real_nvp.nn_rigid as nn

#tf.set_random_seed(0)
layers = []


def construct_model_spec(scale_init=2, no_of_layers=8, add_scaling=True):
    global layers
    num_scales = scale_init
    for scale in range(num_scales - 1):
        layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
        layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
        layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

    # # final layer
    scale = num_scales - 1
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale, num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))


def construct_nice_spec(init_type="uniform", hidden_layers=1000, no_of_layers=1):
    global layers

    layers.append(nn.NICECouplingLayer('checkerboard0', name='Checkerboard_1', seed=0,
                                       init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
    layers.append(nn.NICECouplingLayer('checkerboard1', name='Checkerboard_2', seed=1,
                                       init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
    layers.append(nn.NICECouplingLayer('checkerboard0', name='Checkerboard_3', seed=2,
                                       init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
    layers.append(nn.NICECouplingLayer('checkerboard1', name='Checkerboard_4', seed=3,
                                       init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
    layers.append(nn.NICEScaling(name='Scaling', seed=4))


# the final dimension of the latent space is recorded here
# so that it can be used for constructing the inverse model
final_latent_dimension = []


def model_spec(x,
               final_sigmoid=True,
               reuse=True,
               model_type="nice",
               train=False,
               alpha=1e-7,
               init_type="uniform",
               hidden_layers=1000,
               no_of_layers=1,
               batch_norm_adaptive=0,
               ):
    counters = {}
    xs = nn.int_shape(x)
    sum_log_det_jacobians = tf.zeros(xs[0])

    y = x

    x_abs = tf.abs(x)
    if final_sigmoid:
        jac_sigmoid = tf.reduce_sum(x_abs + 2*tf.log(1 + tf.exp(-x_abs)), axis=[1, 2, 3])
    else:
        jac_sigmoid = tf.zeros([tf.shape(x).eval()[0]], dtype=tf.float32)


    if len(layers) == 0:
        if model_type == "nice":
            construct_nice_spec(init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers)
        else:
            construct_model_spec(no_of_layers=no_of_layers, add_scaling=(batch_norm_adaptive != 0))

    # construct forward pass
    z = None
    jac = sum_log_det_jacobians + jac_sigmoid
    i = 0
    for layer in layers:
        i += 1
        y, jac, z = layer.forward_and_jacobian(y, jac, z, reuse=reuse, train=train)
        y = tf.check_numerics(y, 'NaN or Inf found in y in layer %i' % (i))
        jac = tf.check_numerics(jac, 'NaN or Inf found in jac in layer %i' % (i))
        if z is not None:
            z = tf.check_numerics(z, 'NaN or Inf found in z in layer %i' % (i))

    if model_type == "nice":
        z = y
    else:
        z = tf.concat(axis=3, values=[z, y])

    # record dimension of the final variable
    global final_latent_dimension
    final_latent_dimension = nn.int_shape(z)


    return z, jac


def inv_model_spec(y,
               final_sigmoid=True,
               reuse=True,
               model_type="nice",
               train=False,
               alpha=1e-7,
               init_type="uniform",
               hidden_layers=1000,
               no_of_layers=1,
               batch_norm_adaptive=0
            ):
    # construct inverse pass for sampling
    xs = tf.shape(y).eval()
    jac = tf.zeros(xs[0])

    if model_type == "nice":
        z = y
    else:
        shape = final_latent_dimension
        z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
        y = None


    if len(layers) == 0:
        if model_type == "nice":
            construct_nice_spec(init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers)
        else:
            construct_model_spec(no_of_layers=no_of_layers, add_scaling=(batch_norm_adaptive != 0))

    i = 0
    for layer in reversed(layers):
        i += 1
        y, jac, z = layer.backward_and_jacobian(y, jac, z, reuse=reuse, train=train)
        y = tf.check_numerics(y, 'NaN or Inf found in y during inv pass layer %i' % (i))
        z = tf.check_numerics(z, 'NaN or Inf found in z during inv pass layer %i' % (i))


    if final_sigmoid:
        m_y_abs = -tf.abs(y)
        jac_sigmoid = tf.reduce_sum(m_y_abs - 2*tf.log(1 + tf.exp(m_y_abs)), axis=[1, 2, 3])
        jac += jac_sigmoid

    # inverse logit
    x = y
    x = tf.check_numerics(x, 'NaN or Inf found in inv output')
    return x, jac

