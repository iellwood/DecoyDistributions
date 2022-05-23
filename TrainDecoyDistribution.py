"""
Code for training a decoy distribution on a dataset for the paper, "Validating density estimation models with decoy
distributions"

Can be run from the console as

``python TrainDecoyDistribution.py -w Width -d Dataset -b BatchSize``

Run

``python -TrainingDecoyDistribution.py -h``

For a complete list of options.

Dataset can be either MNIST or CIFAR10. To implement other datasets, see datasets.py

For example, to train a decoy distribution with width 2048 on MNIST, run

``python TrainDecoyDistribution.py -w 2048 -d MNIST -b 50``

Note that the amount of time it takes for to complete the default 20K steps can be quite variable, as the number of
generator training steps per discriminator step is dynamic. The discriminator is often trained very slowly at first,
occasionally hitting the lower bound of 10K generator steps per discriminator step early in training. The generator was
trained anywhere from 300K to 900K steps relative to the discriminator's 20K.
Typical training times were around 4-7 days on a single Titan V GPU.
"""


import tensorflow as tf
import os
import datasets
from DecoyDistributionModel.DecoyDistribution import DecoyDistribution
from AdvEntGAN.adventgan import AdventGAN
from Util.parse import get_info_from_argparse
from Util.plotbatch import save_batch_plot
from Util.progresstracker import ProgressTracker
model_info, file_name_header, path, whichGPU, total_discriminator_steps = get_info_from_argparse()


#Select which GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=whichGPU

# Make sure the save path exists
if not os.path.isdir(path):
    os.makedirs(path)

sess = tf.InteractiveSession()


# Load a dataset and get a batch of images
real_images = datasets.get_image_input(model_info['dataset'], model_info['image_shape'][0], width=model_info['image_shape'][1], pad=True, pad_value=1)

# Build the decoy distribution model
print('Building decoy distribution...')
decoy = DecoyDistribution(model_info, input_image=real_images, image_type='nchw')

# Build the AdvEnt-GAN discriminator and controller
print('Building discriminator...')
advent_gan = AdventGAN(sess, real_images, decoy.generated_image, decoy.get_vars())

# Initialize the variables
print('Initializing variables...')
tf.compat.v1.global_variables_initializer().run()
decoy.initialize(sess)
advent_gan.initialize(sess)

# Finalize the graph
tf.compat.v1.get_default_graph().finalize()

print('Beginning training...')

discriminator_step = -1
progress_tracker = ProgressTracker(total_discriminator_steps)

while discriminator_step < total_discriminator_steps + 1:

    # Training Step
    current_discriminator_step, fraction_of_steps_with_discriminator_training = advent_gan.train_step()

    # Test if the discriminator has been trained
    if current_discriminator_step > discriminator_step:

        # Log progress in the console every 10 discriminator steps
        if current_discriminator_step % 10 == 0:
            print(
                progress_tracker.progress(current_discriminator_step),
                'f =', fraction_of_steps_with_discriminator_training
            )

        # Save decoy images every 5000 steps
        if current_discriminator_step % 5000 == 0:
            save_name = path + file_name_header + '_' + str(current_discriminator_step)
            print('Saving decoy and images to', save_name)
            decoy.save_variables(save_name + '.obj', sess)
            save_batch_plot(sess.run(decoy.generated_image), save_name + '.pdf', image_type='nchw')

        discriminator_step = current_discriminator_step



