"""
Example code for loading decoy distributions from the paper,
 "Validating density estimation models with decoy distributions"

Run it with

``python ExampleDecoyDistributionLoad.py -f ModelFileName``

Written in tensorflow 1.15

This file shows how to load one of the pretrained decoy distributions from a filename.
It plots samples from the distribution sorted by log-likelihood and a histogram of log-probabilities.
"""


from DecoyDistributionModel.DecoyDistribution import DecoyDistribution
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(
        description='Loads a pretrained decoy distribution model, plots samples from the distribution and plots a histogram of the the log-likelihoods of the distribution.')

parser.add_argument("-f", "--filename", help="Path to the model saved data", required=True)

args = parser.parse_args()

filename = args.filename

sess = tf.InteractiveSession()

# Create the decoy distribution
decoy = DecoyDistribution(filename)

tf.compat.v1.global_variables_initializer().run()

# Initialize the variables using the saved values.
# This line must be after tf.compat.v1.global_variables_initializer().run()
decoy.initialize(sess)

# Create a collection of images and their corresponding log-likelihoods
image, log_p = sess.run([decoy.generated_image, decoy.log_p_of_generated_image])

# Sort the images by their log-likelihood
I = np.argsort(log_p)
image = image[I, :, :, :]

# Clip the images to a range [0, 1], for plotting
image = np.minimum(np.maximum(image, 0), 1)

# Plot the images using matplotlib.pyplot.imshow()
fig, axes = plt.subplots(5, 10, figsize=(3, 2))
for i in range(5):
    for j in range(10):
        axes[i][j].imshow(image[j + 10*i, :, :, :]*np.ones(shape=(1, 1, 3)))
        axes[i][j].axis('off')
plt.suptitle('Decoy distribution samples')
plt.show()

# Plot the distribution of log-likelihoods:
log_ps = []
print('Collecting log-likelihoods for histogram...')

for i in range(200):
    if i % 10 == 0:
        print('%i/%i'%(i, 200))
    log_ps.append(sess.run(decoy.log_p_of_generated_image))

plt.hist(np.concatenate(log_ps, 0), 50, density=True, histtype='step', color='k', linewidth=2)
plt.title('Distribution of log-likelihoods')
plt.xlabel('log p')
plt.ylabel('P(log p)')
print('Done')
plt.show()
