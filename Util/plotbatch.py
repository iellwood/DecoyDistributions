import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def save_batch_plot(images, filename, title='Decoy distribution samples', image_type='nhwc'):
    if image_type == 'nchw':
        images = np.transpose(images, [0, 2, 3, 1])

    images = np.maximum(np.minimum(images, 1), 0)
    number_of_images = images.shape[0]
    if number_of_images < 50:
        raise Exception('batch size, ' + str(number_of_images) + ', must be at least 50')

    # Plot the images using matplotlib.pyplot.imshow()
    fig, axes = plt.subplots(5, 10, figsize=(5, 4))
    for i in range(5):
        for j in range(10):
            axes[i][j].imshow(images[j + 10 * i, :, :, :] * np.ones(shape=(1, 1, 3)))
            axes[i][j].axis('off')
    plt.suptitle(title)

    plt.savefig(filename, format='pdf', transparent=True)
    plt.close()

