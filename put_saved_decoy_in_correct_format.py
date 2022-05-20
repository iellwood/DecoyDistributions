import pickle
import numpy as np
import ListOfModels

models_to_reformat = [
    'Advent_MNIST_Fixed_Width_32_32_x_32',
    'Advent_MNIST_Fixed_Width_128_32_x_32',
    'Advent_MNIST_Fixed_Width_512_32_x_32',
    'Advent_MNIST_Fixed_Width_2048_32_x_32',
    'Advent_MNIST_Fixed_Width_4096_32_x_32',

    'Advent_CIFAR10_Fixed_Width_32_32_x_32',
    'Advent_CIFAR10_Fixed_Width_128_32_x_32',
    'Advent_CIFAR10_Fixed_Width_512_32_x_32',
    'Advent_CIFAR10_Fixed_Width_2048_32_x_32',
    'Advent_CIFAR10_Fixed_Width_4096_32_x_32',
]

widths = {
    'Advent_MNIST_Fixed_Width_32_32_x_32': 32,
    'Advent_MNIST_Fixed_Width_128_32_x_32': 128,
    'Advent_MNIST_Fixed_Width_512_32_x_32': 512,
    'Advent_MNIST_Fixed_Width_2048_32_x_32': 2048,
    'Advent_MNIST_Fixed_Width_4096_32_x_32': 4096,

    'Advent_CIFAR10_Fixed_Width_32_32_x_32': 32*3,
    'Advent_CIFAR10_Fixed_Width_128_32_x_32': 128*3,
    'Advent_CIFAR10_Fixed_Width_512_32_x_32': 512*3,
    'Advent_CIFAR10_Fixed_Width_2048_32_x_32': 2048*3,
    'Advent_CIFAR10_Fixed_Width_4096_32_x_32': 4096*3,
}

for key in models_to_reformat:

    model_info = {
        'name': ListOfModels.model_data_set[key] + '_decoy_with_distribution_of_log_likelihoods_width_' + str(widths[key]),
        'image_shape': [50, 32, 32, ListOfModels.model_number_of_channels[key]],
        'prior_info': {
            'prior_type':'uniform_distribution_of_log_likelihoods',
            'W': widths[key],
        },
        'dataset': ListOfModels.model_data_set[key],
        'training_method': 'AdvEnt-GAN',
        'model_description': 'RealNVP normalizing flow network with fixed constant jacobian',
        'Reference': 'github.com/iellwood/DecoyDistributions',
    }

    file_name = ListOfModels.model_path[key]
    data = np.load(file_name, allow_pickle=True)
    if type(data) == np.ndarray:
        data = list(data)
    else: exit()

    save_filename = 'TrainedDecoyDistributions/' + ListOfModels.model_data_set[key] + '/' + ListOfModels.model_data_set[key] + '_Decoy_Width_' + str(widths[key]) + '.obj'

    model_info['variable_saved_values'] = data

    saved_model_file_handle =  open(save_filename, 'wb')
    pickle.dump(model_info, saved_model_file_handle)
    saved_model_file_handle.close()

