


import argparse

def get_info_from_argparse():

    parser = argparse.ArgumentParser(
        description='Trains a decoy distribution on either MNIST or CIFAR10 with fixed width')

    parser.add_argument(
        "-w", "--width", help="Selects the width of the uniform distribution of log-likelihoods.", default="32"
    )

    parser.add_argument(
        "-d", "--dataset", help="Selects the dataset to train on. Either MNIST or CIFAR10", default="MNIST"
    )

    parser.add_argument(
        "-b", "--batch_size", help="Minibatch size", default="50"
    )

    parser.add_argument(
        "-s", "--discriminator_steps", help="Number of discriminator steps to take", default="20000"
    )

    parser.add_argument(
        "-g", "--gpu", help="Select which GPU to use", default="0"
    )

    args = parser.parse_args()

    try:
        width = int(args.width)
    except:
        try:
            width = float(args.width)
        except:
            print('Supplied width, ' + args.width + ', cannot be converted to a float.')
            exit()

    try:
        discriminator_steps = int(args.discriminator_steps)
        if discriminator_steps < 1:
            print('Supplied discriminator steps, ' + args.discriminator_steps + ', must be an integer greater than 0.')
            exit()

    except:
        print('Supplied discriminator steps, ' + args.discriminator_steps + ', must be an integer greater than 0.')
        exit()

    dataset = args.dataset
    if not (dataset.upper() == 'MNIST' or dataset.upper() == 'CIFAR10'):
        print('Unknown dataset, ' + dataset + '.')
        exit()

    try:
        batch_size = int(args.batch_size)
        if int(batch_size) < 50:
            print('Selected batch size, ' + batch_size + ', should be at least 50')
            exit()

        if batch_size > 1024:
            print('Warning: Large batch size,', batch_size)
    except:
        print('Selected batch size, ' + args.batch_size + ', is not a integer >= 50.')
        exit()

    try:
        gpu = str(int(args.gpu))
        if int(gpu) < 0:
            print('Selected GPU, ' + gpu + ', must be an integer > 0')
            exit()
    except:
        print('Selected GPU, ' + args.gpu + ', is not a integer >= 0.')
        exit()

    path = 'DecoyTrainingSavedData/' + dataset + '/Width_' + str(width) + '/'

    file_name_header = dataset + '_decoy_width_' + str(width)

    if dataset == 'CIFAR10':
        channels = 3
        image_shape = [batch_size, 32, 32, 3]
    elif dataset == 'MNIST':
        channels = 1
        image_shape = [batch_size, 32, 32, 1]


    model_info = {
        'name': dataset + '_decoy_with_distribution_of_log_likelihoods_width_' + str(width),
        'image_shape': image_shape,
        'prior_info': {
            'prior_type': 'uniform_distribution_of_log_likelihoods',
            'W': width,
        },
        'dataset': dataset,
        'training_method': 'AdvEnt-GAN',
        'model_description': 'RealNVP normalizing flow network with fixed constant Jacobian',
        'Reference': 'github.com/iellwood/DecoyDistributions',
    }

    return model_info, file_name_header, path, gpu, discriminator_steps
