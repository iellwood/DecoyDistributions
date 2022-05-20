"""
Code from "Validating density estimation models with decoy distributions"

Written in tensorflow 1.15

This class constructs decoy distributions using real_nvp.model_rigid.py and real_nvp.nn_rigid

Can also be used to load a pretrained model

"""


import tensorflow as tf
import numpy as np
import DecoyDistributionModel.real_nvp.model_rigid as flow_model
import DecoyDistributionModel.real_nvp.nn_rigid as nn
import DecoyDistributionModel.seed as seed
import pickle

class DecoyDistribution:


    def __init__(self, model_info_or_saved_model_filename, input_image=None):
        """
        Builds a decoy distribution based on the data provided in model_info and, optionally, initializes the network
        with the variables stored in saved_model_filename

        :param model_info_or_saved_variables_filename: Either a dict of the model parameters or a saved model filename.
        If a dict, keys must include 'name', 'image_shape', 'prior_info'. The saved model
        :param saved_variables_filename: An optional filename to load the model
        :param input_image: An optional input_image, stored in DecoyDistribution.image_input. If None, a placeholder
        will be created.
        """

        if type(model_info_or_saved_model_filename) is str:
            saved_model_file_handle =  open(model_info_or_saved_model_filename, 'rb')
            loaded_dict = pickle.load(saved_model_file_handle)
            saved_model_file_handle.close()

            self.model_info = loaded_dict
        else:
            self.model_info = model_info_or_saved_model_filename
            self._model_variable_saved_values = None


        self.name = self.model_info['name']
        self.image_shape = self.model_info['image_shape']
        self.saved_variables_filename = None
        self.prior_info = self.model_info['prior_info']

        # If an input image is provided, overwrite the value of image_shape from the model_info.
        # Note that only the batch_size can be changed in this manner.
        if input_image is not None:
            self.image_shape = input_image.shape.as_list()
        else:
            input_image = tf.placeholder(dtype=tf.float32, shape=self.image_shape)

        self.input_image = input_image


        self.backwards_generator_template_train = tf.make_template('model', lambda x:
            flow_model.model_spec(
                x,
                final_sigmoid=False,
                model_type="real_nvp",
                train=True,
                alpha=1e-7,
                init_type="normal",
                hidden_layers=1000,
                no_of_layers=7,
                batch_norm_adaptive=1,
            ), unique_name_=self.name
        )

        self.backwards_generator_template_no_train = tf.make_template('model', lambda x:
            flow_model.model_spec(
                x,
                final_sigmoid=False,
                model_type="real_nvp",
                train=False,
                alpha=1e-7,
                init_type="normal",
                hidden_layers=1000,
                no_of_layers=7,
                batch_norm_adaptive=1,
            ), unique_name_=self.name
        )

        self.generator_template_train = tf.make_template('model', lambda x:
            flow_model.inv_model_spec(
                x,
                final_sigmoid=False,
                model_type="real_nvp",
                train=True,
                alpha=1e-7,
                init_type="normal",
                hidden_layers=1000,
                no_of_layers=7,
                batch_norm_adaptive=1,
            ), unique_name_=self.name
        )


        self.generator_template_no_train = tf.make_template('model', lambda x:
            flow_model.inv_model_spec(
                x,
                final_sigmoid=False,
                model_type="real_nvp",
                train=False,
                alpha=1e-7,
                init_type="normal",
                hidden_layers=1000,
                no_of_layers=7,
                batch_norm_adaptive=1,
            ), unique_name_=self.name
        )


        self.seed = seed.generate_seed(self.prior_info, self.image_shape)


        # WARNING: the normalizing flow must be applied backwards first, because of the way the RealNVP model is initialized
        self.code, self.log_p_of_input_image = self._apply_backward_no_train(self.input_image - 0.5, return_jac=False)

        self._batchnorm_warmup, _ = self._apply_forward_train(self.seed)

        self.generated_image, self.log_p_of_generated_image = self._apply_forward_no_train(self.seed)
        self.generated_image = self.generated_image + 0.5

        self.variables = None
        self.trainable_variables = None

    def save_variables(self, filename, session):
        """
        Saves the variables of the model

        :param filename: file where the variables will be saved
        :param session: the current tensorflow session
        """
        vars_eval = session.run(self.get_vars(trainable=False))

        self.model_info['variable_saved_values'] = vars_eval

        saved_model_file_handle = open(filename, 'wb')
        pickle.dump(self.model_info, saved_model_file_handle)
        saved_model_file_handle.close()


    def _apply_backward_train(self, x, return_jac=False):
        z, jac = self.backwards_generator_template_train(x)
        log_p = nn.compute_log_density_x(z, jac, self.prior_info)

        if return_jac:
            return z, log_p, jac
        else:
            return z, log_p

    def _apply_backward_no_train(self, x, return_jac=False):
        z, jac = self.backwards_generator_template_no_train(x)
        log_p = nn.compute_log_density_x(z, jac, self.prior_info)


        if return_jac:
            return z, log_p, jac
        else:
            return z, log_p

    def _apply_forward_no_train(self, z, return_jac=False):
        x, jac = self.generator_template_no_train(z)
        log_p = nn.compute_log_density_x(z, -jac, self.prior_info)

        if return_jac:
            return x, log_p, jac
        else:
            return x, log_p

    def _apply_forward_train(self, z, return_jac=False):
        x, jac = self.generator_template_train(z)
        log_p = nn.compute_log_density_x(z, -jac, self.prior_info)

        if return_jac:
            return x, log_p, jac
        else:
            return x, log_p

    def get_vars(self, only_trainable=True):
        if only_trainable:
            if self.trainable_variables is None:
                self.trainable_variables = tf.compat.v1.trainable_variables(scope=self.name)
            return self.trainable_variables
        else:
            if self.variables is None:
                self.variables = tf.compat.v1.global_variables(scope=self.name)
            return self.variables


    def initialize(self, sess):
        if 'variable_saved_values' in self.model_info.keys():
            print('Initializing model with saved variables')
            variables = self.get_vars(only_trainable=False)
            assign_ops = []

            vars = self.model_info['variable_saved_values']

            for i in range(len(variables)):
                if variables[i].shape != vars[i].shape:
                    print(variables[i].shape, vars[i].shape)
                    print(variables[i].name)
                    raise Exception('variables and saved variables have different shape')

                assign_ops.append(tf.assign(variables[i], vars[i]))

            sess.run(assign_ops)

            # Frees up the memory of the numpy array of saved variables
            self.model_info['variable_saved_values'] = None
        else:
            for i in range(1000):
                sess.run(self._batchnorm_warmup)
