'''
Created on Thu Nov 5 16:21:04 2020 
@author: jdickey & rpena

To use this model import it like this: 
    from ArrNet import ArrNet
ArrNet.get_default_par(json_file) to get par and add create json file
ArrNet.get_network(par) to get the full model
'''

import tensorflow as tf

from model_utils import Params
from model_utils import save_dict_to_json 

class ArrNet():

    @staticmethod
    def get_default_par(par_file): 
        '''
        Parameters
        ------
        par_file : model utils.Param
            A json file where the default parameters will be saved. Only call this if you dont know
            what parameters are needed by the model. You may create your own Param object and
            create the model that way if you know the necessary parameters.
        Returns
        ------
        TYPE
        Default Param object.
        '''

        par = { 'lr': 0.001,
                'bs': 1,
                'loss': 'quantile',
                'optimizer':'adam',
                'f_1ow': 0.8,
                'f_high': 4.5,
                'f_pad': 3 ,
                's_rate': 40,
                'w_len': 3,
                'shift': 0,
                'f': 5,
                'd': [2, 4, 8],
                'k': 15,
                's': 1,
                'dense': [3],
                'pat': 20,
                't_step': 1000,
                'v_step': 60,
                'cmps': 'B',
                'project_name': '', 
                'model_name': '', 
                'model_file': '',
                'model_save': '',
                'log_folder': '',
                'data_folder': '',
                'image_folder': '',
                'model_folder': '',
                'trnRET': False,
                'catalog': '',
                'type': 'tcn',
                'quantiles': [0.9772, 0.8413, 0.1538, 0.0228], }

        save_dict_to_json(par, par_file) 
        return Params(par_file)

    def __residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0):
        prev_x = x
        x = tf.keras.layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   dilation_rate=dilation_rate,
                                   padding=padding)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.SpatialDropout1D(rate=dropout_rate)(x)

        # lxl conv to match the shapes (channel dimension)
        prev_x = tf.keras.layers.Conv1D(nb_filters, 1, padding='same')(prev_x) 
        res_x = tf.keras.layers.add([prev_x, x])
        return res_x, x   
    
    
    def __get_loss(q):

        def quantile_loss(q, y_p, y):
            e = y_p-y
            return tf.keras.backend.mean(tf.math.maximum(q*e, (q-1)*e))

        def q_loss(y_true, y_pred):
            return quantile_loss(q, y_true, y_pred) 
        
        return q_loss


    @classmethod
    def get_network(cls, par, padding = 'causal', drop = 0.05):
        nb_chan = len(par.cmps)
        nb_filters = par.f
        filter_len = par.k
        dilations = par.d
        nb_stacks = par.s
        
        input_layer = tf.keras.layers.Input(shape=(None, nb_chan))
        
        x = input_layer
        skip_connections = []
        
        for s in range(nb_stacks):
            for d in dilations:
                x, skip_out = cls.__residual_block(x,
                                                   dilation_rate=d,
                                                   nb_filters=nb_filters,
                                                   kernel_size=filter_len,
                                                   padding=padding,
                                                   dropout_rate=drop)
                
                skip_connections.append(skip_out)

        x = tf.keras.layers.add(skip_connections)
        x = tf.keras.layers.Lambda(lambda tt: tt[:, -1, :])(x)

        for dense in par.dense:
            x = tf.keras.layers.Dense(dense, activation='relu')(x)
            if (len(par.dense) > 1):
                x = tf.keras.layers.Dropout(0.2)(x)

        if (par.loss == 'quantile'):
            loss_array = {} 
            output_layer = []
            out= tf.keras.layers.Dense(1, activation='linear', name='huber')(x)
            output_layer.append(out)
            loss_array['huber'] = tf.keras.losses.Huber() 
            for q in par.quantiles:
                name= f'output_{int(100*q):02d}'
                out= tf.keras.layers.Dense(1, activation='linear', name=name)(x)
                output_layer.append(out)

                loss_array[name] = cls.__get_loss(q)

            model = tf.keras.Model(input_layer, output_layer, name='q_model')
            model.compile(optimizer=par.optimizer, loss=loss_array, metrics=['mean_absolute_error'])
        else :
            output_layer = tf.keras.layers.Dense(1, activation='linear', name='output')(x)
            model = tf.keras.Model(input_layer ,output_layer, name='model') 
            model.compile(optimizer=par.optimizer, loss=par.loss, metrics=['mean_absolute_error'])
        return model