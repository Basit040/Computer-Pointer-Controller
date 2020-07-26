# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:17:58 2020

@author: Abdul Basit
"""

import numpy as np #Importing numpy library
import cv2  #Importing OpenCV
# Importing IENetwork and IECore
from openvino.inference_engine import IENetwork, IECore


class HeadPoseEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        
        self.device = device
        # String contains device name
        self.model_name = model_name
        # String contains model name
        self.extensions = extensions
        # String contains extensions
        self.network = None
        # String contains network attribute
        self.plugin = None
        # String contains plugin attribute
        self.exec_net = None
        self.model_structure = self.model_name
        # String contains model structure path i.e. .xml
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        # String contains model weights path i.e. .bin
        self.input_name = None
        # String contains input_name attribute
        self.output_names = None
        # String contains output_names attribute
        self.input_shape = None
        # String contains input_shape attribute
        self.output_shape = None
        # String contains output_shape attribute
        
        
        try:
            self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Did you enter the correct model path? Could not Initialise the network")

        # A tuple of the input shape : input_shape
        # A list of output name : output_name
        # A tuple of the output shape : output_shape
        
        
        self.input_name = next(iter(self.network.inputs))
        # Get the name of the input node
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        # Get the name of the output node
        self.output_shape = self.network.outputs[self.output_names].shape
        #raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # Initialize the plugin
        self.plugin = IECore()
        # Loading the model in the plugin
        self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        
        # Now working with unsupported layers
        # Extracting supporting layers 
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        
        # Take unsupported layers from supported layers
        unsupported_layers = [lyr for lyr in self.network.layers.keys() if lyr not in supported_layers]
        
        # Following code will check if any unsupported layer is there and if it is there, it will ask will ask for extension to resolve them
        if len(unsupported_layers) > 0:
            print("Check extention for these unsupported layers:" + str(unsupported_layers))
            exit(1)
        print("All the layers are supported..")
        
        
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        
        #raise NotImplementedError       
    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        preprocessed_img = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:preprocessed_img})
        finalOutput = self.preprocess_output(outputs)
        return finalOutput
    
    def preprocess_input(self, image):
        
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        ''' An input image in the format [BxCxHxW], where:

            B - batch size
            C - number of channels
            H - image height
            W - image width
            The expected color order is BGR and shape:  [1x3x60x60]'''
        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_img = preprocessed_img.transpose(2, 0, 1)
        preprocessed_img = preprocessed_img.reshape(1, *preprocessed_img.shape)
        return preprocessed_img
    
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        '''Output layer names in Inference Engine format:

            name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
            name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
            name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).'''

        return np.array([outputs['angle_y_fc'][0][0], outputs['angle_p_fc'][0][0], outputs['angle_r_fc'][0][0]])
    
   