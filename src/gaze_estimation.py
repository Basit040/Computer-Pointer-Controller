# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:48:26 2020

@author: Abdul Basit
"""

import numpy as np #Importing numpy library
import cv2  #Importing OpenCV
# Importing IENetwork and IECore
from openvino.inference_engine import IENetwork, IECore
import math


class GazeEstimationModel:
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
        
    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_preprocessed_img, re_preprocessed_img = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        # Detecting right and left eye processed image
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_preprocessed_img, 'right_eye_image':re_preprocessed_img})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)
        
        

        return new_mouse_coord, gaze_vector
        # Extracting mouse coordinates and gaze vector
         
        
        
    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        
        
        # Firstly preprocessed left eye image
        le_preprocessed_img = cv2.resize(left_eye, (60, 60))
        le_preprocessed_img = le_preprocessed_img.transpose((2, 0, 1))
        le_preprocessed_img = le_preprocessed_img.reshape(1, *le_preprocessed_img.shape)
        
        # Secondly preprocessed right eye image
        re_preprocessed_img = cv2.resize(right_eye, (60, 60))
        re_preprocessed_img = re_preprocessed_img.transpose((2, 0, 1))
        re_preprocessed_img = re_preprocessed_img.reshape(1, *re_preprocessed_img.shape)

        #return p_left_eye, p_right_eye
        return le_preprocessed_img, re_preprocessed_img
    
    def preprocess_output(self, outputs,hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        '''The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates 
        of gaze direction vector. Please note that the output vector is not normalizes and 
        has non-unit length'''
        
        #gaze_vector = outputs[self.output_names[0]].tolist()[0]
        gaze_vector = outputs[self.output_names][0]
        # Extract the gaze vector from outputs
        rollValue = hpa[2] 
        # Extracting rollvalue from head pose angle
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        x_value = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        y_value = -gaze_vector[0] *  sinValue+ gaze_vector[1] * cosValue
        return (x_value,y_value), gaze_vector