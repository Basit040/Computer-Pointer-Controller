# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:20:19 2020

@author: Abdul Basit
"""

import numpy as np #Importing numpy library
import cv2  #Importing OpenCV
# Importing IENetwork and IECore
from openvino.inference_engine import IENetwork, IECore


class FacialLandmarksDetectionModel:
    '''
    Class for the Facial Landmarks Model.
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
        # Taking th output
        coords = self.preprocess_output(outputs)
        # Taking the coordinates
        height=image.shape[0]
        # Taking the height from input shape
        width=image.shape[1]
        # Taking the width from input shape
        # coords = coords* np.array([w, h, w, h])
        # coords = coords.astype(np.int32) 
        # Converting the coord in int32 form
        coords=[int(coords[0]*width),int(coords[1]*height),int(coords[2]*width),int(coords[3]*height)]
        # Now calculating the right and left eye points from coords, calculated above
        #(lefteye_x, lefteye_y, righteye_x, righteye_y)
        
        # Now take a variable for eye surroundng area and valued as 10
        eye_surrounding_area=10
        
        # Now calculate the min max point for left eye 
        lefteye_xmin=coords[0]-eye_surrounding_area
        lefteye_ymin=coords[1]-eye_surrounding_area
        lefteye_xmax=coords[0]+eye_surrounding_area
        lefteye_ymax=coords[1]+eye_surrounding_area
        
        # Now calculate the min max point for right eye
        righteye_xmin=coords[2]-eye_surrounding_area
        righteye_ymin=coords[3]-eye_surrounding_area
        righteye_xmax=coords[2]+eye_surrounding_area
        righteye_ymax=coords[3]+eye_surrounding_area
        
        # Now create an image for left eye by using above left eye min max values
        left_eye =  image[lefteye_ymin:lefteye_ymax, lefteye_xmin:lefteye_xmax]
        # Now create an image for left eye by using above left eye min max values
        right_eye = image[righteye_ymin:righteye_ymax, righteye_xmin:righteye_xmax]
        
        # Creating left eye coordinates
        lefteye_coords= [lefteye_xmin,lefteye_ymin,lefteye_xmax,lefteye_ymax]
        # Creating left eye coordinates
        righteye_coords= [righteye_xmin,righteye_ymin,righteye_xmax,righteye_ymax]
        # Now creating eye coords variable
        eye_coords = [lefteye_coords, righteye_coords]
        return left_eye, right_eye, eye_coords
        # return left_eye, right_eye and eye_coords
        
    #def check_model(self):
        ''

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
            The expected color order is BGR and shape: [1x3x48x48]'''
        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        preprocessed_img = preprocessed_img.transpose(2, 0, 1)
        preprocessed_img = preprocessed_img.reshape(1, *preprocessed_img.shape)
        return preprocessed_img
        
        ''' Above code can also be written as:
            image = cv2.resize(image, (w, h))
            pp_image = image.transpose((2, 0, 1))
            pp_image = pp_image.reshape(1, *pp_image.shape)
            return pp_image
        '''
        
            

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        '''The net outputs a blob with the shape: [1, 10], containing a row-vector of 
        10 floating point values for five landmarks coordinates in the 
        form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized 
        to be in range [0,1]'''
        output= outputs[self.output_names][0]
        
        # Determine Coordinates of eye
        lefteye_x = output[0].tolist()[0][0]
        lefteye_y = output[1].tolist()[0][0]
        righteye_x = output[2].tolist()[0][0]
        righteye_y = output[3].tolist()[0][0]

        box = (lefteye_x, lefteye_y, righteye_x, righteye_y)
        return box
        