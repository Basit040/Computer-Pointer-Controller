# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:15:45 2020

@author: Abdul Basit
"""
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np #Importing numpy library
import cv2  #Importing OpenCV
# Importing IENetwork and IECore
from openvino.inference_engine import IENetwork, IECore


class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
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

    def predict(self, image,prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        preprocessed_img = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name:preprocessed_img})
        coords = self.preprocess_output(outputs, prob_threshold)
        # Took the coordinates from outputs
        if (len(coords)==0):
            print("No Face has been detected, Next frame will be processed..")
            return 0, 0
        coords = coords[0] 
        #take the first detected face
        
        # Take the height and width from the image.shape
        height=image.shape[0]
        # Take the height from image.shape
        width=image.shape[1]
        # Take the width from image shape
        
        # Calculate the coord from width and height accordingly
        # Convert coords as int type
        coords=[int(coords[0]*width),int(coords[1]*height),int(coords[2]*width),int(coords[3]*height)]
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        # Take the cropped face from coord by slicing
        return cropped_face, coords
        ### It is noted that the above code  can be written in one line and will be processed okay, like this
        ### cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        #raise NotImplementedError

   

    def preprocess_input(self, image):
        '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
        '''
        ''' It preproprocessed input
        An input image in the format [BxCxHxW], where:

            B - batch size (here we use n)
            C - number of channels
            H - image height
            W - image width
        '''
        preprocessed_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # Change image from HWC to CHW
        preprocessed_img = preprocessed_img.transpose((2,0,1))
        preprocessed_img = preprocessed_img.reshape(1, *preprocessed_img.shape)

        return preprocessed_img
        
        #raise NotImplementedError
        ''' Above code can also be written as:
            image = cv2.resize(image, (w, h))
            pp_image = image.transpose((2, 0, 1))
            pp_image = pp_image.reshape(1, *pp_image.shape)
            return pp_image
        '''

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # Rectangle need two coordinates, one is top left corner and second one is bottom right
        # Top left corner will be (xmin,ymin)
        # Bottom right corner will be (xmax,ymax)
        
        # Loop through detections and determine what and where the objects are in the image
        # For each detection , it has 7 values i.e. [image_id,label,conf,x_min,y_min,x_max,y_max]
        # image_id - ID of the image in the batch
        # label - predicted class ID
        # conf - confidence for the predicted class
        # (x_min, y_min) - coordinates of the top left bounding box corner
        # (x_max, y_max) - coordinates of the bottom right bounding box corner
        area = []
        coords = []
        for id, label, confidence, x_min, y_min, x_max, y_max in outputs[self.output_names][0][0]:
            if confidence > prob_threshold:
                width = x_max - x_min
                height = y_max - y_min
                area.append(width * height)
                coords.append([x_min, y_min, x_max, y_max])

                
        return coords
        #raise NotImplementedError
