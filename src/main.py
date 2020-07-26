# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:36:43 2020

@author: Abdul Basit
"""
# Importing Relevant Libraries for pointer controller application
# os, sys, time, OpenCV, numpy
import os
import sys
import time 
import cv2
import numpy as np

from argparse import ArgumentParser
import logging as log

# Importing Classes from the files which we have created for intel pre trained models

# Face Detection Model
from face_detection import FaceDetectionModel
# Facial Landmark Detection Model
from facial_landmarks_detection import FacialLandmarksDetectionModel
# Gaze Estimation Model
from gaze_estimation import GazeEstimationModel
# Head Pose Estimation Model
from head_pose_estimation import HeadPoseEstimationModel
# Mouse Controller
from mouse_controller import MouseController
# Input Feeder
from input_feeder import InputFeeder


def build_argparser():
    '''
    Parse command line arguments.

    :return: command line arguments
    '''
    parser = ArgumentParser()
    
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to Intel Pre Trained Face Detection model .xml file.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to Intel Pre Trained Facial Landmark Detection model .xml file.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to Intel Pre Trained Head Pose Estimation model .xml file.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to Intel Pre Trained Gaze Estimation model .xml file.")
                        
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
                        
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="targeted custom layers (CPU).")
                             
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.5,
                        help="Probability threshold for detection fitering.")
                        
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+', default=[],
                        help="Specify the flags from f, fl, hp, g to Visualize different model output on frame"
                             "f: Face Detection Model,       fl: Facial Landmark Detection Model"
                             "hp: Head Pose Estimation Model, g: Gaze Estimation Model")
    
    return parser



def main():

    # Grabing command line args
    args = build_argparser().parse_args()
    # Getting Input File Path
    inputFilePath = args.input
    # For Visualization
    visual_flag = args.visualization_flag
    # Initialize inputfeeder
    inputFeeder = None
    
    # Handle video file or CAM (like webcam)
    if args.input =="CAM":
            inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(args.input):
            log.info("Unable to find specified video file")
            sys.exit(1)
        inputFeeder = InputFeeder("video",args.input)

    
    # Now define model path dictionary for all 04 intel pre trained models
    modelPathDict = {'FaceDetectionModel':args.face_detection_model, 'FacialLandmarksDetectionModel':args.facial_landmark_model, 
    'GazeEstimationModel':args.gaze_estimation_model, 'HeadPoseEstimationModel':args.head_pose_model}
    
    # Check model XML file
    for fileNameKey in modelPathDict.keys():
        if not os.path.isfile(modelPathDict[fileNameKey]):
            log.info("Unable to find specified "+fileNameKey+" xml file")
            sys.exit(1)
    
    # Defining Intel Pre Trained Models Objects
    fdm = FaceDetectionModel(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    # Determining Precision and Speed for mouse controller 
    mc = MouseController('medium','fast')
    
    # Loading Input Feeder
    inputFeeder.load_data()
    
    # Loading our four pre trained models and calculate the total models loading time
    # This will help us to find different model time for different models precison like F32,F16 & INT8
    
    start_time_1= time.time()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    total_model_load_time= (time.time()-start_time_1)
    print("Total Model Load Time for All our Intel Pre Trained Models is (in seconds): {:.3f}".format(total_model_load_time))
    # Above print statement will give total model load time for our 04 models for different precisions as well
    
    
    frame_count = 0
    start_time = time.time()
    
    # Start Loop till break through input feeder
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video',cv2.resize(frame,(450,450)))
    
        key = cv2.waitKey(60)
        # Extracting face detection features
        croppedFace, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(croppedFace)==int:
            log.info("Unable to detect the face.")
            if key==27:
                break
            continue
        
        # Head position detection
        hp_out = hpem.predict(croppedFace.copy())
        
        # Landmarks detection (left_eye, right_eye, eyes coordinates)
        left_eye, right_eye, eye_coords = fldm.predict(croppedFace.copy())
        
        # Mouse coordinates and gaze vector Detection
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        
        # Creating variables for visualisation
        # Extracting four face coordinates for rectangle (xmin,ymin,xmax,ymax) 
        x_minimum= face_coords[0]
        y_minimum=face_coords[1]
        x_maximum=face_coords[2]
        y_maximum=face_coords[3]
        
        # Take eye surrounding area
        eye_surrounding_area=10
        
        # Now extracting few features from eye coordinates
        # Extracting four coordinates of left eye from eye coordinates
        l_l1= eye_coords[0][0]
        l_l2=eye_coords[0][1]
        l_l3=eye_coords[0][2]
        l_l4=eye_coords[0][3]
        
        # Extracting four coordinates of left eye from eye coordinates
        r_r1=eye_coords[1][0]
        r_r2=eye_coords[1][1]
        r_r3=eye_coords[1][2]
        r_r4=eye_coords[1][3]
        
        # Extracting pose angle, pitch and roll from head pose output
        pose_angle= hp_out[0]
        pitch=hp_out[1]
        roll=hp_out[2]
            
        # Visualizing face, landmarks, head pose and gaze
        if (not len(visual_flag)==0):
            preview_frame = frame.copy()
            if 'fd' in visual_flag:
                # Drawing a rectangle with our four face coordiantes (xmin,ymin,xmax,ymax)
                cv2.rectangle(preview_frame, (x_minimum, y_minimum), (x_maximum, y_maximum), (20,20,150), 3)
                
            if 'fld' in visual_flag:
                # Drawing a rectangle for each eyes with the help of eye coordinates and eye surrounding area
                # Left Eye
                cv2.rectangle(preview_frame, (l_l1-eye_surrounding_area, l_l2-eye_surrounding_area), (l_l3+eye_surrounding_area, l_l4+eye_surrounding_area), (60,255,0), 2)
                # Right Eye
                cv2.rectangle(preview_frame, (r_r1-eye_surrounding_area, r_r2-eye_surrounding_area), (r_r3+eye_surrounding_area, r_r4+eye_surrounding_area), (60,255,0), 2)
                
            if 'hp' in visual_flag:
                # We have extracted pose angle, pitch and roll from head pose output, now we put text on preview_frame
                cv2.putText(preview_frame, "Pose Angles:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(pose_angle, pitch, roll), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 60), 1)
                
                
            if 'ge' in visual_flag:
                # Calculating coordinates for left eye to obtain left eye center
                le_x= (l_l1 + l_l3)/2
                le_y= (l_l2 + l_l4)/2
                # Calculating coordinates for right eye to obtain right eye center
                re_x= (r_r1 + r_r3)/2
                re_y= (r_r2 + r_r4)/2
                # Calculating left eye center
                le_center= int(x_minimum + le_x), int(y_minimum + le_y)
                # Calculating right eye center
                re_center= int(x_minimum + re_x), int(y_minimum + re_y)
                # Now put both eyes center in a list                
                eyes_center = [le_center, re_center ]
                # Extracting left eye x and y coordinates from eyes_center
                le_center_x = int(eyes_center[0][0])
                le_center_y = int(eyes_center[0][1])
                # Extracting right eye x and y coordinates from eyes_center
                re_center_x = int(eyes_center[1][0])
                re_center_y = int(eyes_center[1][1])
                # Extracting x and y (first and second) value from gaze_vector
                g_x, g_y = gaze_vector[0:2]
                
                # With the help of above parameters, draw arrowed lines for gaze on left and right eyes
                cv2.arrowedLine(preview_frame, (le_center_x, le_center_y), (le_center_x + int(g_x * 100), le_center_y + int(-g_y * 100)), (0,50,160), 1)
                cv2.arrowedLine(preview_frame, (re_center_x, re_center_y), (re_center_x + int(g_x * 100), re_center_y + int(-g_y * 100)), (0,50,160), 1)
                
            
            cv2.imshow("visualization",cv2.resize(preview_frame,(450,450)))
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0],new_mouse_coord[1])    
        if key==27:
            break
    log.info("VideoStream has been ended")
    cv2.destroyAllWindows()
    inputFeeder.close()
    
    # Calculating Inference time and frame per seconds
    total_time = time.time() - start_time
    total_inference_time=total_time
    fps=frame_count/total_inference_time
    print("Inference time: {:.3f}".format(total_inference_time))
    print("FPS: {}".format(fps))
    
   

if __name__ == '__main__':
    main() 
 
