import os
import torch
import numpy as np


class config():

    DATA_PATH = os.path.join('','data', 'global-wheat-detection')

    # Model_weight_path
    WEIGHT_PATH = os.path.join('','saved_weights', 'fasterrcnn')

    # Training Batch Size
    BATCH_SIZE = 10

    # Number of steps at which the results will be printed
    STEP_SIZE=5

    # Step at which model is saved
    CHECK_POINT_STEPS = 100

    # training Device
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # training split fraction
    SPLIT = 0.8

    # Whether to perform augmentation or not - which further can be changed in sugmentation module
    AUGMENT = False

    # Mean and Std of Images
    MEAN = np.array([0.31528999, 0.31725333, 0.21455572]) # in RGB - cv2 returns BGR 
    STD = np.array([0.1226716 , 0.10225389, 0.06746761])

    # required for scoring calculation as per competition rules for evaluation
    PRECISION_THRESH = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    # TODO: Implement optimizer and scheduler