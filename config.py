import os
import torch

class config():
    DATA_PATH = os.path.join('','data', 'global-wheat-detection')
    BATH_SIZE = 8
    STEP_SIZE=5
    CHECK_POINT_STEPS = 100
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')