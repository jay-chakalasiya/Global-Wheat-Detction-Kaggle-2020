import os
import torch
from imgaug import augmenters as iaa



def get_augmentors():
    return iaa.Sequential([
        iaa.AddToHueAndSaturation((-60, 60)),
        iaa.Sometimes(.5, iaa.AdditiveGaussianNoise(scale=(20, 80))),
        iaa.flip.Fliplr(p=0.5),
        iaa.flip.Flipud(p=0.5), 
        iaa.Rot90([0,1,2,3]),   
        iaa.Affine(rotate=(-12, 12)),
        #iaa.Sometimes(.5, iaa.ElasticTransformation(alpha=90, sigma=9)), 
        #iaa.Sometimes(.8, iaa.WithBrightnessChannels(iaa.Add((-50, 50)))),#])
        iaa.Sometimes(.8, iaa.size.Crop(keep_size=False))], random_order=True)


class config():
    DATA_PATH = os.path.join('','data', 'global-wheat-detection')
    BATH_SIZE = 1
    STEP_SIZE=5
    CHECK_POINT_STEPS = 100
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    SPLIT = 0.8
    AUG = get_augmentors()
    PRECISION_THRESH = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    

    