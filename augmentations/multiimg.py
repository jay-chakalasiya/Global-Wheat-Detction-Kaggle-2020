import numpy as np

def fuse_whole(img1_dir, img2_dir):
    '''
    img1_dir and img2_dir both are directory of format {'img': image_array, 'boxes': boxes, 'labels': labels}
    
    image_array = np arrays (W, H, C) -> C in RGB sequence
    boxes = np arrays (N, 4) -> N=number of boxes -> (xmin, ymin, xmax, ymax)
    labels = N-sized 1-d arrayu
    '''
    
    img = (img1_dir['img'] + img2_dir['img']) / 2
    boxes = np.concatenate((img1_dir['boxes'], img2_dir['boxes']))
    labels = np.concatenate((img1_dir['labels'], img2_dir['labels']))
    
    return {'img':img, 'boxes':boxes, 'labels':labels}
    



def fuse_same_cropped_parts():
    return True
    
    
    
def fuse_diff_cropped_parts():
    return True
    
    

def replace_same_cropped_parts():
    return True
    
    
    
def fuse_diff_cropped_parts():
    return True