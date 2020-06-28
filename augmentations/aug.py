from augmentations.transforms import get_transforms
from augmentations.multiimg import fuse_whole, fuse_same_cropped_parts, fuse_diff_cropped_parts
import numpy as np

# TODO: Multi Image Augmentation

# TODO: Single Image Augmentations

class get_augmentor():
    def __init__(self, min_visibility=0.3, blur = [0.15, 5], gaussian_noise = [0.15, 5/255], rgb_shift = [0.2, 15/255], dropout = 0.1):
        self.transfrom_multi = get_transforms(multi=True, min_visibility=min_visibility, blur = blur, gaussian_noise = gaussian_noise, rgb_shift = rgb_shift, dropout = dropout)
        self.transfrom_single = get_transforms(min_visibility=min_visibility, blur = blur, gaussian_noise = gaussian_noise, rgb_shift = rgb_shift, dropout = dropout)
    
    def get_single_augmentor(self):
        return self.transfrom_single
    
    def get_multi_augmentor(self):
        return self.transfrom_multi
    
    def multi_augment(self, img1_dir, img2_dir):
        return fuse_whole(img1_dir, img2_dir)
    
    def augment(self, img_dir):
        annotations = {'image': img_dir['img'], 
                      'bboxes': img_dir['boxes'], 
                      'category_id': img_dir['labels']}
        
        augmented_image = self.transfrom_single(**annotations)
        
        
        return {'img': augmented_image['image'], 
                'boxes': np.array([
                    np.array(box_tuple, dtype=np.float32) 
                    for box_tuple 
                    in augmented_image['bboxes']]), 
                'labels': np.array(augmented_image['category_id'], dtype=np.int64)}
        
        
    