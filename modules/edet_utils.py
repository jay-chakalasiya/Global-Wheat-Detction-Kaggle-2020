import torch
import numpy as np
import pandas as pd
import os
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torch.nn as nn


class WheatDataset_edet(torch.utils.data.Dataset):
    def __init__(self, config, train=True, augmentation=True, random_seed=0, transform=None):       # transforms,
        
        self.CONFIG=config
        self.TRAIN_DF = pd.read_csv(os.path.join(self.CONFIG.DATA_PATH, 'train.csv'))
        self.drop_indices = [173, 3687, 4412, 113947, 117344, 118211, 121633, 121634, 147504, 147552]
        self.TRAIN_DF = self.TRAIN_DF.drop(self.TRAIN_DF.index[self.drop_indices])
        self.transform = transform
        
        #self.END_IDX = int(len(os.listdir(os.path.join(self.CONFIG.DATA_PATH, 'train')))*self.CONFIG.SPLIT)
        self.END_IDX = int(len(self.TRAIN_DF.image_id.unique())*self.CONFIG.SPLIT)
        self.AUGMENT = augmentation
        if train==True:  
            #self.IMGS = [img.split('.')[0] for img in os.listdir(os.path.join(self.CONFIG.DATA_PATH, 'train'))[:self.END_IDX]]
            self.IMGS = self.TRAIN_DF.image_id.unique()[:self.END_IDX]
        else:
            #self.IMGS = [img.split('.')[0] for img in os.listdir(os.path.join(self.CONFIG.DATA_PATH, 'train'))[self.END_IDX:]]
            self.IMGS = self.TRAIN_DF.image_id.unique()[self.END_IDX:]
        np.random.seed(random_seed)
        
    def parse_bbox_string(self, bbox_string):
        parsed_string = bbox_string.strip('[').strip(']').split(', ')
        parsed_values = list(map(float, parsed_string))
        parsed_values[2]+=parsed_values[0]
        parsed_values[3]+=parsed_values[1]
        return np.array(parsed_values, dtype=np.float32)
    
    def augment_img(self, img, boxes):  # Boxes = (x1, y1, x2, y2), img = channel last (W, H, C)
        b_boxes = BoundingBoxesOnImage([BoundingBox(x1=b[0], x2=b[2], y1=b[1], y2=b[3]) for b in boxes], shape=img.shape)
        img, aug_boxes = self.CONFIG.AUG(image=img, bounding_boxes=b_boxes)
        aug_boxes = aug_boxes.remove_out_of_image_fraction(.6)
        if len(aug_boxes)>0:
            aug_boxes = np.array(([list(b[0])+list(b[1]) for b in aug_boxes]), dtype=np.float32)
        else:
            aug_boxes = np.array([], dtype=np.float32)
        return img, aug_boxes
        
    
    def __getitem__(self, idx):
        img_path =  os.path.join(self.CONFIG.DATA_PATH, 'train', self.IMGS[idx]+'.jpg')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        
        bbox_strings = self.TRAIN_DF[self.TRAIN_DF.image_id==self.IMGS[idx]].bbox
        
        if len(bbox_strings)>0:
            boxes = np.array([self.parse_bbox_string(bbox_string) for bbox_string in bbox_strings], dtype=np.float32)
            labels = np.array([1]*len(boxes), dtype=np.int64)
            
            if self.AUGMENT:
                img, boxes = self.augment_img(img, boxes)
                labels = np.array([], dtype=np.int64) if boxes==np.array([], dtype=np.float32) else np.array([1]*len(boxes), dtype=np.int64)
        else:
            boxes = np.array([], dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        labels = np.array([[1]]*len(labels))
        
        annot = np.concatenate((boxes, labels), 1)
        img = np.array(img, dtype=np.float32)/255
        sample = {'img': img, 'annot':annot}
        
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def __len__(self):
        return len(self.IMGS)


def feeze_backbone(model):
    classname = model.__class__.__name__
    for ntl in ['EfficientNet', 'BiFPN']:
        if ntl in classname:
            for param in model.parameters():
                param.requires_grad = False
    model.apply(freeze_backbone)
    return model


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
    

