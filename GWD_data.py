import torch
import numpy as np
import pandas as pd
import os
import cv2


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, config, random_seed=0):       # transforms,
        
        self.CONFIG=config
        self.TRAIN_DF = pd.read_csv(os.path.join(self.CONFIG.DATA_PATH, 'train.csv'))
        self.IMGS = self.TRAIN_DF.image_id.unique()
        np.random.seed(random_seed)
        
    def parse_bbox_string(self, bbox_string):
        parsed_string = bbox_string.strip('[').strip(']').split(', ')
        parsed_values = list(map(float, parsed_string))
        parsed_values[2]+=parsed_values[0]
        parsed_values[3]+=parsed_values[1]
        return np.array(parsed_values, dtype=np.float32)
    
    def __getitem__(self, idx):
        img_path =  os.path.join(self.CONFIG.DATA_PATH, 'train', self.TRAIN_DF.image_id.unique()[idx]+'.jpg')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255
        img = np.array(np.moveaxis(img, -1, 0), dtype=np.float32)
        
        bbox_strings = self.TRAIN_DF[self.TRAIN_DF.image_id==self.IMGS[idx]].bbox
        labels = np.array([1]*len(bbox_strings), dtype=np.int64)
        boxes = np.array([self.parse_bbox_string(bbox_string) for bbox_string in bbox_strings], dtype=np.float32)
        
        return img, {'boxes':boxes, 'labels':labels}

    def __len__(self):
        return self.IMGS.shape[0]
def collate_fn(batch):
    return tuple(zip(*batch))