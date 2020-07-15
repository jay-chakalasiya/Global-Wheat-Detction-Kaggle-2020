import torch
import numpy as np
import pandas as pd
import os
import cv2

# TODO : Import Augmentatin functions from augmentation package


class WheatDataset(torch.utils.data.Dataset):
    def __init__(self, config, is_train=True, augmentation=True, normalize=True, augmentor=None, random_seed=0):

        # Read dataset file
        self.CONFIG=config
        self.TRAIN_DF = pd.read_csv(os.path.join(self.CONFIG.DATA_PATH, 'train.csv'))

        # Remove some big bounding boxes which are assigned wrongly in dataset
        self.DROP_INDICES = [173, 3687, 4412, 113947, 117344, 118211, 121633, 121634, 147504, 147552]
        self.TRAIN_DF = self.TRAIN_DF.drop(self.TRAIN_DF.index[self.DROP_INDICES])

        # check if augmentation to do or not
        self.IS_AUGMENT = augmentation
        self.AUGMENTOR = augmentor

        # check to do perform normalization or not
        self.NORMALIZE = normalize

        # extract unique Image_ids from the dataframe
        self.UNIQUE_IMAGE_IDS = self.TRAIN_DF.image_id.unique()

        # Get training and testing splits
        self.END_IDX = int(len(self.UNIQUE_IMAGE_IDS)*self.CONFIG.SPLIT)
        self.IMGS = self.UNIQUE_IMAGE_IDS[:self.END_IDX] if is_train else self.UNIQUE_IMAGE_IDS[self.END_IDX:]

        # fix random seeds to get same results again
        np.random.seed(random_seed)

    def parse_bbox_string(self, bbox_string):
        '''
        INPUT : bbox_starting = '[xmin ymin width height]'
        OUTPUT: numpy-array - [xmin, ymin, xmax, ymax]
        will chnage the box format to detr-format at the end because augmentation requires this format
        '''
        parsed_string = bbox_string.strip('[').strip(']').split(', ')
        parsed_values = list(map(float, parsed_string))
        parsed_values[2]+=parsed_values[0]
        parsed_values[3]+=parsed_values[1]
        parsed_values = np.array(parsed_values, dtype=np.float32)
        return parsed_values


    def __getitem__(self, idx):
        '''
        INPUT : ID(number)
        OUTPUT: Image(np-array), {'boxes': np-array(n,4), 'labels': np-array(n)} where n is number of objects for that class
        * class ids starts at 1, 0=background
        '''
        #TODO: handle multi image augmentations
        image_id = self.IMGS[idx]

        # boxes
        bbox_strings = self.TRAIN_DF[self.TRAIN_DF.image_id==image_id].bbox
        if len(bbox_strings)==0:
            # return some other image if boxes are zero in current image
            print('jumped')
            return self.__getitem__(np.random.randint(self.__len__()))
        boxes = np.array([self.parse_bbox_string(bbox_string) for bbox_string in bbox_strings], dtype=np.float32)
        labels = np.array([0]*len(boxes), dtype=np.int64)

        #image
        img_path =  os.path.join(self.CONFIG.DATA_PATH, 'train', self.IMGS[idx]+'.jpg')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255

        if self.IS_AUGMENT:
            current_img = {'img':img, 'boxes':boxes, 'labels':labels}

            # Augmentation
            augmented_dir = self.AUGMENTOR.augment(current_img)

            img, boxes, labels = augmented_dir['img'], augmented_dir['boxes'], augmented_dir['labels']

            # return some other image if boxes are zero in current image
            # Boxes can change after augmentation due to some opereations like cropping and rotation
            if len(boxes)<1:
                return self.__getitem__(np.random.randint(self.__len__()))

        #I'm performing normalization on image after augmentation, because some augmentors operate on colors and brightness
        if self.NORMALIZE:
            img = (img-self.CONFIG.MEAN)/self.CONFIG.STD

        # Convert it to channel-first format: image(W,H,C) -> (C,W,H)
        img = np.array(np.moveaxis(img, -1, 0), dtype=np.float32)

        #Retinanet requires combined boxes+labels
        annotations = np.concatenate((boxes,np.expand_dims(labels,axis=1)),axis=1)

        return img, annotations #{'boxes':boxes, 'labels':labels}

    def __len__(self):
        return len(self.IMGS)



def collate_fn(batch):
    return tuple(zip(*batch))
