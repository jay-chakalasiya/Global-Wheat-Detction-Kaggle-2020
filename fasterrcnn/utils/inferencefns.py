import torch
import numpy as np
import pandas as pd
import os
import cv2


class WheatInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, conf, normalize=False):
        
        # Read sample_submission file
        self.CONFIG = conf
        self.SUBMISSION_DF = pd.read_csv(self.CONFIG.DATA_PATH+'/sample_submission.csv')
        self.IMGS = list(self.SUBMISSION_DF['image_id'])  
        self.NORMALIZE = normalize
    
    def __getitem__(self, idx):
        ''' 
        INPUT : ID(number)
        OUTPUT: Image(np-array)
        '''
        #TODO: handle multi image augmentations
        image_id = self.IMGS[idx]        
         
        #image
        img_path =  os.path.join(self.CONFIG.DATA_PATH, 'test', self.IMGS[idx]+'.jpg')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)/255
        
        
        #IF THE MODEL IS TRAINED ON NORMALIZED IMAGES
        if self.NORMALIZE:
            img = (img-self.CONFIG.MEAN)/self.CONFIG.STD
        
        # Convert it to channel-first format: image(W,H,C) -> (C,W,H)
        img = np.array(np.moveaxis(img, -1, 0), dtype=np.float32)
        
        return image_id, img

    def __len__(self):
        return len(self.IMGS)
    

def collate_fn(batch):
    return tuple(zip(*batch))


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], 
                                                             int(round(j[1][0])), 
                                                             int(round(j[1][1])), 
                                                             int(round(j[1][2]-j[1][0])), 
                                                             int(round(j[1][3]-j[1][1]))))
    return " ".join(pred_strings)

def make_submission_file(image_ids, prediction_strings):
    submission_frame = pd.DataFrame()
    submission_frame['image_id'] = image_ids
    submission_frame['PredictionString'] = prediction_strings
    submission_frame.to_csv('submission.csv', index=False)
    print('file_saved')

