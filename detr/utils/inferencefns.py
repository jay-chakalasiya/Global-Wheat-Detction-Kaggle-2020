import numpy as np
import pandas as pd
import os
import cv2
import torch
from collections import OrderedDict
import torch.nn.functional as F

import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from detr.detr.util.misc import NestedTensor
from detr.detr.models.detr import DETR
from detr.detr.models.backbone import Joiner, FrozenBatchNorm2d  #, Backbone
from detr.detr.models.transformer import Transformer
from detr.detr.models.position_encoding import PositionEmbeddingSine


def get_detr(backbone_name: str, dilation=False, num_classes=1):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=False, return_interm_layers=False, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)

    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True)
    return DETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100)

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': 0}
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], pretrained=False, norm_layer=FrozenBatchNorm2d)
       
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
        
        
        
        
        
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
        
        # resizing it to 1024 just to be sure
        if img.shape != (1024, 1024, 3):
            img = cv2.resize(img, (1024,1024), interpolation = cv2.INTER_AREA)
        
        
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


        
        
