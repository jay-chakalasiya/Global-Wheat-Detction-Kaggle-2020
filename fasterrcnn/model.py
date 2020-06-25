import torch
import torchvision



def get_model(device, num_classes=1,
              saved_weights=None): # saved_weights=None if no model is saved
    
    if saved_weights:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                     num_classes=num_classes+1,
                                                                     pretrained_backbone=False)
        try:
            ret = model.load_state_dict(torch.load(saved_weights, map_location=device) , strict=False)
            
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print('\n\nSome error occured during loading weigths, please check path and compatibility')
            
       
            
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                     pretrained_backbone=True)
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features, 
                                                                  num_classes+1, 
                                                                  bias=True)
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features, 
                                                                  (num_classes+1)*4, 
                                                                  bias=True)
        print('new_model_created...')
        
    return model.to(device)


