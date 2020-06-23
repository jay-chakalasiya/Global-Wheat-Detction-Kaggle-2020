from backbone import EfficientDetBackbone

def edet(compound_coef, obj_list):
    
    anchors_scales= '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios= '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    
    model = EfficientDetBackbone(num_classes=len(obj_list), 
                                 compound_coef=compound_coef, 
                                 ratios=eval(anchors_ratios), 
                                 scales=eval(anchors_scales))                  
    return model


    