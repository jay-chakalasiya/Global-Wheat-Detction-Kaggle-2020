from .metric import calculate_image_precision
import numba
import numpy as np
def calculate_model_precision(ground_truths, predictions, 
                              thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75], 
                              form='pascal_voc'):
    iou_thresholds = numba.typed.List()
    for x in thresholds:
        iou_thresholds.append(x)
    image_precisions = np.array([calculate_image_precision(gt, 
                                                           pred, 
                                                           thresholds=iou_thresholds, 
                                                           form=form) 
                                 for gt,pred 
                                 in zip(ground_truths, predictions)])
    return np.mean(image_precisions), image_precisions