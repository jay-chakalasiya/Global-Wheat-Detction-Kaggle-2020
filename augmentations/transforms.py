import albumentations as A


def get_transforms(min_visibility=0., min_area=0., multi=False, blur = [0.15, 5], gaussian_noise = [0.15, 5/255]):
    
    if multi:
        return A.Compose([
                          # Every action will be performed at given probability p

                          # Random Horizontal, Vertical or Horizontal-Vertical Flip
                          A.Flip(p=0.75),

                          # Randomly cropping some area of image
                          A.RandomCrop(p=0.5, height=int(1024*0.8), width=int(1024*0.8)),

                          # Column to Row
                          A.Transpose(p=0.5),

                          # Introducing black patches over image
                          A.CoarseDropout(p=0.5,
                                          max_holes=10, 
                                          min_holes=5, 
                                          max_height=40, 
                                          max_width=40,  
                                          min_height=20, 
                                          min_width=20,
                                          fill_value=0.07),

                          # Introducing white patches over image
                          A.CoarseDropout(p=0.5, 
                                          max_holes=10, 
                                          min_holes=5, 
                                          max_height=40, 
                                          max_width=40,  
                                          min_height=20, 
                                          min_width=20, 
                                          fill_value=0.93),

                          # Shifting HSV value by abut 6%
                          #A.HueSaturationValue(p=0.20, 
                          #                     hue_shift_limit=15/255, 
                          #                     sat_shift_limit=15/255, 
                          #                     val_shift_limit=15/255),

                          # Shifting RGB value by abut 6%
                          A.RGBShift(p=0.20, 
                                     r_shift_limit=15/255, 
                                     g_shift_limit=15/255, 
                                     b_shift_limit=15/255),

                          # Adding Gaiussian Noise
                          A.GaussNoise(p=0.5, 
                                       var_limit=(0, 2/255), 
                                       mean=0, ),

                          # Resizing image to required size by model
                          A.Resize(p=1, 
                                   height=1024, 
                                   width=1024)
                         ], 

                         bbox_params=A.BboxParams(format='pascal_voc', 
                                                  min_area=min_area, 
                                                  min_visibility=min_visibility, 
                                                  label_fields=['category_id'])
                        )
        
    return A.Compose([
                      # Every action will be performed at given probability p
                      
                      # Random Horizontal, Vertical or Horizontal-Vertical Flip
                      A.Flip(p=0.75),
                      
                      # Randomly cropping some area of image
                      A.RandomCrop(p=0.5, height=int(1024*0.8), width=int(1024*0.8)),
        
                      # Column to Row
                      A.Transpose(p=0.5),
        
                      # Rotating Image with 10 degrees
                      A.Rotate(p=0.3, limit=10),
        
                      # Introducing black patches over image
                      A.CoarseDropout(p=0.1,
                                      max_holes=20, 
                                      min_holes=10, 
                                      max_height=50, 
                                      max_width=50,  
                                      min_height=25, 
                                      min_width=25,
                                      fill_value=0.07),
                      
                      # Introducing white patches over image
                      A.CoarseDropout(p=0.1,
                                      max_holes=20, 
                                      min_holes=10, 
                                      max_height=50, 
                                      max_width=50,  
                                      min_height=25, 
                                      min_width=25, 
                                      fill_value=0.93),
        
                      # Random Blur
                      A.Blur(p=0.,
                             blur_limit=5),
        
                      # Shifting HSV value by abut 6%
                      #A.HueSaturationValue(p=0.35, 
                      #                     hue_shift_limit=15/255, 
                      #                     sat_shift_limit=15/255, 
                      #                     val_shift_limit=15/255),
        
                      # Shifting RGB value by abut 6%
                      A.RGBShift(p=0.15,
                                 r_shift_limit=15/255, 
                                 g_shift_limit=15/255, 
                                 b_shift_limit=15/255),
        
                      # Shifting Brighness and Contrast
                      #A.RandomBrightnessContrast(p=0.5, 
                      #                           brightness_limit=1, 
                      #                           contrast_limit=1),
        
                      # Adding Gaiussian Noise
                      A.GaussNoise(p=0.15,
                                   var_limit=(0, 5/255), 
                                   mean=0, ),
                      
                      # Resizing image to required size by model
                      A.Resize(p=1, 
                               height=1024, 
                               width=1024)
                     ], 
                     
                     bbox_params=A.BboxParams(format='pascal_voc', 
                                              min_area=min_area, 
                                              min_visibility=min_visibility, 
                                              label_fields=['category_id'])
                    )