from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import torch
from fasterrcnn.model import get_model

class get_training_handler():
    
    def __init__(self, conf, new=False, starter_dir={'date': '0000', 'version':'00'}):
        
        self.config = conf
        self.model_header = 'GWD_FASTERRCNN_'
        self.headers = ['model_name', 'epoch', 'checkpoint', 'tensorboard_step', 'loss', 'score']
        self.state_path = 'fasterrcnn/fasterrcnn_states.csv'
        
        if not new and os.path.exists(self.state_path):
            self.state_df = pd.read_csv(self.state_path)
            self.model_version = list(self.state_df[self.headers[0]])[-1]
            self.last_epoch = list(self.state_df[self.headers[1]])[-1]
            self.last_checkpoint = list(self.state_df[self.headers[2]])[-1]
            self.tensorboard_step = list(self.state_df[self.headers[3]])[-1]
            self.last_loss = list(self.state_df[self.headers[4]])[-1]
            self.last_score = list(self.state_df[self.headers[5]])[-1]

        
        else:
            self.state_df = pd.DataFrame(columns=self.headers)
            self.model_version = starter_dir['date']+'V'+starter_dir['version']
            self.last_epoch = 0
            self.last_checkpoint = 0
            self.tensorboard_step = 0
            self.last_loss=0
            self.last_score=0
            
        self.writer_path = 'runs/'+self.model_header+self.model_version
        self.writer = SummaryWriter(self.writer_path)
            
    def save_weights(self, model, score, loss, new_epoch=False):
        '''
        model = pytorch model
        '''
        self.last_checkpoint+=1
        if new_epoch:
            self.last_checkpoint=0
            self.last_epoch+=1
            
        saved_weight_name = self.model_header+self.model_version+'_EPOCH_{}_CHECKPOINT_{}_SCORE_{:.4f}_LOSS_{:.4f}.pth'.format(self.last_epoch, self.last_checkpoint, score, loss)
        torch.save(model.state_dict(), os.path.join(self.config.WEIGHT_PATH,saved_weight_name))
        current_state_dir = {self.headers[0]: self.model_version, 
                             self.headers[1]: self.last_epoch, 
                             self.headers[2]: self.last_checkpoint, 
                             self.headers[3]: self.tensorboard_step, 
                             self.headers[4]: score, 
                             self.headers[5]: loss}
        self.state_df = self.state_df.append(current_state_dir, ignore_index=True)
        self.state_df.to_csv(self.state_path, index=False)
        print('weights_saved...')
    
    def publish_to_board(self, total_loss=0, classifier_loss=0, reg_box_loss=0, objectness_loss=0, rpn_reg_loss=0):
        self.tensorboard_step+=1
        self.writer.add_scalar('Running Loss/Summed', total_loss, self.tensorboard_step)
        self.writer.add_scalar('Running Loss/Classifier', classifier_loss, self.tensorboard_step)
        self.writer.add_scalar('Running Loss/Box_Regress', reg_box_loss, self.tensorboard_step)
        self.writer.add_scalar('Running Loss/Objectness', objectness_loss, self.tensorboard_step)
        self.writer.add_scalar('Running Loss/RPN_Box_regress', rpn_reg_loss, self.tensorboard_step)
        
    def get_fasterrcnn(self, external_weight_path=None):
        
        if external_weight_path:
            try:
                model = get_model(saved_weights = external_weight_path)
                return model
            except: 
                pass
                
        weight_name = self.model_header+self.model_version+'_EPOCH_{}_CHECKPOINT_{}_SCORE_{:.4f}_LOSS_{:.4f}.pth'.format(self.last_epoch, self.last_checkpoint, self.last_score, self.last_loss)
        weight_path = os.path.join(self.config.WEIGHT_PATH, weight_name)
        if os.path.exists(weight_path):
            model = get_model(saved_weights = weight_path)
        else:
            model = get_model()
        return model