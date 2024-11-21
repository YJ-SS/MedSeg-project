import os

import torch
from train_process.record_func import log_print
class EarlyStopping(object):
    def __init__(self, patience:int, model_config, model_save_path:str, mode='min'):
        self.patience = patience
        self.mode = mode
        self.model_config = model_config
        self.model_save_path = model_save_path
        self.counter = 0
        if mode == 'min':
            self.record_val = float('inf')
        elif mode == 'max':
            self.record_val = float('-inf')
        else:
            log_print('ERROR', "Parameter 'mode' must be 'min' or 'max', current  'model' is {0}.".format(mode))

    def save_model_(self, epoch: int, net: torch.nn.Module):
        '''
        Save best model and record current epoch
        :param net:
        :param epoch:
        :return:
        '''
        model_save_path = os.path.join(
            self.model_save_path,
            "{0} best.pth".format(self.model_config['model_name'])
        )
        torch.save(
            {
                'epoch': epoch,
                "model_name": self.model_config['model_name'],
                "in_channel": self.model_config['in_channel'],
                "num_class": self.model_config['num_class'],
                'model_state_dict': net.state_dict()
            },
            model_save_path
        )

    def check_and_early_stop(self, epoch:int, net:torch.nn.Module, value:float)->bool:
        if self.mode == 'min':
            # Save model parameter with min value. Eg: save model parameter with min loss
            if self.record_val > value:
                self.counter = 0
                # Record smaller value
                self.record_val = value
                log_print("INFO", "Update best record, current record={0}".format(self.record_val))
                # Do not early stop, save model parameter
                self.save_model_(epoch, net)
                return False
            elif self.record_val <= value:
                self.counter += 1
                log_print("INFO", "Early stopping... {0}/{1}".format(self.counter, self.patience))
                # counter >= patience, return True to stop training
                if self.counter >= self.patience:
                    log_print("INFO", "Early stop!!!")
                    return True
                return False
        elif self.mode == 'max':
            # Save model parameter with max value. Eg: save model parameter with max dice score.
            if self.record_val < value:
                self.counter = 0
                # Record bigger value
                self.record_val = value
                log_print("INFO", "Update best record, current record={0}".format(self.record_val))
                # Update model parameter
                self.save_model_(epoch, net)
                return False
            elif self.record_val >= value:
                self.counter += 1
                log_print("INFO", "Early stopping... {0}/{1}".format(self.counter, self.patience))
                if self.counter >= self.patience:
                    log_print("INFO", "Early stop!!!")
                    return True
                return False


