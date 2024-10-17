import sys
sys.path.append("../model")
from dual_MBConv_VAE import MBConvNet
import json
import torch
class SimpleTrainer(object):
    def __init__(self, config_path):
        train_config = json.load(open(config_path))
        self.model_config = train_config['model_config']
        self.model_name = self.model_config['model_name']
        self.net = None
        if self.model_name == "dual_MBConv_VAE":
            self.net = MBConvNet(
                in_channel=self.model_config['in_channel'],
                num_class=self.model_config['num_class'],
                residual=self.model_config['residual'],
                channel_list=self.model_config['channel_list'],
                MBConv=self.model_config['MBConv'],
                device=self.model_config['device']
            )
        elif self.model_name == "monai_3D_unet":
            pass
    def train_dual_MBConv_VAE_(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass




Trainer = SimpleTrainer(config_path="../../train_configuration/test.json")
Trainer.train()