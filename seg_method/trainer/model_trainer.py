import sys


sys.path.append("../model")
sys.path.append("../../data_process")
sys.path.append("../train_process")
from dual_MBConv_VAE import MBConvNet
import json
import torch
from data_process.get_data import get_oasis_MRI_path_list, get_cranial_CT_path_list
from data_process.dataset import myDataSet
from train_process.log_func import log_print

class SimpleTrainer(object):
    def __init__(self, config_path):
        # Get training configration
        train_config = json.load(open(config_path))
        self.model_config = train_config['model_config']
        self.hyper_para_config = train_config['hyper_para_config']
        self.training_info_config = train_config['training_info_config']
        self.model_name = self.model_config['model_name']
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_print("INFO", "Use device: {0}".format(self.device))
        # Build model
        if self.model_name == "dual_MBConv_VAE":
            self.net = MBConvNet(
                in_channel=self.model_config['in_channel'],
                num_class=self.model_config['num_class'],
                residual=self.model_config['residual'],
                channel_list=self.model_config['channel_list'],
                MBConv=self.model_config['MBConv'],
                device=self.device
            )
        elif self.model_name == "monai_3D_unet":
            pass

        # Get data path
        if self.training_info_config['dataset_name'] == 'oasis':
            assert (self.training_info_config['dataset_num_class'] == 4
                    or self.training_info_config['dataset_num_class'] == 35),\
            log_print("ERROR","Check training_info_config[dataset_num_class], oasis dataset only allows dataset_num_class to be"
             " 4 or 35")
            img_path_list, label_path_list = get_oasis_MRI_path_list(
                root_path=self.training_info_config["dataset_root_path"],
                num_class=self.training_info_config['dataset_num_class']
            )
        elif self.training_info_config['dataset_name'] == 'cranial':
            pass

        # Build dataset

        # Build dataloader

        # Make log saving direction

        # Make model parameter saving direction

    def get_dataset_(
            self,
            img_path_list,
            label_path_list,
            cutmix,
            map,
            discard,
            merge,
            transform2both,
            transform2img
    ):
        dataset = myDataSet(
            img_paths=img_path_list,
            label_paths=label_path_list,
            cutmix=cutmix,
            map=map,
            discard=discard,
            merge=merge,
            transform2both=transform2both,
            transform2img=transform2img
        )
        return dataset

    def get_dataloader_(
            self,
            img_path_list,
            label_path_list,
            cutmix,
            map,
            discard,
            merge,
            transform2both,
            transform2img,
            batch_size,
            prefetch_factor,
            num_workers
    ):
        pass




    def train_dual_MBConv_VAE_(self):
        x = torch.randn([2,1,32,32,32]).to(self.device)
        print(self.net(x)[0].shape)
        pass
    def train(self):
        pass
    def evaluate(self):
        pass




Trainer = SimpleTrainer(config_path="../../train_configuration/test.json")
Trainer.train_dual_MBConv_VAE_()