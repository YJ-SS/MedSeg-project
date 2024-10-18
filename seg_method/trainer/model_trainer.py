import sys

from torchio import Compose
from tqdm import tqdm
sys.path.append("../model")
sys.path.append("../../data_process")
sys.path.append("../train_process")
from dual_MBConv_VAE import MBConvNet
import json
import torch
from data_process.get_data import get_data_path_list
from data_process.dataset import myDataSet
from data_process.data_process_method import get_dataloader_transform
from train_process.log_func import log_print
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


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
        self.img_path_list = None
        self.label_path_list = None
        self.sup_img_list = None
        self.sup_label_list = None
        self.unsup_img_list = None
        self.unsup_label_list = None
        self.val_img_list = None
        self.val_label_list = None
        self.sup_data_num = self.hyper_para_config['sup_data_num']
        self.unsup_data_num = self.hyper_para_config['unsup_data_num']
        self.val_data_num = self.hyper_para_config['val_data_num']
        self.epoch = self.hyper_para_config['epoch']
        self.sup_dataloader = None
        self.unsup_dataloader = None
        self.val_dataloader = None

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
        self.img_path_list, self.label_path_list = get_data_path_list(
            dataset_name=self.training_info_config['dataset_name'],
            root_path=self.training_info_config['dataset_root_path'],
            num_class=self.training_info_config['dataset_num_class'],
        )

        assert self.img_path_list is not None and self.label_path_list is not None,\
        log_print("ERROR", "Check image path list and label path list, which is None!!!")
        log_print("INFO", "Total image path list length: {0}".format(len(self.img_path_list)))
        log_print("INFO", "Total label path list length: {0}".format(len(self.label_path_list)))
        # Build dataset
        # Build dataloader
        self.sup_dataloader = self.get_dataloader_(
            img_path_list=self.img_path_list[:self.sup_data_num],
            label_path_list=self.label_path_list[:self.sup_data_num],
            stage="supervise"
        )
        log_print("INFO", "Supervised dataloader length: {0}".format(len(self.sup_dataloader)))

        self.unsup_dataloader = self.get_dataloader_(
            img_path_list=self.img_path_list[: self.unsup_data_num],
            label_path_list=self.label_path_list[: self.unsup_data_num],
            stage="unsupervise"
        )
        log_print("INFO", "Unsupervised dataloader length: {0}".format(len(self.unsup_dataloader)))

        self.val_dataloader = self.get_dataloader_(
            img_path_list=self.img_path_list[-self.val_data_num:],
            label_path_list=self.label_path_list[-self.val_data_num:],
            stage="validation"
        )
        log_print("INFO", "Validation dataloader length: {0}".format(len(self.val_dataloader)))

        # Make log saving direction

        # Make model parameter saving direction


    def get_dataset_(
            self,
            img_path_list,
            label_path_list,
            transform2both,
            transform2img
    ):
        dataset = myDataSet(
            img_paths=img_path_list,
            label_paths=label_path_list,
            cutmix=self.hyper_para_config['cutmix'],
            map=self.hyper_para_config['label_mapping'],
            discard=self.hyper_para_config['label_discard_list'],
            merge=self.hyper_para_config['label_merge_list'],
            transform2both=transform2both,
            transform2img=transform2img
        )
        return dataset

    def get_dataloader_(
            self,
            img_path_list,
            label_path_list,
            stage
    ):
        transform2both, transform2img = get_dataloader_transform(
            stage=stage,
            resize=self.hyper_para_config['resize']
        )
        dataset = self.get_dataset_(
            img_path_list=img_path_list,
            label_path_list=label_path_list,
            transform2both=transform2both,
            transform2img=transform2img
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hyper_para_config['batch_size'],
            prefetch_factor=self.hyper_para_config['prefetch_factor'],
            num_workers=self.hyper_para_config['num_workers'],
        )
        return dataloader

    def train_dual_MBConv_VAE_(self):
        for epoch in range(self.epoch):
            # Training process
            log_print("CRITICAL", "Training---")
            for data1, data2 in tqdm(zip(self.sup_dataloader, self.unsup_dataloader), total=len(self.unsup_dataloader)):
                # Supervised data
                img1 = data1[0].to(self.device)
                label = data1[0].to(self.device)
                # Unsupervised data
                img2 = data2[0].to(self.device)

            # Validation process
            log_print("CRITICAL", "Validation---")


        pass
    def train(self):
        pass
    def evaluate(self):
        pass



if __name__ == '__main__':
    Trainer = SimpleTrainer(config_path="../../train_configuration/test.json")
    Trainer.train_dual_MBConv_VAE_()