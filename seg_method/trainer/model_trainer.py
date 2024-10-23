import itertools
import os.path
import sys

import numpy as np
from torch import optim
from tqdm import tqdm

sys.path.append("../model")
sys.path.append("../../data_process")
sys.path.append("../train_process")
from dual_MBConv_VAE import MBConvNet
import json
import torch
from data_process.get_data import get_data_path_list
from data_process.dataset import myDataSet
from data_process.evalution import get_dice
from data_process.data_process_method import get_dataloader_transform, get_recon_region_weights, get_sup_label_weights
from train_process.record_func import log_print, make_model_saving_dir, write2log
from train_process.loss import KL_divergence, self_contrastive_loss
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn
from datetime import datetime
from monai.networks.nets import UNet
warnings.filterwarnings("ignore")


class SimpleTrainer(object):
    def __init__(self, config_path):
        assert  os.path.isfile(config_path), log_print("ERROR", "Config file {0} not found!!!".format(config_path))
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

        # Write training information to log file
        self.training_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + self.training_info_config['log_stamp_str']
        write2log(
            log_file_path=self.training_info_config['log_save_path'],
            log_status="WARNING",
            content="Training start!! Learning rate={0} seg_w={1} rec_w={2} KL_w={3} con_w={4}".format(
                self.hyper_para_config['lr'],
                self.hyper_para_config['seg_weight'],
                self.hyper_para_config['recon_weight'],
                self.hyper_para_config['kl_weight'],
                self.hyper_para_config['contras_weight']
            )
        )

        self.recon_region_weights = get_recon_region_weights(
            data_resolution=self.hyper_para_config['data_resolution'],
            resize=self.hyper_para_config['resize'],
            device=self.device
        )
        log_print("INFO", "Reconstruction region weights initialized: {}".format(
            self.recon_region_weights.shape
        ))
        assert os.path.isfile(self.training_info_config['log_save_path']),\
            log_print("ERROR", "{0} is not exist!!!".format(self.training_info_config['log_save_path']))

        assert os.path.isdir(self.training_info_config['model_para_save_path']),\
            log_print("ERROR", "{0} is not exist!!!".format(self.training_info_config['model_para_save_path']))

        # Make model parameter saving direction
        self.model_para_save_path = make_model_saving_dir(
            model_config=self.model_config,
            hyper_para_config=train_config['hyper_para_config'],
            model_para_save_path=self.training_info_config['model_para_save_path']
        )

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

        self.label_weights = get_sup_label_weights(
            template_label_path=self.label_path_list[0],
            map=self.hyper_para_config['label_mapping'],
            discard=self.hyper_para_config['label_discard_list'],
            merge=self.hyper_para_config['label_merge_list']

        ).to(self.device)

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
            self.net = UNet(
                spatial_dims=3,
                in_channels=self.model_config['in_channel'],
                out_channels=self.model_config['num_class'],
                channels=self.model_config['channel_list'],
                strides=self.model_config['strides'],
                dropout=self.model_config['dropout'],
            ).to(self.device)

        # Build dataset
        # Build dataloader
        self.sup_dataloader = self.get_dataloader_(
            img_path_list=self.img_path_list[:self.sup_data_num],
            label_path_list=self.label_path_list[:self.sup_data_num],
            stage="supervise"
        )
        log_print("INFO", "Supervised dataloader length: {0}".format(len(self.sup_dataloader)))
        self.sup_dataloader = itertools.cycle(self.sup_dataloader)

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
        checkpoint_cnt = 0.
        best_val_dice = 0.

        optimizer = optim.Adamax(
            params=self.net.parameters(),
            lr=self.hyper_para_config['lr'],
            weight_decay=self.hyper_para_config['decay'],
            eps=1e-7
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epoch):
            # Record training date
            train_total_loss = 0.
            train_seg_loss = 0.
            train_recon_loss = 0.
            train_kl_loss = 0.
            train_contras_loss = 0.
            train_dice = 0.
            train_dice_matrix = np.array([0. for i in range(self.model_config['num_class'])])

            val_total_loss = 0.
            val_seg_loss = 0.
            val_recon_loss = 0.
            val_kl_loss = 0.
            val_contras_loss = 0.
            val_dice = 0.
            val_dice_matrix = np.array([0. for i in range(self.model_config['num_class'])])

            checkpoint_cnt += 1
            # Training process
            log_print("CRITICAL", "Training---")
            self.net.train()
            for data1, data2 in tqdm(zip(self.sup_dataloader, self.unsup_dataloader), total=len(self.unsup_dataloader)):
                # Supervised data
                img1 = data1[0].to(self.device)
                label = data1[1].to(self.device)
                # Unsupervised data
                img2 = data2[0].to(self.device)
                with torch.amp.autocast(device_type=str(self.device), dtype=torch.float16):
                    # Self supervised
                    _, recon, mu, logvar, latent = self.net(img2)
                    # Supervised
                    pre_label, _, _, _, _ = self.net(img1)
                    loss_dict = self.get_loss_dict_dual_VAE_(
                        seg_img=img1,
                        seg_gt=label,
                        recon_img=img2,
                        pre_label=pre_label,
                        pre_recon=recon,
                        mu=mu,
                        logvar=logvar,
                        latent_var=latent
                    )
                    total_loss = loss_dict['total_loss']
                    seg_loss = loss_dict['seg_loss']
                    recon_loss = loss_dict['recon_loss']
                    kl_loss = loss_dict['kl_loss']
                    contrastive_loss = loss_dict['contrastive_loss']

                    train_total_loss += total_loss.item()
                    train_seg_loss += seg_loss.item()
                    train_recon_loss += recon_loss.item()
                    train_kl_loss += kl_loss.item()
                    train_contras_loss += contrastive_loss.item()

                    dice_temp, dice_matrix_temp = get_dice(
                        y_pred=pre_label,
                        y_true=label,
                        num_clus=self.model_config['num_class']
                    )
                    # print("Training avg dice: ", dice_temp)
                    train_dice += dice_temp
                    train_dice_matrix += dice_matrix_temp



                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Validation process
            log_print("CRITICAL", "Validation---")
            self.net.eval()
            with torch.no_grad():
                for img, label in tqdm(self.val_dataloader):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    pre_label, recon, mu, logvar, latent = self.net(img)
                    loss_dict = self.get_loss_dict_dual_VAE_(
                        seg_gt=label,
                        recon_img=img,
                        pre_label=pre_label,
                        pre_recon=recon,
                        mu=mu,
                        logvar=logvar,
                        latent_var=latent
                    )
                    total_loss = loss_dict['total_loss']
                    seg_loss = loss_dict['seg_loss']
                    recon_loss = loss_dict['recon_loss']
                    kl_loss = loss_dict['kl_loss']
                    contrastive_loss = loss_dict['contrastive_loss']
                    val_total_loss += total_loss.item()
                    val_seg_loss += seg_loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()
                    val_contras_loss += contrastive_loss.item()

                    dice_temp, dice_matrix_temp = get_dice(
                        y_pred=pre_label,
                        y_true=label,
                        num_clus=self.model_config['num_class']
                    )
                    # print("Validation avg dice: ", dice_temp)
                    val_dice += dice_temp
                    val_dice_matrix += dice_matrix_temp


            epoch_log_content = "EPOCH={0} train_total_loss={11:.2f} train_seg_loss={1:.2f} train_recon_loss={2:.2f} train_KL_loss={3:.2f} "\
                "train_contras_loss={4:.2f} train_dice={5:.2f} "\
                "val_total_loss={12:.2f} val_seg_loss={6:.2f} val_recon_loss={7:.2f} val_KL_loss={8:.2f} val_contras_loss={9:.2f} "\
                "val_dice={10:.2f}".format(
                    epoch,
                    train_seg_loss / len(self.unsup_dataloader),
                    train_recon_loss / len(self.unsup_dataloader),
                    train_kl_loss / len(self.unsup_dataloader),
                    train_contras_loss / len(self.unsup_dataloader),
                    train_dice / len(self.unsup_dataloader),
                    val_seg_loss / len(self.val_dataloader),
                    val_recon_loss / len(self.val_dataloader),
                    val_kl_loss / len(self.val_dataloader),
                    val_contras_loss / len(self.val_dataloader),
                    val_dice / len(self.val_dataloader),
                    train_total_loss / len(self.unsup_dataloader),
                    val_total_loss / len(self.val_dataloader)
                )
            log_print("CRITICAL", epoch_log_content)
            write2log(
                log_file_path=self.training_info_config['log_save_path'],
                log_status='INFO',
                content=str(self.training_stamp) + " " + epoch_log_content
            )
            if val_dice / len(self.val_dataloader) > best_val_dice:
                # Save model with better dice
                best_val_dice = val_dice / len(self.val_dataloader)
                model_save_path = os.path.join(self.model_para_save_path, "{0} best.pth".format(self.model_config['model_name']))
                torch.save({
                    'epoch': epoch,
                    "model_name": self.model_config['model_name'],
                    "in_channel": self.model_config['in_channel'],
                    "num_class": self.model_config['num_class'],
                    "residual": self.model_config['residual'],
                    "MBConv": self.model_config['MBConv'],
                    "channel_list": self.model_config['channel_list'],
                    'model_state_dict': self.net.state_dict()
                }, model_save_path)
                log_print("INFO", "Best model saved!!!")


            if checkpoint_cnt % self.hyper_para_config['save_checkpoint_per_epoch'] == 0:
                # Save checkout point
                model_save_path = os.path.join(
                    self.model_para_save_path,
                    "{0} checkpoint{1}.pth".format(self.model_config['model_name'], epoch)
                )
                torch.save({
                    'epoch': epoch,
                    "model_name": self.model_config['model_name'],
                    "in_channel": self.model_config['in_channel'],
                    "num_class": self.model_config['num_class'],
                    "residual": self.model_config['residual'],
                    "MBConv": self.model_config['MBConv'],
                    "channel_list": self.model_config['channel_list'],
                    'model_state_dict': self.net.state_dict()
                }, model_save_path)
                log_print("INFO", "Checkpoint saved!!!")


    def get_loss_dict_dual_VAE_(self, seg_gt, recon_img, pre_label, pre_recon, mu, logvar, latent_var):
        '''
        get loss dict when training dual VAE
        :param seg_img:
        :param seg_gt:
        :param recon_img:
        :param pre_label:
        :param mu:
        :param logvar:
        :param latent_var:
        :return:
        '''
        seg_loss_fn = nn.CrossEntropyLoss(weight=self.label_weights)
        recon_loss_fn = nn.MSELoss(reduction='none')

        seg_loss = seg_loss_fn(pre_label, seg_gt.squeeze(dim=1).long())
        recon_loss = recon_loss_fn(recon_img, pre_recon)
        recon_loss = (recon_loss * self.recon_region_weights).mean()
        kl_loss = KL_divergence(mu=mu, logvar=logvar)
        contrastive_loss = self_contrastive_loss(latent_var, avg_pool=True)
        total_loss = seg_loss * self.hyper_para_config['seg_weight']\
                    + recon_loss * self.hyper_para_config['recon_weight']\
                    + kl_loss * self.hyper_para_config['kl_weight']\
                    + contrastive_loss * self.hyper_para_config['contras_weight']

        loss_dict = {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "contrastive_loss": contrastive_loss
        }
        return loss_dict

    def train_normal_Net_(self):
        '''
        Can be used to train any segmentation model with only one output
        :return: None
        '''
        checkpoint_cnt = 0.
        best_val_dice = 0.
        optimizer = optim.Adamax(
            params=self.net.parameters(),
            lr=self.hyper_para_config['lr'],
            weight_decay=self.hyper_para_config['decay'],
            eps=1e-7
        )
        seg_loss_func = nn.CrossEntropyLoss(weight=self.label_weights)
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.epoch):
            train_seg_loss = 0.
            train_dice = 0.
            train_dice_matrix = np.array([0. for i in range(self.model_config['num_class'])])

            val_seg_loss = 0.
            val_dice = 0.
            val_dice_matrix = np.array([0. for i in range(self.model_config['num_class'])])

            log_print("CRITICAL", "Training---")
            self.net.train()
            for img, label in tqdm(self.unsup_dataloader):
                img = img.to(self.device)
                label = label.to(self.device)
                with torch.amp.autocast(device_type=str(self.device), dtype=torch.float16):
                    pre_label = self.net(img)
                    seg_loss = seg_loss_func(pre_label, label.squeeze(dim=1).long())
                    train_seg_loss += seg_loss.item()
                    dice_temp, dice_matrix_temp = get_dice(
                        y_pred=pre_label,
                        y_true=label,
                        num_clus=self.model_config['num_class']
                    )
                    train_dice += dice_temp
                    train_dice_matrix += dice_matrix_temp
                optimizer.zero_grad()
                scaler.scale(seg_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Validation process
            log_print("CRITICAL", "Validation---")
            self.net.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=str(self.device), dtype=torch.float16):
                    for img, label in tqdm(self.val_dataloader):
                        img = img.to(self.device)
                        label = label.to(self.device)
                        pre_label = self.net(img)
                        seg_loss = seg_loss_func(pre_label, label.squeeze(dim=1).long())
                        val_seg_loss += seg_loss.item()
                        dice_temp, dice_matrix_temp = get_dice(
                            y_pred=pre_label,
                            y_true=label,
                            num_clus=self.model_config['num_class']
                        )
                        val_dice += dice_temp
                        val_dice_matrix += dice_matrix_temp

            epoch_log_content = "EPOCH={0} train_seg_loss={1:.2f} train_dice={2:.2f} "\
                                "val_seg_loss={3:.2f} val_dice={4:.2f}".format(
                epoch,
                train_seg_loss / len(self.unsup_dataloader),
                train_dice / len(self.unsup_dataloader),
                val_seg_loss / len(self.val_dataloader),
                val_dice / len(self.val_dataloader)
            )

            log_print("CRITICAL", epoch_log_content)
            write2log(
                log_file_path=self.training_info_config['log_save_path'],
                log_status='INFO',
                content=str(self.training_stamp) + " " + epoch_log_content
            )

            if val_dice / len(self.val_dataloader) > best_val_dice:
                # Save model with better dice
                best_val_dice = val_dice / len(self.val_dataloader)
                model_save_path = os.path.join(self.model_para_save_path, "{0} best.pth".format(self.model_config['model_name']))
                torch.save({
                    'epoch': epoch,
                    "model_name": self.model_config['model_name'],
                    "in_channel": self.model_config['in_channel'],
                    "num_class": self.model_config['num_class'],
                    'model_state_dict': self.net.state_dict()
                }, model_save_path)
                log_print("INFO", "Best model saved!!!")

            if checkpoint_cnt % self.hyper_para_config['save_checkpoint_per_epoch'] == 0:
                # Save checkout point
                model_save_path = os.path.join(
                    self.model_para_save_path,
                    "{0} checkpoint{1}.pth".format(self.model_config['model_name'], epoch)
                )
                torch.save({
                    'epoch': epoch,
                    "model_name": self.model_config['model_name'],
                    "in_channel": self.model_config['in_channel'],
                    "num_class": self.model_config['num_class'],
                    'model_state_dict': self.net.state_dict()
                }, model_save_path)
                log_print("INFO", "Checkpoint saved!!!")




    def train(self):
        if self.model_name == 'dual_MBConv_VAE':
            self.train_dual_MBConv_VAE_()
        elif self.model_name == 'monai_3D_unet':
            self.train_normal_Net_()

    def evaluate(self):
        pass



# if __name__ == '__main__':
#     Trainer = SimpleTrainer(config_path="../../train_configuration/test_unet_monai.json")
#     Trainer.train()