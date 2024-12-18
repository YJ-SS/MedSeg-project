import itertools
import os.path
import sys
import logging
import SimpleITK as sitk
import numpy as np
from torch import optim
from tqdm import tqdm
from data_process_method import resample, cut_mix

sys.path.append("../model")
sys.path.append("../../data_process")
sys.path.append("../train_process")
from dual_MBConv_VAE import MBConvNet
from dual_transfromer_VAE import transNet
import json
import torch
from data_process.get_data import get_data_path_list
from data_process.dataset import myDataSet
from data_process.evalution import get_dice
from data_process.data_process_method import get_dataloader_transform, get_recon_region_weights, \
    get_sup_label_weights, get_data_4_val
from train_process.record_func import log_print, make_model_saving_dir, write2log
from train_process.loss import KL_divergence, self_contrastive_loss
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn
from datetime import datetime
from monai.networks.nets import UNet, SwinUNETR
from train_process.early_stop import EarlyStopping
from dual_multi_scale_VAE import MultiScaleNet
warnings.filterwarnings("ignore")


class SimpleTrainer(object):
    def __init__(self, config_path):
        assert  os.path.isfile(config_path), log_print("ERROR", "Config file {0} not found!!!".format(config_path))
        # Get training configration
        train_config = json.load(open(config_path))
        self.model_config = train_config['model_config']
        self.hyper_para_config = train_config['hyper_para_config']
        self.training_info_config = train_config['training_info_config']
        self.validation_info_config = train_config['validation_info_config']
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


        # Initialize log file
        logging.basicConfig(
            filename=self.training_info_config['log_save_path'],
            level=logging.DEBUG,  # 设置日志级别
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M' # 日志格式
        )

        # Write training information to log file
        self.training_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + self.training_info_config['log_stamp_str']
        logging.warning("Training start!! Learning rate={0} seg_w={1} rec_w={2} KL_w={3} con_w={4}".format(
                self.hyper_para_config['lr'],
                self.hyper_para_config['seg_weight'],
                self.hyper_para_config['recon_weight'],
                self.hyper_para_config['kl_weight'],
                self.hyper_para_config['contras_weight']
            ))
        # write2log(
        #     log_file_path=self.training_info_config['log_save_path'],
        #     log_status="WARNING",
        #     content="Training start!! Learning rate={0} seg_w={1} rec_w={2} KL_w={3} con_w={4}".format(
        #         self.hyper_para_config['lr'],
        #         self.hyper_para_config['seg_weight'],
        #         self.hyper_para_config['recon_weight'],
        #         self.hyper_para_config['kl_weight'],
        #         self.hyper_para_config['contras_weight']
        #     )
        # )

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
            hyper_para_config=self.hyper_para_config,
            training_info_config=self.training_info_config,
            model_para_save_path=self.training_info_config['model_para_save_path']
        )

        # Set early stopping
        self.early_stop_training = EarlyStopping(
            patience=self.hyper_para_config['patience'],
            model_config=self.model_config,
            model_save_path=self.model_para_save_path,
            mode='min'
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

        self.sup_img_list, self.sup_label_list = self.img_path_list[:self.sup_data_num], \
            self.label_path_list[:self.sup_data_num]

        self.unsup_img_list, self.unsup_label_list = self.img_path_list[: self.unsup_data_num],\
            self.label_path_list[: self.unsup_data_num]

        self.val_img_list, self.val_label_list = self.img_path_list[-self.val_data_num:],\
            self.label_path_list[-self.val_data_num:]


        self.label_weights = get_sup_label_weights(
            template_label_path=self.label_path_list[0],
            map=self.hyper_para_config['label_mapping'],
            discard=self.hyper_para_config['label_discard_list'],
            merge=self.hyper_para_config['label_merge_list']

        ).to(self.device)

        log_print("INFO", "Use device: {0}".format(self.device))

        # Build model1
        if self.model_name == "dual_MBConv_VAE":
            self.net = MBConvNet(
                in_channel=self.model_config['in_channel'],
                num_class=self.model_config['num_class'],
                residual=self.model_config['residual'],
                channel_list=self.model_config['channel_list'],
                MBConv=self.model_config['MBConv'],
                device=self.device
            )
        elif self.model_name == 'dual_transformer_VAE':
            self.net = transNet(
                in_channels=self.model_config['in_channel'],
                num_class=self.model_config['num_class'],
                embed_dim=self.model_config['embed_dim'],
                patch_size=self.model_config['patch_size'],
                window_size=self.model_config['window_size'],
            ).to(self.device)
        elif self.model_name == "monai_3D_unet":
            self.net = UNet(
                spatial_dims=3,
                in_channels=self.model_config['in_channel'],
                out_channels=self.model_config['num_class'],
                channels=self.model_config['channel_list'],
                strides=self.model_config['strides'],
                dropout=self.model_config['dropout'],
            ).to(self.device)
        elif self.model_name == "SwinUNETR":
            self.net = SwinUNETR(
                img_size=self.hyper_para_config['data_resolution']
                    if self.hyper_para_config['resize'] is None
                    else self.hyper_para_config['data_resolution'],
                in_channels=self.model_config['in_channel'],
                out_channels=self.model_config['num_class'],
                drop_rate=self.model_config['dropout'],
                feature_size=self.model_config['feature_size'],
                depths=self.model_config['depths'],
                num_heads=self.model_config['num_heads']
            ).to(self.device)
        elif self.model_name == 'dual_multi_scale_VAE':
            self.net = MultiScaleNet(
                in_channel=self.model_config['in_channel'],
                num_class=self.model_config['num_class'],
                channel_list=self.model_config['channel_list'],
                device=torch.device(self.device),
            ).to(self.device)
        else:
            log_print("ERROR", "Model {0} has not been supported!!!".format(self.model_name))
            return


        # Build dataset
        # Build dataloader
        self.sup_dataloader = self.get_dataloader_(
            img_path_list=self.sup_img_list,
            label_path_list=self.sup_label_list,
            stage="supervise"
        )
        log_print("INFO", "Supervised dataloader length: {0}".format(len(self.sup_dataloader)))
        self.sup_dataloader = itertools.cycle(self.sup_dataloader)

        self.unsup_dataloader = self.get_dataloader_(
            img_path_list=self.unsup_img_list,
            label_path_list=self.unsup_label_list,
            stage="unsupervise"
        )
        log_print("INFO", "Unsupervised dataloader length: {0}".format(len(self.unsup_dataloader)))

        self.val_dataloader = self.get_dataloader_(
            img_path_list=self.val_img_list,
            label_path_list=self.val_label_list,
            stage="validation"
        )
        log_print("INFO", "Validation dataloader length: {0}".format(len(self.val_dataloader)))



    def get_dataset_(
            self,
            img_path_list,
            label_path_list,
            stage,
            transform2both,
            transform2img
    ):
        dataset = myDataSet(
            img_paths=img_path_list,
            label_paths=label_path_list,
            cutmix=self.hyper_para_config['cutmix'] if stage == 'supervise' else False,
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
            stage=stage,
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

    def train_dual_VAE_(self):
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
            logging.info(str(self.training_stamp) + " " + epoch_log_content)
            # Save checkpoint and best model
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

            early_stop_flag = self.early_stop_training.check_and_early_stop(
                epoch=epoch,
                net=self.net,
                value=train_total_loss / len(self.unsup_dataloader),
            )
            if early_stop_flag:
                log_print("INFO", "Early stop training!!!")
                return



    def get_loss_dict_dual_VAE_(
            self,
            seg_gt,
            recon_img,
            pre_label,
            pre_recon,
            mu,
            logvar,
            latent_var
    ):
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
        # recon_loss_fn = nn.MSELoss(reduction='none')
        recon_loss_fn = nn.MSELoss()

        seg_loss = seg_loss_fn(pre_label, seg_gt.squeeze(dim=1).long())
        recon_loss = recon_loss_fn(recon_img, pre_recon)
        # recon_loss = (recon_loss * self.recon_region_weights).mean()
        kl_loss = KL_divergence(mu=mu, logvar=logvar)
        contrastive_loss = torch.tensor(0.0)
        if self.hyper_para_config['calcu_contras_loss']:
            '''
            When use dual_transformer_VAE, the latent shape is [1, 128, 7, 6, 5] on OASIS4 dataset,
            which will cause error when calculating contrastive loss.
            This BUG can be fixed if the input image is reshaped or reduce the number of layers in 
            dual_transformer_VAE
            '''
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
        Can be used to train any segmentation model1 with only one output
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
            checkpoint_cnt += 1

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
            logging.info(str(self.training_stamp) + " " + epoch_log_content)
            # write2log(
            #     log_file_path=self.training_info_config['log_save_path'],
            #     log_status='INFO',
            #     content=str(self.training_stamp) + " " + epoch_log_content
            # )
            if val_dice / len(self.val_dataloader) > best_val_dice:
                # Save model with better dice on validation set
                best_val_dice = val_dice / len(self.val_dataloader)
                model_save_path = os.path.join(self.model_para_save_path, "{0} best.pth".format(self.model_config['model_name']))
                torch.save({
                    'epoch': epoch,
                    "model_name": self.model_config['model_name'],
                    "in_channel": self.model_config['in_channel'],
                    "num_class": self.model_config['num_class'],
                    'model_state_dict': self.net.state_dict()
                }, model_save_path)
                log_print("INFO", "Best model1 saved!!!")

            if checkpoint_cnt % self.hyper_para_config['save_checkpoint_per_epoch'] == 0:
                # Save checkpoint
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
        if self.model_name == 'dual_MBConv_VAE' or self.model_name == 'dual_transformer_VAE' or self.model_name == 'dual_multi_scale_VAE':
            self.train_dual_VAE_()
        elif self.model_name == 'monai_3D_unet' or self.model_name == 'SwinUNETR':
            self.train_normal_Net_()
        else:
            log_print("ERROR", "{0} has not been supported!!!".format(self.model_name))

    def load_model_info_(self):
        '''
        Load model1 parameter based on json file
        :return: Model information dict, include model1 parameter
        '''
        model_save_path = ""
        if self.validation_info_config['model_para_file_name'] is None:
            # 没有显式指定用什么模型，默认用best模型
            model_save_path = os.path.join(
                self.model_para_save_path,
                "{0} best.pth".format(self.model_config['model_name']))
        else:
            model_save_path = os.path.join(
                self.model_para_save_path,
                self.validation_info_config['model_para_file_name']
            )
        assert os.path.exists(model_save_path),\
        log_print("ERROR", "Validation model {} is not exist!!!".format(model_save_path))
        log_print("INFO", "Validation model path={0}".format(model_save_path))
        model_info_dict = torch.load(model_save_path)
        return model_info_dict


    def evaluate_one_(self, model, val_img_path, val_label_path):
        '''
        Get a segmentation of the input image
        :param val_img_path:
        :return: A SimpleITK Image predicted by model1, dice score and dice matrix
        '''
        # Get original image information to restore image
        # Spacing, Origin, Direction, Size and Processed Image
        img_info_dict = get_data_4_val(
            val_img_path,
            status='image',
            resize=None if self.hyper_para_config['resize'] is None else self.hyper_para_config['resize']
        )
        label_info_dict = get_data_4_val(
            val_label_path,
            status='label',
            resize=None if self.hyper_para_config['resize'] is None else self.hyper_para_config['resize']
        )
        img = img_info_dict['img'].to(self.device)
        gt = label_info_dict['img'].to(self.device)
        log_print("INFO", "Validation: Image shape={0}, GT shape={1}".format(img.shape, gt.shape))
        model.eval()
        with torch.no_grad():
            if self.model_name == 'dual_MBConv_VAE' or self.model_name == 'dual_transformer_VAE' or self.model_name == 'dual_multi_scale_VAE':
                pre_label, _, _, _, _ = model(img, is_train=False)
            elif self.model_name == 'monai_3D_unet' or self.model_name == 'SwinUNETR':
                pre_label = model(img)
        dice, dice_matrix = get_dice(y_pred=pre_label, y_true=gt, num_clus=self.model_config['num_class'])
        log_print("INFO", "Dice score={0:.4f}, Dice Matrix={1}".format(dice, dice_matrix))
        pre_label = torch.argmax(pre_label.to('cpu').squeeze(dim=0), dim=0).numpy().astype(np.int16)
        pre_mask = sitk.GetImageFromArray(pre_label)
        pre_mask.SetSpacing(img_info_dict['ori_spacing'])
        pre_mask.SetDirection(img_info_dict['ori_direction'])
        pre_mask.SetOrigin(img_info_dict['ori_origin'])
        if self.hyper_para_config['resize']:
            # Restore mask resolution
            pre_mask = resample(
                pre_mask,
                new_size=img_info_dict['img_size']
            )
        return pre_mask, dice, dice_matrix



    def evaluate(self, img_path_list=None, label_path_list=None):
        '''
        Evaluate model1, product predict segmentation.
        :return:
        '''
        if img_path_list is None and label_path_list is None:
            img_path_list = self.val_img_list
            label_path_list = self.val_label_list
        state_dict = self.load_model_info_()['model_state_dict']
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        avg_dice = 0.
        avg_dice_matrix = np.array([0. for i in range(self.model_config['num_class'])])
        validation_time = datetime.now().strftime("%Y-%m-%d")
        validation_time_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predict_mask_name = self.model_name + " " \
                            + self.training_info_config['dataset_name'] + " pre_mask " + str(validation_time)
        for img_path, label_path in tqdm(zip(img_path_list, label_path_list), total=len(img_path_list)):
            pre_mask, dice, dice_matrix = self.evaluate_one_(self.net, img_path, label_path)
            avg_dice += dice
            avg_dice_matrix += dice_matrix
            dir_path = os.path.dirname(label_path)
            pre_mask_path = os.path.join(dir_path, predict_mask_name + ".nii.gz")
            sitk.WriteImage(pre_mask, pre_mask_path)
            val_log_content = "img_path={0} dice={1:.4f} dice_matrix={2}".format(
                img_path,
                dice,
                dice_matrix
            )
            logging.info("Validation " + str(validation_time_log) + " " + val_log_content)
        #     write2log(
        #         log_file_path=self.training_info_config['log_save_path'],
        #         log_status='INFO',
        #         content="Validation " + str(validation_time_log) + " " + val_log_content
        #     )
        # # Write average dice and average dice matrix
        avg_dice_info = "INFO", "Average dice={0}, average dice matrix={1}".format(
            avg_dice / len(img_path_list),
            avg_dice_matrix / len(img_path_list)
        )
        logging.info("Validation " + str(validation_time_log) + " " + str(avg_dice_info))
        # write2log(
        #     log_file_path=self.training_info_config['log_save_path'],
        #     log_status='INFO',
        #     content="Validation " + str(validation_time_log) + " " + avg_dice_info
        # )
        log_print("INFO", str(avg_dice_info))





# if __name__ == '__main__':
#     Trainer = SimpleTrainer(config_path="../../train_configuration/test_unet_monai.json")
#     Trainer.train()