from datetime import datetime
from typing import Union, Sequence

from monai.networks.nets import ResNet, ResNetBlock
import torch
import os

from torch import dtype

from seg_method.train_process.record_func import log_print
import SimpleITK as sitk
from monai.transforms import Compose, Resized, Resize, apply_transform, ToTensord, ToTensor, AddChannel, ScaleIntensity
from age_embed_dataset import myDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm

def get_CT_data(root_path, get_num=160):
    '''
    get image path list and it's label list for age classification
    :param root_path:
    :param get_num:
    :return:
    '''
    file_name_list = os.listdir(root_path)
    assert len(file_name_list) >= get_num,\
    log_print("ERROR", "get_num:{0} is larger than the total number of images in {1}!!!".format(
        get_num,
        root_path
    ))
    img_path_list = []
    label_list = None
    for i in range(get_num):
        file_path = os.path.join(root_path, file_name_list[i] + "/brain.nii.gz")
        img_path_list.append(file_path)
    if '2y' in root_path:
        label_list = [0 for i in range(get_num)]
    elif '3y' in root_path:
        label_list = [1 for i in range(get_num)]
    elif '4y' in root_path:
        label_list = [2 for i in range(get_num)]
    return img_path_list, label_list

def get_transform(spatial_size:Union[Sequence[int]]):
    transform_list = [
        ToTensor(),
        AddChannel(),
        Resize(spatial_size=spatial_size, mode='area'),
        ScaleIntensity(minv=0., maxv=1.)
    ]
    return Compose(transform_list)

def get_acc_count(pre_label, label):
    '''

    :param pre_label: [B,C]
    :param label: [B]
    :return:
    '''
    # [B,C] -> [B]
    pre_label = torch.argmax(pre_label, dim=-1).squeeze()
    return (pre_label == label).sum()

def split_train_and_val(
        img_path_list_2y,
        label_list_2y,
        img_path_list_3y,
        label_list_3y,
        img_path_list_4y,
        label_list_4y,
        train_data_num_4each):
    '''
      Concatenate path of 2y, 3y, 4y and split train and val dataset
    :param img_path_list_2y: 
    :param label_list_2y: 
    :param img_path_list_3y: 
    :param label_list_3y: 
    :param img_path_list_4y: 
    :param label_list_4y: 
    :param train_data_num_4each: 
    :return: 
    '''
    train_img_path_list_2y = img_path_list_2y[:train_data_num_4each]
    train_label_list_2y = label_list_2y[:train_data_num_4each]
    train_img_path_list_3y = img_path_list_3y[:train_data_num_4each]
    train_label_list_3y = label_list_3y[:train_data_num_4each]
    train_img_path_list_4y = img_path_list_4y[:train_data_num_4each]
    train_label_list_4y = label_list_4y[:train_data_num_4each]

    val_img_path_list_2y = img_path_list_2y[train_data_num_4each:]
    val_label_list_2y = label_list_2y[train_data_num_4each:]
    val_img_path_list_3y = img_path_list_3y[train_data_num_4each:]
    val_label_list_3y = label_list_3y[train_data_num_4each:]
    val_img_path_list_4y = img_path_list_4y[train_data_num_4each:]
    val_label_list_4y = label_list_4y[train_data_num_4each:]

    train_img_path_list = train_img_path_list_2y + train_img_path_list_3y + train_img_path_list_4y
    train_label_list = train_label_list_2y + train_label_list_3y + train_label_list_4y

    val_img_path_list = val_img_path_list_2y + val_img_path_list_3y + val_img_path_list_4y
    val_label_list = val_label_list_2y + val_label_list_3y + val_label_list_4y
    return train_img_path_list, train_label_list, val_img_path_list, val_label_list


def train():
    # Initialize log file
    logging.basicConfig(
        filename="test_age_classification.log",
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M'  # 日志格式
    )
    training_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    img_path_list_2y, label_list_2y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\2y")
    img_path_list_3y, label_list_3y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\3y")
    img_path_list_4y, label_list_4y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\4y")
    print(len(img_path_list_2y), len(img_path_list_3y), len(img_path_list_4y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet(
        block=ResNetBlock,
        layers=[1,1,2,2],
        block_inplanes=[16, 32,64,128],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3
    ).to(device)
    train_img_path_list, train_label_list,val_img_path_list, val_label_list = \
        split_train_and_val(
        img_path_list_2y=img_path_list_2y,
        label_list_2y=label_list_2y,
        img_path_list_3y=img_path_list_3y,
        label_list_3y=label_list_3y,
        img_path_list_4y=img_path_list_4y,
        label_list_4y=label_list_4y,
        train_data_num_4each=120
    )
    # print(np.unique(train_label_list), np.unique(val_label_list))
    transform = get_transform(spatial_size=(256, 256, 256))
    train_dataset = myDataset(
        img_path_list=train_img_path_list,
        label_list=train_label_list,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, prefetch_factor=16)

    val_dataset = myDataset(
        img_path_list=val_img_path_list,
        label_list=val_label_list,
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=8, prefetch_factor=16)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-7)
    for epoch in range(500):
        train_loss = 0.
        train_acc_cnt = 0
        train_total_cnt = 0
        val_loss = 0.
        val_acc_cnt = 0
        val_total_cnt = 0
        net.train()
        log_print("INFO", "Training---")
        for img, label in tqdm(train_loader):
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                img, label = img.to(device), label.to(device)
                pre_label = net(img)
                loss = loss_fn(pre_label, label)
                train_acc_cnt += get_acc_count(pre_label, label)
                train_total_cnt += len(label)
                train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        log_print("INFO", "Validation---")
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                with torch.autocast(device_type=str(device), dtype=torch.float16):
                    img, label = img.to(device), label.to(device)
                    pre_label = net(img)
                    loss = loss_fn(pre_label, label)
                    val_loss += loss.item()
                    val_acc_cnt += get_acc_count(pre_label, label)
                    val_total_cnt += len(label)

        epoch_log_content = "EPOCH={0} train_loss={1} train_acc = {2} "\
                            "val_loss = {3} val_acc = {4}".format(
            epoch,
            train_loss / len(train_loader),
            train_acc_cnt / train_total_cnt,
            val_loss / len(val_loader),
            val_acc_cnt / val_total_cnt
        )
        log_print("CRITICAL", epoch_log_content)
        logging.info(training_stamp + " " + epoch_log_content)





if __name__ == '__main__':
    train()
