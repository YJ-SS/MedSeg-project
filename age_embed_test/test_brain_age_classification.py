from typing import Union, Sequence

from monai.networks.nets import ResNet, ResNetBlock
import torch
import os
from seg_method.train_process.record_func import log_print
import SimpleITK as sitk
from monai.transforms import Compose, Resized, Resize, apply_transform, ToTensord, ToTensor, AddChannel, ScaleIntensity
from age_embed_dataset import myDataset
from torch.utils.data import DataLoader
def get_CT_data(root_path, get_num=200):
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

def train():
    img_path_list_2y, label_list_2y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\2y")
    img_path_list_3y, label_list_3y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\3y")
    img_path_list_4y, label_list_4y = get_CT_data("E:\\DataSet\\brainsegNew\\ct_brain\\3y")
    print(len(img_path_list_2y), len(img_path_list_3y), len(img_path_list_4y))
    transform = get_transform(spatial_size=(256, 256, 256))
    dataset = myDataset(
        img_path_list=img_path_list_2y,
        label_list=label_list_2y,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for img, label in dataloader:
        print(img.shape, label.shape)
        return




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = ResNet(
#     block=ResNetBlock,
#     layers=[2,2,2,2],
#     block_inplanes=[64,128,256,512],
#     spatial_dims=3,
#     n_input_channels=1,
#     num_classes=3
# ).to(device)
# x = torch.randn([1,1,32,32,32]).to(device)
# y = net(x)
# print(y.shape)

if __name__ == '__main__':
    train()
