import sys
from typing import Dict, Union, Any, Sequence, Mapping

from torch import Tensor

sys.path.append("../seg_method/train_process")
from numpy import random, ndarray
import SimpleITK as sitk
import torch
from monai.transforms import (Compose, Resized, Resize, apply_transform, ToTensord, ToTensor, AddChanneld, AddChannel,
                              NormalizeIntensityd, CenterSpatialCropd,
                              RandFlipd, GibbsNoise, RandRotate90d, RandRotated, RandZoomd, ThresholdIntensityd,
                              ScaleIntensityRanged, ScaleIntensity,
                              RandShiftIntensityd, RandAffined, Rand3DElasticd, RandGridDistortiond,
                              RandSpatialCropSamplesd)
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
from train_process.record_func import log_print



def get_dataloader_transform(stage, resize=None):
    assert stage in ['supervise', 'unsupervise', 'validation'],\
    log_print("ERROE", "Stage must in ['supervise', 'unsupervise', 'validation'], "
                       "current stage: {0}".format(stage))
    transform2both_list = None
    transform2img_list = None
    if stage == 'supervise':
        transform2both_list = [
            ToTensord(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            RandRotated(keys=['image', 'label'], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5,
                        mode=['bilinear', 'nearest']),
            RandAffined(keys=['image', 'label'], prob=0.5, scale_range=[0.1, 0.3], mode=['bilinear', 'nearest'],
                        padding_mode='zeros')
        ]
        transform2img_list = [
        ScaleIntensity(minv=0., maxv=1.)
        ]

        if resize is not None:
            transform2both_list.insert(
                2,
                Resized(keys=['image', 'label'], spatial_size=resize, mode=['area', 'nearest'])
            )
    elif stage == 'unsupervise':
        transform2both_list = [
            ToTensord(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
        ]
        transform2img_list = [
            ScaleIntensity(minv=0., maxv=1.)
        ]
        if resize is not None:
            transform2both_list.insert(2, Resize(spatial_size=resize, mode='area'))
    elif stage == 'validation':
        transform2img_list = [
            ScaleIntensity(minv=0., maxv=1.)
        ]
        transform2both_list = [
            ToTensord(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),

        ]
        if resize is not None:
            transform2both_list.insert(
                2,
                Resized(keys=['image', 'label'], spatial_size=resize, mode=['area', 'nearest'])
            )
    transform2img = Compose(transform2img_list)
    transform2both = Compose(transform2both_list)
    return transform2both, transform2img


def mapping(label):
    '''
    Mapping uncontinous label to continous label
    :param label:
    :return: label after mapping
    '''
    label_mapping = {
        0.0: 0,
        1.0: 1,
        2.0: 2,
        3.0: 3,
        4.0: 4,
        5.0: 5,
        6.0: 6,
        7.0: 7,
        8.0: 8,
        9.0: 9,
        10.0: 10,
        11.0: 11,
        12.0: 12,
        13.0: 13,
        14.0: 14,
        15.0: 15,
        16.0: 16,
        17.0: 17,
        18.0: 18,
        19.0: 19,
        20.0: 20,
        21.0: 21,
        24.0: 22,
        25.0: 23,
        28.0: 24,
        29.0: 25,
        30.0: 26
    }

    # 使用映射替换标签数组中的值
    for old_label, new_label in label_mapping.items():
        label[label == old_label] = new_label

    # 现在，label1_array 包含了连续的整数值
    label1_array = label.astype(int)
    return label1_array


def label_mapping(label_img: np.array, discard: list, merge: list):
    if discard is not None:
        for label in discard:
            label_img[label_img == label] = 0

    if merge is not None:
        main_label = merge[0]
        for label in merge:
            label_img[label_img == label] = main_label

    ori_labels = np.unique(label_img)
    # print(ori_labels)
    index = 0
    for label in ori_labels:
        label_img[label_img == label] = index
        index += 1
    new_labels = np.unique(label_img)
    # print('label list after discard and merge: ', new_labels)
    return label_img

def get_recon_region_weights(data_resolution, resize, device):
    recon_region_weights = None
    if resize is not None:
        recon_region_weights = torch.ones(resize)
    else:
        recon_region_weights = torch.ones(data_resolution)
    assert recon_region_weights is not None, \
        log_print("ERROR", "Initialize reconstruction region weights failed!!!")
    recon_region_weights = recon_region_weights.unsqueeze(dim=0).unsqueeze(dim=0).to(device)

    return recon_region_weights

def get_sup_label_weights(template_label_path, map, discard, merge, f_order_para=10., s_order_para=3.):
    label = sitk.ReadImage(template_label_path)
    label = sitk.GetArrayFromImage(label)
    if map:
        label = label_mapping(label, discard=discard, merge=merge)
    element, cnt = np.unique(label, return_counts=True)
    cnt = np.log(cnt)
    weights = 1 / cnt * f_order_para
    weights = weights ** s_order_para
    # softmax
    weights = np.exp(weights - np.max(weights))
    weights = weights / np.sum(weights)
    log_print("INFO", "Label index: {0}".format(str(element)))
    log_print("INFO", "Label weights: {0}".format(str(weights)))
    weights = torch.Tensor(weights)
    return weights

def get_data_4_val(data_path:str, status:str, resize=None)-> dict[str, Any]:
    '''
    Get image or label for validation
    :param data_path:
    :param resize:list[float]
    :return: data information dict
    '''
    assert status in ['image', 'label'],\
        log_print("ERROR", "Status must be 'image' or 'label', "
                           "current status={0}".format(status))
    img = sitk.ReadImage(data_path)
    ori_spacing = img.GetSpacing()
    ori_origin = img.GetOrigin()
    ori_direction = img.GetDirection()
    ori_size = img.GetSize()
    img_array = sitk.GetArrayFromImage(img)
    img_transform_list = [
        ToTensor(),
        ScaleIntensity(minv=0., maxv=1.),
        AddChannel(),
        AddChannel()
    ]
    label_transform_list = [
        ToTensor(),
        AddChannel()
    ]
    if resize is not None:
        # No need to resize
        img_transform_list.insert(
            3,
            Resize(spatial_size=resize, mode='area')
        )
        label_transform_list.insert(
            2,
            Resize(spatial_size=resize, mode='nearest')
        )
    transform = None
    if status == 'image':
        transform = Compose(img_transform_list)
    elif status == 'label':
        transform = Compose(label_transform_list)
    img_array = transform(img_array)
    result_dict = {
        'ori_spacing': ori_spacing,
        'ori_origin': ori_origin,
        'ori_direction': ori_direction,
        'img_size': ori_size,
        'img': img_array
    }
    return result_dict


def resample(input_image: sitk.Image, new_size: list[int], interpolator=sitk.sitkNearestNeighbor):
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()

    # Compute the new spacing based on the new size
    new_spacing = [
        (original_size[0] * original_spacing[0]) / new_size[0],
        (original_size[1] * original_spacing[1]) / new_size[1],
        (original_size[2] * original_spacing[2]) / new_size[2],
    ]

    resample_item = sitk.ResampleImageFilter()
    resample_item.SetOutputSpacing(new_spacing)
    resample_item.SetSize(new_size)
    resample_item.SetInterpolator(interpolator)
    resample_item.SetOutputDirection(input_image.GetDirection())
    resample_item.SetOutputOrigin(input_image.GetOrigin())
    resampled_image = resample_item.Execute(input_image)
    return resampled_image



def cut_mix(
        img1: sitk.Image,
        label1: sitk.Image,
        img2: sitk.Image,
        label2: sitk.Image,
        resize=None
):
    '''
    将image1和image2进行cutmix，并对其label做同样的操作
    :param img1: SimpleITK image     :param label1:
    :param img2: SimpleITK image    :param label2:
    :param resize: 判断在cutmix阶段是否需要resize，避免分辨率不一致
    :return: sitk.Image, images and labels after CutMix process
    '''

    # Save image1 information
    direction1 = img1.GetDirection()
    origin1 = img1.GetOrigin()
    spacing1 = img1.GetSpacing()

    # Save image2 information
    direction2 = img2.GetDirection()
    origin2 = img2.GetOrigin()
    spacing2 = img2.GetSpacing()

    # Get numpy form
    img1 = sitk.GetArrayFromImage(img1)
    label1 = sitk.GetArrayFromImage(label1)
    shape1 = img1.shape

    img2 = sitk.GetArrayFromImage(img2)
    label2 = sitk.GetArrayFromImage(label2)
    shape2 = img2.shape

    # Set image1 and image2 to the same resolution
    if resize is not None:
        # If need resize before cutmix
        transform = Compose([
            ToTensor(),
            AddChanneld(keys=['image', 'label']),
            Resized(keys=['image', 'label'], spatial_size=resize, mode=['area', 'nearest'])
        ])
    else:
        # Do not need resize
        transform = Compose([
            ToTensor(),
            AddChanneld(keys=['image', 'label'])
        ])
    # Transform image1 to original resolution
    transform_back1 = Compose([
        # Resized(keys=['image', 'label'], spatial_size=shape1, mode=['area', 'nearest'])
    ])
    # Transform image2 to original resolution
    transform_back2 = Compose([
        # Resized(keys=['image', 'label'], spatial_size=shape2, mode=['area', 'nearest'])
    ])

    data_dict = {'image': img1, 'label': label1}
    trans_data_dict = apply_transform(transform, data_dict)
    img1 = trans_data_dict['image']
    label1 = trans_data_dict['label']
    # print(img1.shape, label1.shape)

    data_dict = {'image': img2, 'label': label2}
    trans_data_dict = apply_transform(transform, data_dict)
    img2 = trans_data_dict['image']
    label2 = trans_data_dict['label']
    # print(img2.shape, label2.shape)

    # Get CutMix region
    x = int(random.uniform(low=0, high=img1.shape[1]))
    y = int(random.uniform(low=0, high=img1.shape[2]))
    z = int(random.uniform(low=0, high=img1.shape[3]))
    d = int(random.uniform(low=1, high=img1.shape[1] - x - 1))
    h = int(random.uniform(low=1, high=img1.shape[2] - y - 1))
    w = int(random.uniform(low=1, high=img1.shape[3] - z - 1))

    # print("(x,y,z)==>", x, y, z)
    # print("(d,h,w)==>", d, h, w)

    # Creat CutMix mask to combine images
    mask1 = torch.ones_like(img1)
    mask1[x:x + d, y:y + h, z:z + w] = 0
    mask2 = torch.zeros_like(img1)
    mask2[x:x + d, y:y + h, z:z + w] = 1
    # CutMix image1 and label1
    mix_img1 = img1 * mask1 + img2 * mask2
    mix_label1 = label1 * mask1 + label2 * mask2

    # CutMix image2 and label2
    mix_img2 = img2 * mask1 + img1 * mask2
    mix_label2 = label2 * mask1 + label1 * mask2

    # Restore resolution of image1
    data_dict = {'image': mix_img1, 'label': mix_label1}
    trans_back_data = apply_transform(transform_back1, data_dict)
    mix_img1 = trans_back_data['image']
    mix_label1 = trans_back_data['label']

    # Restore resolution of image2
    data_dict = {'image': mix_img2, 'label': mix_label2}
    trans_back_data = apply_transform(transform_back2, data_dict)
    mix_img2 = trans_back_data['image']
    mix_label2 = trans_back_data['label']

    # Make numpy to simpleITK image
    # Restore image1 information
    mix_img1 = sitk.GetImageFromArray(mix_img1.squeeze(dim=0))
    mix_img1.SetOrigin(origin1)
    mix_img1.SetDirection(direction1)
    mix_img1.SetSpacing(spacing1)

    # Restore label1 information
    mix_label1 = sitk.GetImageFromArray(mix_label1.squeeze(dim=0))
    mix_label1.SetOrigin(origin1)
    mix_label1.SetDirection(direction1)
    mix_label1.SetSpacing(spacing1)

    # Restore image2 information
    mix_img2 = sitk.GetImageFromArray(mix_img2.squeeze(dim=0))
    mix_img2.SetOrigin(origin2)
    mix_img2.SetDirection(direction2)
    mix_img2.SetSpacing(spacing2)

    # Restore label2 information
    mix_label2 = sitk.GetImageFromArray(mix_label2.squeeze(dim=0))
    mix_label2.SetOrigin(origin2)
    mix_label2.SetDirection(direction2)
    mix_label2.SetSpacing((spacing2))
    return mix_img1, mix_label1, mix_img2, mix_label2


def get_data(root, dataset_name):
    # print(os.listdir(path=root))
    assert dataset_name in ['bch', 'oasis4', 'oasis35', 'verse']

    img_paths = []
    label_paths = []
    for file in os.listdir(path=root):
        file_path = os.path.join(root, file)
        if os.path.isdir(file_path):
            if dataset_name == 'bch':
                img_path = os.path.join(file_path, "brain.nii.gz")
                label_path = os.path.join(file_path, "label.nii.gz")
            elif dataset_name == 'oasis4':
                img_path = os.path.join(file_path, "aligned_norm.nii.gz")
                label_path = os.path.join(file_path, "aligned_seg4.nii.gz")
            elif dataset_name == 'oasis35':
                img_path = os.path.join(file_path, "aligned_norm.nii.gz")
                label_path = os.path.join(file_path, "aligned_seg35.nii.gz")
            elif dataset_name == 'verse':
                img_path = os.path.join(file_path, "img.nii.gz")
                label_path = os.path.join(file_path, "label.nii.gz")

            img_paths.append(img_path)
            label_paths.append(label_path)
        else:
            print(file_path, " is not a directory")

    return img_paths, label_paths


def get_weight(label_path, map, discard, merge, f_order=10., s_order=3.):
    # Generate label weights based on it's volumetric
    label = sitk.ReadImage(label_path)
    label = sitk.GetArrayFromImage(label)
    if map:
        label = label_mapping(label, discard=discard, merge=merge)
    element, cnt = np.unique(label, return_counts=True)
    cnt = np.log(cnt)
    weights = 1 / cnt * f_order
    weights = weights ** s_order
    # softmax
    weights = np.exp(weights - np.max(weights))
    weights = weights / np.sum(weights)
    print("Label index: ", element)
    print("Label weights: ", weights)
    return weights


def get_region_weight(true_label, dice):
    label_weights = 1 / (dice + 1e-3)
    region_weights = torch.ones_like(true_label)
    for i in range(len(dice)):
        region_weights[true_label == i] = label_weights[i]
    return torch.sqrt(region_weights)


def get_supervised_data(
        train_img_paths: list[str],
        val_img_paths: list[str],
        size: tuple,
        train_num: int,
        dataset_name: str
) -> list[str]:
    img_transformer = Compose([
        ToTensor(),
        AddChannel(),
        Resize(spatial_size=size, mode='area'),
        AddChannel()
    ])
    # Calculate average validation image
    average_val_img = None
    print("Calculate average validation image")
    for img_path in tqdm(val_img_paths):
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        img = img_transformer(img)
        img = img / torch.max(img)

        if average_val_img is None:
            average_val_img = img
        else:
            average_val_img += img

    average_val_img /= len(val_img_paths)
    # Compare with average validation image

    print("Calculate similarity")
    sim = []
    supervised_img_paths = []
    supervised_label_paths = []
    for img_path in tqdm(train_img_paths):
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        img = img_transformer(img)
        img = img / torch.max(img)

        similarity = 1 / torch.sum((img - average_val_img) ** 2)
        if len(sim) < train_num:
            sim.append(similarity)
            supervised_img_paths.append(img_path)
            if dataset_name == 'oasis35':
                supervised_label_paths.append(img_path.replace('aligned_norm', "aligned_seg35"))
            elif dataset_name == 'oasis4':
                supervised_img_paths.append(img_path.replace('aligned_norm', "aligned_seg4"))
        else:
            # Choose the dissimilar one
            switch_index = np.argmin(sim)
            if similarity > sim[switch_index]:
                # Update
                sim[switch_index] = similarity
                supervised_img_paths[switch_index] = img_path
                if dataset_name == 'oasis35':
                    supervised_label_paths[switch_index] = img_path.replace('aligned_norm', "aligned_seg35")
                elif dataset_name == 'oasis4':
                    supervised_img_paths[switch_index] = img_path.replace('aligned_norm', "aligned_seg4")

    return supervised_img_paths, supervised_label_paths


if __name__ == '__main__':
    get_region_weight([1, 1, 32, 32, 32], torch.Tensor([0.1, 0.2, 0.3, 0.4]))
    '''
    img_pathes, label_pathes = get_data(root="E:\\DataSet\\neurite-oasis.v1.0", dataset_name="oasis35")
    print(img_pathes, label_pathes)
    print(len(img_pathes), len(label_pathes))
    img = sitk.ReadImage(img_pathes[0])
    label = sitk.ReadImage(label_pathes[0])
    print(img.GetSize(), label.GetSize())


    img1 = sitk.ReadImage("E:\\DataSet\\brainsegNew\\ct_brain\\3y\\BCH_00201\\brain.nii.gz")
    label1 = sitk.ReadImage("E:\\DataSet\\brainsegNew\\ct_brain\\3y\\BCH_00201\\label.nii.gz")

    img2 = sitk.ReadImage("E:\\DataSet\\brainsegNew\\ct_brain\\3y\\BCH_00202\\brain.nii.gz")
    label2 = sitk.ReadImage("E:\\DataSet\\brainsegNew\\ct_brain\\3y\\BCH_00202\\label.nii.gz")


    print(img1.GetSize(), label1.GetSize())
    print(img2.GetSize(), label2.GetSize())

    print(img1.GetDirection(), img1.GetOrigin(), img1.GetSpacing())
    print(img2.GetDirection(), img2.GetOrigin(), img2.GetSpacing())

    mix_img1, mix_label1, mix_img2, mix_label2 = cut_mix(img1, label1, img2, label2)
    print(mix_img1.GetSize(), mix_img2.GetSize())
    sitk.WriteImage(mix_img2, "C:\\Users\\Administrator\\Desktop\\mix_img2.nii.gz")
    sitk.WriteImage(mix_label2, "C:\\Users\\Administrator\\Desktop\\mix_label2.nii.gz")
    '''
