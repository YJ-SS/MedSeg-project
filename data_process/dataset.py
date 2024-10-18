import sys


import torch
from torch.utils.data import Dataset
from monai.transforms import apply_transform, Resized
from torchvision import transforms
sys.path.append('..')
from data_process_method import cut_mix, mapping, label_mapping
from numpy import random
import SimpleITK as sitk
import numpy as np


class myDataSet(Dataset):
    def __init__(
            self,
            img_paths: list[str],
            label_paths: list[str],
            cutmix: bool,
            map: bool,
            discard: list,
            merge: list,
            transform2img=None,
            transform2both=None
    ):
        super().__init__()
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.cutmix = cutmix
        self.map = map
        self.discard = discard
        self.merge = merge
        self.transform2img = transform2img
        self.transform2both = transform2both
        # To distinguish the labeled data and unlabeled data
        self.has_label = False
        # 保存训练过程中是否进行resize
        self.resize = None
        if self.label_paths is not None:
            self.has_label = True

        if self.transform2both is not None:
            # 获取resize中的分辨率参数
            for t in self.transform2both.transforms:
                if isinstance(t, Resized):
                    self.resize =  t.resizer.spatial_size


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        global label
        img = sitk.ReadImage(self.img_paths[item])
        if self.has_label:
            label = sitk.ReadImage(self.label_paths[item])

        if self.cutmix:
            # Get random index for cutmix
            random_index = np.random.randint(low=0, high=self.__len__())
            while random_index == item:
                random_index = np.random.randint(low=0, high=self.__len__())
            # Use cutmix to shuffle pseudo label and real label
            random_img = sitk.ReadImage(self.img_paths[random_index])
            random_label = sitk.ReadImage(self.label_paths[random_index])
            img, label, _, _ = cut_mix(
                img1=img,
                label1=label,
                img2=random_img,
                label2=random_label,
                resize=self.resize,
            )

        img = sitk.GetArrayFromImage(img)
        if self.has_label:
            label = sitk.GetArrayFromImage(label)
            # 将label中的数据类型转换成int16.若为uint16会导致dataloader报错
            label = label.astype(np.int16)

        # print(type(img[0][0][0]), type(label[0][0][0]))

        # Make label continuous
        if self.has_label and self.map:
            label = label_mapping(label, discard=self.discard, merge=self.merge)

        if self.has_label and self.transform2both:
            data_dict = {'image': img, "label": label}
            trans_data = apply_transform(self.transform2both, data_dict)
            img = trans_data['image']
            label = trans_data['label']

        # Apply transformation
        if self.transform2img:
            img = apply_transform(self.transform2img, img)

        if self.has_label:
            return img, label
        else:
            return img





