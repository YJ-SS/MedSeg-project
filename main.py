from monai.networks.nets import UNet, SwinUNETR
import torch
import SimpleITK as sitk
import numpy as np
img = sitk.ReadImage("E:\\DataSet\\brainsegNew\\CT_0_2996\\2y\\BCH_00002\\Brain_w_bone_cropped.mha")
label = sitk.ReadImage("E:\\DataSet\\brainsegNew\\CT_0_2996\\2y\\BCH_00002\\Cranial_Head_bone_gt.mha")
print(img.GetSize(), label.GetSize())
label_array = np.array(label)
print(np.unique(label_array))