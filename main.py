from monai.networks.nets import UNet, SwinUNETR
import torch
import SimpleITK as sitk
import numpy as np
label = sitk.ReadImage("E:\\DataSet\\neurite-oasis.v1.0\\OASIS_OAS1_0388_MR1\\aligned_seg4.nii.gz")
label = sitk.GetArrayFromImage(label)
print(label.shape)
print(np.unique(label))
print((label == 3).sum())

