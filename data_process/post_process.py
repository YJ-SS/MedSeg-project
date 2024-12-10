import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import cdist


def extract_largest_connected_components_for_labels(mask_image):
    """
    针对多目标分割mask，提取每个目标label的最大连通区域。

    Parameters:
        mask_image (sitk.Image): 多目标分割的3D图像，label值为正整数。

    Returns:
        sitk.Image: 仅保留每个label的最大连通区域的3D图像。
    """
    # 获取分割mask中的所有非零label
    labels = sitk.GetArrayViewFromImage(mask_image)
    unique_labels = sorted(set(labels.ravel()) - {0})  # 排除背景（值为0）

    # 创建一个空白图像用于存储结果
    largest_components_image = sitk.Image(mask_image.GetSize(), sitk.sitkUInt8)
    largest_components_image.CopyInformation(mask_image)

    for label in unique_labels:
        # 提取当前label的二值mask
        binary_mask = sitk.BinaryThreshold(mask_image, lowerThreshold=float(label), upperThreshold=float(label), insideValue=1,
                                           outsideValue=0)

        # 标记连通组件
        connected_components = sitk.ConnectedComponent(binary_mask)

        # 计算每个连通组件的大小
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(connected_components)

        # 找到最大的连通组件标签
        largest_label = max(label_stats.GetLabels(), key=lambda l: label_stats.GetPhysicalSize(l))

        # 提取最大的连通组件
        largest_component = sitk.BinaryThreshold(connected_components, largest_label, largest_label, 1, 0)

        # 将该最大连通组件重新赋值为当前label
        largest_component = sitk.Cast(largest_component, sitk.sitkUInt8) * label  # 修复乘法类型错误

        # 将当前结果合并到最终图像中
        largest_components_image = sitk.Add(largest_components_image, largest_component)

    return largest_components_image





if __name__ == '__main__':
    '''
    使用路径 E:\DataSet\brainsegNew\CT_0_2996\2y\BCH_02934 下的不使用 cutmix 训练的 dual_MBConv_VAE 训练的结果进行后处理测试
    '''
    img_path = "./test_data/Brain_w_bone_cropped.nii.gz"
    mask_path = "./test_data/dual_MBConv_VAE cranial pre_mask 2024-12-09.nii.gz"
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    largest_component_mask = extract_largest_connected_components_for_labels(mask)
    # largest_component_fill_holes_mask = fill_holes_per_label(img, largest_component_mask)
    # sitk.WriteImage(largest_component_fill_holes_mask, "./test_data/post_process_test.nii.gz")