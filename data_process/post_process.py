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


# def fill_large_holes_per_label(mask_image, kernel_radius=2):
#     """
#     针对多目标分割，使用形态学闭合填补每个目标label区域中的较大空洞。
#
#     Parameters:
#         mask_image (sitk.Image): 多目标分割的3D图像，包含多个目标label。
#         kernel_radius (int): 结构元素的半径，用于形态学闭合操作，默认值为3。
#
#     Returns:
#         sitk.Image: 填补空洞后的3D图像，每个label区域内的空洞被填充。
#     """
#     # 获取分割mask中的所有非零label
#     labels = sitk.GetArrayViewFromImage(mask_image)
#     unique_labels = sorted(set(labels.ravel()) - {0})  # 排除背景（值为0）
#
#     # 创建一个空白图像用于存储结果
#     filled_mask_image = sitk.Image(mask_image.GetSize(), sitk.sitkUInt8)
#     filled_mask_image.CopyInformation(mask_image)
#
#     # 定义结构元素（球形结构元素）
#     structuring_element = [kernel_radius] * 3
#
#     for label in unique_labels:
#         # 提取当前label的二值mask
#         binary_mask = sitk.BinaryThreshold(mask_image, lowerThreshold=float(label), upperThreshold=float(label), insideValue=1,
#                                            outsideValue=0)
#
#         # 膨胀操作（增加区域）
#         dilated_mask = sitk.BinaryDilate(binary_mask, structuring_element)
#
#         # 腐蚀操作（去除区域）
#         eroded_mask = sitk.BinaryErode(dilated_mask, structuring_element)
#
#         # 使用膨胀和腐蚀的结果进行形态学闭合操作
#         filled_binary_mask = sitk.BinaryDilate(eroded_mask, structuring_element)
#         print("filled_binary_mask value: ", np.unique(sitk.GetArrayViewFromImage(filled_binary_mask)))
#
#         # 将填补后的mask重新赋值为当前label
#         filled_binary_mask = sitk.Cast(filled_binary_mask, sitk.sitkUInt8) * label  # 保持标签值不变
#
#         # 将当前label填补后的mask合并到最终图像中
#         filled_mask_image = sitk.Or(filled_mask_image, filled_binary_mask)
#
#     return filled_mask_image

def fill_holes_per_label(image, mask):
    ori_spacing =mask.GetSpacing()
    ori_direction = mask.GetDirection()
    ori_origin = mask.GetOrigin()
    print(image.GetSize(), mask.GetSize())
    img_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    print(img_array.shape, mask_array.shape)
    filled_mask = np.zeros(mask_array.shape, dtype=np.uint8)
    whole_cranial_mask_array = ((img_array > 200) & (img_array < 1500)).astype(np.uint8)
    print(whole_cranial_mask_array.shape)
    # 根据分割结果计算最小边界框
    non_zero_coords  = np.argwhere(mask_array > 0)
    # 计算最小包围框
    z_min, y_min, x_min = non_zero_coords.min(axis=0)
    z_max, y_max, x_max = non_zero_coords.max(axis=0)

    # 确保框包含边界
    z_max += 1
    y_max += 1
    x_max += 1
    print(x_min, x_max)
    print(y_min, y_max)
    print(z_min, z_max)
    cranial_wo_bottom_mask_array = np.zeros(whole_cranial_mask_array.shape, dtype=np.uint8)
    cranial_wo_bottom_mask_array[z_min:z_max, y_min:y_max, x_min:x_max] = whole_cranial_mask_array[z_min:z_max, y_min:y_max, x_min:x_max]

    filled_mask_array = np.zeros(mask_array.shape, dtype=np.uint8)

    indices_mask_array = np.array(np.where(mask_array == 0))
    indices_cranial_wo_bottom_array = np.array(np.where(cranial_wo_bottom_mask_array == 1))
    print(indices_mask_array)
    print(indices_cranial_wo_bottom_array)
    for index in indices_mask_array.T:
        x, y, z = index
        distances = cdist([index], indices_cranial_wo_bottom_array.T, metric='euclidean').flatten()
        # 按距离升序排序，选择前 n 个最近的体素
        closest_indices = distances.argsort()[:100]

        # 获取这 n 个最近体素的索引
        closest_voxels = indices_cranial_wo_bottom_array[:, closest_indices]
        # 获取这些体素在 a 中对应的值
        closest_values = indices_mask_array[tuple(closest_voxels)]
        print(closest_values)




    # cranial_wo_bottom_mask = sitk.GetImageFromArray(cranial_wo_bottom_mask_array)
    # cranial_wo_bottom_mask.CopyInformation(image)
    # sitk.WriteImage(cranial_wo_bottom_mask, "./test_data/cranial_wo_bottom_mask.nii.gz")




if __name__ == '__main__':
    '''
    使用路径 E:\DataSet\brainsegNew\CT_0_2996\2y\BCH_02934 下的不使用 cutmix 训练的 dual_MBConv_VAE 训练的结果进行后处理测试
    '''
    img_path = "./test_data/Brain_w_bone_cropped.nii.gz"
    mask_path = "./test_data/dual_MBConv_VAE cranial pre_mask 2024-12-09.nii.gz"
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    largest_component_mask = extract_largest_connected_components_for_labels(mask)
    largest_component_fill_holes_mask = fill_holes_per_label(img, largest_component_mask)
    # sitk.WriteImage(largest_component_fill_holes_mask, "./test_data/post_process_test.nii.gz")