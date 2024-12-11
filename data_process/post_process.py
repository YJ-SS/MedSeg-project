import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


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

def get_valid_boundary(mask_array: np.ndarray, sum_boundary=20):
    '''
    对于一个numpy格式的mask array，获取D H W方向的有效边界，避免遍历无效的体素
    :param sum_boundary:
    :param mask_array:
    :return:
    '''
    D, H, W = mask_array.shape
    # 记录每个维度有效帧开始和结束的位置
    D_start = 0
    D_end = D - 1
    H_start = 0
    H_end = H - 1
    W_start = 0
    W_end = W - 1
    while D_start < D_end and H_start < H_end and W_start < W_end:
        D_start_slice = mask_array[D_start,:,:]
        D_end_slice = mask_array[D_end,:,:]
        D_start_sum = np.sum(D_start_slice)
        D_end_sum = np.sum(D_end_slice)

        H_start_slice = mask_array[:,H_start,:]
        H_end_slice = mask_array[:,H_end,:]
        H_start_sum = np.sum(H_start_slice)
        H_end_sum = np.sum(H_end_slice)

        W_start_slice = mask_array[:,:,W_start]
        W_end_slice = mask_array[:,:,W_end]
        W_start_sum = np.sum(W_start_slice)
        W_end_sum = np.sum(W_end_slice)

        if D_start_sum < sum_boundary:
            D_start += 1
        if D_end_sum < sum_boundary:
            D_end -= 1
        if H_start_sum < sum_boundary:
            H_start += 1
        if H_end_sum < sum_boundary:
            H_end -= 1
        if W_start_sum < sum_boundary:
            W_start += 1
        if W_end_sum < sum_boundary:
            W_end -= 1
        if D_start_sum > 0 and D_end_sum > 0 and H_start_sum > 0 and H_end_sum > 0 and W_start_sum > 0 and W_end_sum > 0:
            break
    print(D, H, W)
    print("D_start: ", D_start, "D_end: ", D_end)
    print("H_start: ", H_start, "H_end: ", H_end)
    print("W_start: ", W_start, "W_end: ", W_end)
    return D_start, D_end, H_start, H_end, W_start, W_end


def fill_holes_per_label(pre_mask, cranial_img, kernel_size=7):
    pre_mask_array = sitk.GetArrayFromImage(pre_mask)
    cranial_img_array = sitk.GetArrayFromImage(cranial_img)
    # 从颅骨图像中通过阈值提取完整的颅骨mask 其中，在阈值内为1 表示目标颅骨，阈值外为0 表示背景 脑组织和不感兴趣的颅骨区域
    cranial_whole_mask_array = ((cranial_img_array > 200) & (cranial_img_array < 1300)).astype(int)
    D, H, W = pre_mask_array.shape
    D_start, D_end,H_start, H_end, W_start, W_end = get_valid_boundary(pre_mask_array)
    output = pre_mask_array.copy()
    # padding pre_mask_array and cranial_whole_mask_array
    padding_range = (kernel_size - 1) // 2
    pre_mask_array = F.pad(torch.tensor(pre_mask_array), tuple([padding_range] * 6), mode='constant', value=0)
    cranial_whole_mask_array = F.pad(torch.tensor(cranial_whole_mask_array), tuple([padding_range] * 6), mode='constant', value=0)
    print(pre_mask_array.shape)
    for z in range(padding_range + D_start, D_end):
        for y in range(padding_range + H_start, H_end):
            for x in range(padding_range + W_start, W_end):
                # 寻找目标颅骨区域预测值为0的地方进行填充
                is_target_cranial_region = (cranial_whole_mask_array[z,y,x] == 1)
                is_hole = (pre_mask_array[z,y,x] == 0)
                if is_target_cranial_region and is_hole:
                    # 获取卷积核内部范围
                    z_start, z_end = z - padding_range, z + padding_range
                    y_start, y_end = y - padding_range, y + padding_range
                    x_start, x_end = x - padding_range, x + padding_range
                    window = pre_mask_array[z_start:z_end, y_start:y_end, x_start:x_end]
                    flat_window = window.flatten()
                    value_cnt = Counter(flat_window.tolist())
                    most_common_value, _ = max(value_cnt.items(), key=lambda x: (x[1], x[0]) if x[0] != 0 else (-1, -1))
                    # print(most_common_value)
                    output[z - padding_range,y - padding_range,x - padding_range] = most_common_value

    output_img = sitk.GetImageFromArray(output)
    output_img.CopyInformation(pre_mask)
    return output_img















if __name__ == '__main__':
    '''
    使用路径 E:\DataSet\brainsegNew\CT_0_2996\2y\BCH_02934 下的不使用 cutmix 训练的 dual_MBConv_VAE 训练的结果进行后处理测试
    '''
    img_path = "./test_data/Brain_w_bone_cropped.nii.gz"
    mask_path = "./test_data/dual_MBConv_VAE cranial pre_mask 2024-12-09.nii.gz"
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    largest_component_mask = extract_largest_connected_components_for_labels(mask)
    sitk.WriteImage(largest_component_mask, "./test_data/largest_component_mask.nii.gz")
    print("largest_component_mask saved!")
    largest_component_fill_holes_mask = fill_holes_per_label(largest_component_mask, img)
    sitk.WriteImage(largest_component_fill_holes_mask, "./test_data/largest_component_fill_holes_mask.nii.gz")
    print("largest_component_fill_holes_mask saved!")