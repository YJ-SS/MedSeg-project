import sys
sys.path.append("../seg_method/train_process")
import os
from train_process.record_func import log_print


def get_data_path_list(dataset_name, root_path, num_class):
    img_path_list = None
    label_path_list = None
    if dataset_name == 'oasis':
        assert (num_class == 4 or num_class == 35), \
            log_print("ERROR",
                      "Check training_info_config[dataset_num_class], oasis dataset only allows dataset_num_class to be"
                      " 4 or 35")
        img_path_list, label_path_list = get_oasis_MRI_path_list(
            root_path=root_path,
            num_class=num_class
        )
    elif dataset_name == 'cranial':
        img_path_list, label_path_list = get_cranial_CT_path_list(
            root_path=root_path,
        )

    return img_path_list, label_path_list


def get_cranial_CT_path_list(root_path: str)->tuple[list[str], list[str]]:
    '''
    @param root_path:
    @return:
    '''
    img_path_list = []
    label_path_list = []
    file_list = os.listdir(root_path)
    for file_name in file_list:
        file_path = os.path.join(root_path, file_name)
        if os.path.isdir(file_path):
            img_path = os.path.join(file_path, "Brain_w_bone_cropped.mha")
            '''
            Cranial_Head_bone_gt为矫正后真实值，但是只有1号2号图像有此文件
            Cranial_Head_bone_predict_mask_240830为根据开源模型预测值，所有图像都有此文件
            这里做以区分
            '''
            label_path = os.path.join(file_path, "Cranial_Head_bone_gt.mha")
            if not os.path.exists(label_path):
                label_path = os.path.join(file_path, "Cranial_Head_bone_predict_mask_240830.mha")
            if os.path.isfile(img_path) and os.path.isfile(label_path):
                img_path_list.append(img_path)
                label_path_list.append(label_path)
            else:
                log_print("WARNING", "{0} is not exist!!!".format(img_path))
                log_print("WARNING", "{0} is not exist!!!".format(label_path))
        else:
            log_print("WARNING", "{0} is not dir!!!".format(file_path))
    return img_path_list, label_path_list

def get_oasis_MRI_path_list(root_path: str, num_class=35)->tuple[list[str], list[str]]:
    '''

    :param root_path:
    :param num_class:
    :return:
    '''
    img_path_list = []
    label_path_list = []
    for file_name in os.listdir(root_path):
        file_path = os.path.join(root_path, file_name)
        if os.path.isdir(file_path):
            img_path = os.path.join(file_path, "aligned_norm.nii.gz")
            label_path = ""
            if num_class == 35:
                label_path = os.path.join(file_path, "aligned_seg35.nii.gz")
            elif num_class == 4:
                label_path = os.path.join(file_path, "aligned_seg4.nii.gz")
            if os.path.isfile(img_path) and os.path.isfile(label_path):
                img_path_list.append(img_path)
                label_path_list.append(label_path)
            else:
                log_print("WARNING", "{0} is not exist!!!".format(img_path))
                log_print("WARNING", "{0} is not exist!!!".format(label_path))
        else:
            log_print("WARNING", "{0} is not dir!!!".format(file_path))
    return img_path_list, label_path_list

if __name__ == '__main__':
    img_path_list, label_path_list = get_cranial_CT_path_list(root_path="E:\\DataSet\\brainsegNew\\CT_0_2996\\2y")
    print(len(img_path_list),len(label_path_list))
    print(img_path_list[:5], label_path_list[:5])
