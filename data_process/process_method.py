import os
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
        img_path = os.path.join(file_path, "Brain_w_bone_cropped.nii.gz")
        label_path = os.path.join(file_path, "Cranial_Head_bone_predict_mask_240830.mha")
        if os.path.isfile(img_path) and os.path.isfile(label_path):
            img_path_list.append(img_path)
            label_path_list.append(label_path)
        else:
            print(img_path, "is not exist!!!")
            print(label_path, "is not exist!!!")
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
            print(img_path, "is not exist!!!")
            print(label_path, "is not exist!!!")
    return img_path_list, label_path_list

if __name__ == '__main__':
    img_path_list, label_path_list = get_oasis_MRI_path_list(root_path="E:\\DataSet\\neurite-oasis.v1.0")
    print(len(img_path_list),len(label_path_list))
