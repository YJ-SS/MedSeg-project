from torch.utils.data import Dataset
from monai.transforms import apply_transform, Resized
import SimpleITK as sitk

class myDataset(Dataset):
    def __init__(
            self,
            img_path_list: list[str],
            label_list: list[int],
            transform=None,
    ):
        super(myDataset, self).__init__()
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        img = sitk.ReadImage(self.img_path_list[item])
        img = sitk.GetArrayFromImage(img)
        label = self.label_list[item]
        img = apply_transform(self.transform, img)
        return img, label


