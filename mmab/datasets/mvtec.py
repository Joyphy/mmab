import os
import random

from mmab.registry import DATASETS
from torch.utils.data import Dataset
from mmengine.dataset import BaseDataset

from typing import List

@DATASETS.register_module()
class MVTecDataset(BaseDataset):
    def __init__(self, data_root, pipeline, is_train=True, is_random=True, num=0):
        self.is_train = is_train
        self.is_random = is_random
        self.num = num
        self.dataset_root = data_root
        super().__init__(data_root=data_root, pipeline=pipeline)

    def load_data_list(self) -> List[dict]:
        x, y, mask = self.load_dataset_folder()
        data_list = list(zip(x, y, mask))
        if self.is_random:
            random.shuffle(data_list)
        if self.num:
            data_list = data_list[:self.num]
        
        real_data_list = []
        for x, y, mask in data_list:
            if self.is_train:
                real_data_list.append({"img_path": x})
            else:
                real_data_list.append({
                    "img_path": x,
                    "gt": y,
                    "gt_seg_map": mask
                })
        return real_data_list

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        gt_dir = os.path.join(self.dataset_root, "..",
                              'ground_truth')

        img_types = sorted(os.listdir(self.dataset_root))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(self.dataset_root, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([
                os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)
                if f.endswith(('.png', ".jpg", ".bmp", ".jpeg"))
            ])

            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            elif not self.is_train:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [
                    os.path.splitext(os.path.basename(f))[0]
                    for f in img_fpath_list
                ]
                gt_fpath_list = [
                    os.path.join(gt_type_dir, img_fname + '_mask.png')
                    for img_fname in img_fname_list
                ]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)