from typing import Any, Tuple
from torch.utils.data import Dataset

from .mvtec import MVTEC
from .btad import BTAD
from .mvtec_3d import MVTEC3D
from .visa import VISA


class UnionDataset(Dataset):
    
    DATASET_CLASSES = {'mvtec': MVTEC, 'btad': BTAD, 'mvtec3d': MVTEC3D, 'visa': VISA}
    
    def __init__(self, 
                 dataset_names=('mvtec', 'btad', 'mvtec3d', 'visa'),
                 dataset_roots=('/data/data1/yxc/datasets/mvtec_anomaly_detection',
                                '/data/data1/yxc/datasets/btad',
                                '/data/data1/yxc/datasets/mvtec_3d_anomaly_detection',
                                '/data/data1/yxc/datasets/visa'),
                 **kwargs):
        self.datasets = {}
        self.ids_to_names = {}
        for di, (dataset_name, dataset_root) in enumerate(zip(dataset_names, dataset_roots)):
            assert dataset_name in self.DATASET_CLASSES.keys(), 'Must choose dataset from {}'.format(self.DATASET_CLASSES.keys())
            dataset_cls = self.DATASET_CLASSES[dataset_name]
            dataset = dataset_cls(root=dataset_root, class_name=None, train=True,
                                  img_size=kwargs.get('img_size'), crp_size=kwargs.get('crp_size'),
                                  msk_size=kwargs.get('msk_size'))
            self.datasets[dataset_name] = dataset
            self.ids_to_names[di] = dataset_name
        
        self.num_images = 0
        self.num_splits = [0] * len(self.datasets.keys())
        for i, dataset in enumerate(self.datasets.values()):
            self.num_images += len(dataset.image_paths)
            self.num_splits[i] = self.num_images
            
        self.CLASS_NAMES = []
        for dataset in self.datasets.values():
            self.CLASS_NAMES.extend(dataset.CLASS_NAMES)
        
        idx, self.class_to_idx = 0, {}
        # encoding a unified class to idx mapping
        for dataset in self.datasets.values():
            class_names = dataset.CLASS_NAMES
            for class_name in class_names:
                self.class_to_idx[class_name] = idx
                idx += 1
        
        # update the class to idx mapping for each dataset
        for dataset in self.datasets.values():
            dataset.update_class_to_idx(self.class_to_idx)
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, label, mask) where label is 0 for normal image and 1 for abnormal image.
        """
        di = self.get_dataset_id(idx + 1)
        dataset = self.datasets[self.ids_to_names[di]]
        if di != 0:  # get idx in the di-th dataset
            idx = idx - self.num_splits[di - 1]
        image, label, mask, file_name, img_type = dataset[idx]

        return image, label, mask, file_name, img_type
    
    def get_dataset_id(self, idx):
        for di, split_num in enumerate(self.num_splits):
            if idx <= split_num:
                return di