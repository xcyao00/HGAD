import os
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class_to_idx = {'bottle': 0, 'cable': 1, 'capsule': 2, 'carpet': 3,
                'grid': 4, 'hazelnut': 5, 'leather': 6, 'metal_nut': 7,
                'pill': 8, 'screw': 9, 'tile': 10, 'toothbrush': 11,
                'transistor': 12, 'wood': 13, 'zipper': 14}
idx_to_class = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                12: 'transistor', 13: 'wood', 14: 'zipper'}


class MVTEC(Dataset):
    """`MVTEC <>`_ Dataset.

    Args:
        root (string): Root directory of dataset, i.e ``../../mvtec_anomaly_detection``.
        train (bool, optional): If True, creates dataset for training, otherwise for testing.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Resize``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    MVTEC_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    def __init__(
            self, 
            root: str,
            class_name: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            **kwargs):
        #assert class_name in self.CLASS_NAMES, "class_name: '{}', should be in {}".format(class_name, self.CLASS_NAMES + ['all'])
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]
        self.masksize = kwargs.get('msk_size')
        
        # load dataset
        if self.class_name is None:  # load all classes
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
            self.class_name = None
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()
        
        # set transforms
        self.transform = transform
        if transform is None or transform == 'None':
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.LANCZOS),
                T.CenterCrop(kwargs.get('crp_size')),
                #T.GaussianBlur(5),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        # mask
        self.target_transform = target_transform
        if target_transform is None or target_transform == 'None':
            self.target_transform = T.Compose([
                T.Resize(self.masksize, Image.NEAREST),
                T.CenterCrop(self.masksize),
                T.ToTensor()])
        
        self.class_to_idx = {'bottle': 0, 'cable': 1, 'capsule': 2, 'carpet': 3,
                'grid': 4, 'hazelnut': 5, 'leather': 6, 'metal_nut': 7,
                'pill': 8, 'screw': 9, 'tile': 10, 'toothbrush': 11,
                'transistor': 12, 'wood': 13, 'zipper': 14}
        self.idx_to_class = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                        4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                        8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                        12: 'transistor', 13: 'wood', 14: 'zipper'}

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, label, mask) where label is 0 for normal image and 1 for abnormal image t.
        """
        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.img_types[idx]
        
        if self.class_name is None:
            class_name = image_path.split('/')[-4]
        else:
            class_name = self.class_name
            
        image = Image.open(image_path)
        if class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            image = np.expand_dims(np.array(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            mask = Image.open(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            label = class_to_idx[class_name]
        
        return image, label, mask, os.path.basename(image_path[:-4]), img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        image_dir = os.path.join(self.root, self.class_name, phase)
        mask_dir = os.path.join(self.root, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        return image_paths, labels, mask_paths, types
    
    def _load_all_data(self):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []
        for class_name in self.CLASS_NAMES:
            self.class_name = class_name
            image_paths, labels, mask_paths, types = self._load_data()
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_types.extend(types)
        return all_image_paths, all_labels, all_mask_paths, all_types
    
    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            self.class_to_idx[class_name] = class_to_idx[class_name]
        class_names = self.class_to_idx.keys()
        idxs = self.class_to_idx.values()
        self.idx_to_class = dict(zip(idxs, class_names))