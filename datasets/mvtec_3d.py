import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


MVTEC3D_CLASS_NAMES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
               'foam', 'peach', 'potato', 'rope', 'tire']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class_to_idx = {'bagel': 0, 'cable_gland': 1, 'carrot': 2, 'cookie': 3,
                'dowel': 4, 'foam': 5, 'peach': 6, 'potato': 7,
                'rope': 8, 'tire': 9}
idx_to_class = {0: 'bagel', 1: 'cable_gland', 2: 'carrot', 3: 'cookie',
                4: 'dowel', 5: 'foam', 6: 'peach', 7: 'potato',
                8: 'rope', 9: 'tire'}


class MVTEC3D(Dataset):
    
    CLASS_NAMES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
               'foam', 'peach', 'potato', 'rope', 'tire']
    
    def __init__(self, 
                 root,
                 class_name,
                 train=True,
                 img_size=256,
                 crp_size=256,
                 msk_size=256):
        self.dataset_path = root
        self.class_name = class_name
        self.train = train
        self.msk_size = msk_size
        # load dataset
        if self.class_name is None:  # load all classes
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_all_data()
            self.class_name = None
        else:
            self.image_paths, self.labels, self.mask_paths, self.img_types = self._load_data()
        # set transforms
        if train:
            self.transform = T.Compose([
                T.Resize(img_size, Image.LANCZOS),
                #T.RandomRotation(5),
                T.CenterCrop(crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform = T.Compose([
                T.Resize(img_size, Image.LANCZOS),
                T.CenterCrop(crp_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(msk_size, Image.NEAREST),
            T.CenterCrop(msk_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.class_to_idx = {'bagel': 0, 'cable_gland': 1, 'carrot': 2, 'cookie': 3,
                'dowel': 4, 'foam': 5, 'peach': 6, 'potato': 7,
                'rope': 8, 'tire': 9}
        self.idx_to_class = {0: 'bagel', 1: 'cable_gland', 2: 'carrot', 3: 'cookie',
                4: 'dowel', 5: 'foam', 6: 'peach', 7: 'potato',
                8: 'rope', 9: 'tire'}

    def __getitem__(self, idx):
        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.img_types[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        image = self.normalize(self.transform(image))
        
        if label == 0:
            mask = torch.zeros([1, self.msk_size, self.msk_size])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.transform_mask(mask)
        
        if self.train:
            if self.class_name is None:
                class_name = image_path.split('/')[-5]
            else:
                class_name = self.class_name
            label = self.class_to_idx[class_name]
        
        return image, label, mask, os.path.basename(image_path[:-4]), img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        phase = 'train' if self.train else 'test'
        image_paths, labels, mask_paths, types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'test')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type, 'rgb')
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
                gt_type_dir = os.path.join(gt_dir, img_type, 'gt')
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        return image_paths, labels, mask_paths, types
    
    def _load_all_data(self):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []
        for class_name in MVTEC3D_CLASS_NAMES:
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