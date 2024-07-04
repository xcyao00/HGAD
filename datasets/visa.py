import os
import pandas
import torch
import numpy as np
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.transforms import RandomHorizontalFlip


VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class_to_idx = {'candle': 0, 'capsules': 1, 'cashew': 2, 'chewinggum': 3, 'fryum': 4,
                'macaroni1': 5, 'macaroni2': 6, 'pcb1': 7, 'pcb2': 8, 'pcb3': 9,
                'pcb4': 10, 'pipe_fryum': 11}
idx_to_class = {0: 'candle', 1: 'capsules', 2: 'cashew', 3: 'chewinggum', 4: 'fryum',
                5: 'macaroni1', 6: 'macaroni2', 7: 'pcb1', 8: 'pcb2', 9: 'pcb3',
                10: 'pcb4', 11: 'pipe_fryum'}


class VISA(Dataset):
    
    CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
               'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    def __init__(self, 
                 root: str,
                 class_name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
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
        if train:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.LANCZOS),
                #T.RandomRotation(5),
                T.CenterCrop(kwargs.get('crp_size')),
                T.ToTensor()])
        # test:
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size'), Image.LANCZOS),
                T.CenterCrop(kwargs.get('crp_size')),
                T.ToTensor()])
        # mask
        self.target_transform = T.Compose([
            T.Resize(self.masksize, Image.NEAREST),
            T.CenterCrop(self.masksize),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.class_to_idx = {'candle': 0, 'capsules': 1, 'cashew': 2, 'chewinggum': 3,
                'fryum': 4, 'macaroni1': 5, 'macaroni2': 6, 'pcb1': 7,
                'pcb2': 8, 'pcb3': 9, 'pcb4': 10, 'pipe_fryum': 11}
        self.idx_to_class = {0: 'candle', 1: 'capsules', 2: 'cashew', 3: 'chewinggum',
                4: 'fryum', 5: 'macaroni1', 6: 'macaroni2', 7: 'pcb1',
                8: 'pcb2', 9: 'pcb3', 10: 'pcb4', 11: 'pipe_fryum'}

    def __getitem__(self, idx):
        image_path, label, mask, img_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.img_types[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        image = self.normalize(self.transform(image))
        
        if label == 0:
            mask = torch.zeros([1, self.masksize, self.masksize])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            if self.class_name is None:
                class_name = image_path.split('/')[-5]
            else:
                class_name = self.class_name
            label = class_to_idx[class_name]
        
        return image, label, mask, os.path.basename(image_path[:-4]), img_type

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        split_csv_file = os.path.join(self.root, 'split_csv', '1cls.csv')
        csv_data = pandas.read_csv(split_csv_file)
        
        class_data = csv_data.loc[csv_data['object'] == self.class_name]
        
        if self.train:
            train_data = class_data.loc[class_data['split'] == 'train']
            image_paths = train_data['image'].to_list()
            image_paths = [os.path.join(self.root, file_name) for file_name in image_paths]
            labels = [0] * len(image_paths)
            mask_paths = [None] * len(image_paths)
            types = ['good'] * len(image_paths)
        else:
            image_paths, labels, mask_paths, types = [], [], [], []
            
            test_data = class_data.loc[class_data['split'] == 'test']
            test_normal_data = test_data.loc[test_data['label'] == 'normal']
            test_anomaly_data = test_data.loc[test_data['label'] == 'anomaly']
            
            normal_image_paths = test_normal_data['image'].to_list()
            normal_image_paths = [os.path.join(self.root, file_name) for file_name in normal_image_paths]
            image_paths.extend(normal_image_paths)
            labels.extend([0] * len(normal_image_paths))
            mask_paths.extend([None] * len(normal_image_paths))
            types.extend(['good'] * len(normal_image_paths))
            
            anomaly_image_paths = test_anomaly_data['image'].to_list()
            anomaly_mask_paths = test_anomaly_data['mask'].to_list()
            anomaly_image_paths = [os.path.join(self.root, file_name) for file_name in anomaly_image_paths]
            anomaly_mask_paths = [os.path.join(self.root, file_name) for file_name in anomaly_mask_paths]
            image_paths.extend(anomaly_image_paths)
            labels.extend([1] * len(anomaly_image_paths))
            mask_paths.extend(anomaly_mask_paths)
            types.extend(['anomaly'] * len(anomaly_image_paths))

        return image_paths, labels, mask_paths, types
    
    def _load_all_data(self):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_types = []
        for class_name in VISA_CLASS_NAMES:
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