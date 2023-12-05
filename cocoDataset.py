import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import pandas as pd
from typing import List,Tuple
from pathlib import Path
import logging

class COCO_DS(Dataset):
    def __init__(self, img_dir, annotate_file, transform = ToTensor, target_transform = ToTensor):
        self.img_dir = Path(img_dir)
        self.img_label = pd.read_csv(annotate_file)
        self.img_label = self.img_label.groupby(['image_id', 'file_name']).apply(lambda x: x.caption.tolist()).reset_index(name='caption')
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.img_label.shape[0]

    def __getitem__(self, index) -> Tuple[List[List[int]], str]:
        # super().__getitem__()
        row = self.img_label.loc[index]
        try:
            img = read_image(str(self.img_dir.joinpath(row['file_name'])))
        except Exception as e:
            logging.info(f"Corrupt image {index=} {row['file_name']}, reuse first image to prevent failure")
            row = self.img_label.iloc[0]
            img = read_image(str(self.img_dir.joinpath(row['file_name'])))
        label = row['caption']
        if self.transform:
            if img.shape[0]==1:
                img = img.repeat(3, 1, 1)
                # print("Issue ")

            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        if len(label) == 0:
            print(row)
            print(self.target_transform(row['caption']))
            print("Issue")
        return img, label

