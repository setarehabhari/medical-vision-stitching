# from skimage import transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from PIL import Image, ImageOps
from numpy import random
import numpy as np
from torchvision import transforms



class NEHOCTDataset(Dataset):

    def __init__(self, data_file, root_dir, class_name_column, image_column, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column: column name from csv file where we take the file path
        """
        self.data_frame = pd.read_csv(data_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.class_name = class_name_column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, str(self.data_frame.loc[idx, self.image_column])
        )
        image = Image.open(img_name)
        image_class = self.data_frame.loc[idx, self.class_name]

        if self.transform:
            image = self.transform(image)

        # sample = {'x': image, 'y': image_class}

        return image, image_class
    



class NEHOCTNoisyDataset(Dataset):

    def __init__(self, data_file, root_dir, class_name_column, image_column, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            image_column: column name from csv file where we take the file path
        """
        self.data_frame = pd.read_csv(data_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_column = image_column
        self.class_name = class_name_column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, str(self.data_frame.loc[idx, self.image_column])
        )
        image = Image.open(img_name)
        image_class = self.data_frame.loc[idx, self.class_name]

        p = random.uniform(0, 1)
        if p > 0.5:
            image = ImageOps.invert(image)

        if self.transform:
            image = self.transform(image)
            if p > 0.5:
                mean = 0.572 #noprespective0.630 #prespective0.572 #justneh 0.589
                std = -0.406 #noprespective00.382 #0.406 #-0.413
            else:
                mean = 0.123 #0.138 #0.123 #0.099
                std = 0.208 #0.211 #0.208 #0.207
            
            image = (image - mean) / std
            noise = torch.tensor(np.random.normal(0, 0.1, image.size()), dtype=torch.float)
            image = image + noise

        # sample = {'x': image, 'y': image_class}

        return image, image_class