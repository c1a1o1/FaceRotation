import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

#the ImageList class is the dataloader of light cnn, the my_dataset class is the dataloader of tp-gan
#I suggest you better understand what is the function of every line of code for fitting your dataloader
#if you want to convert to gray image, modify the convert function using 'L' instead of 'RGB'
#here I load the path of each image instead of storing it in the memory, you can change it if you want

def default_loader(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((128, 128))
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

def read_image_list(filepath):
    image_list = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            image_profile, image_frontal, image_label = line.strip().split(' ')
            image_list.append((image_profile, image_frontal, int(image_label)))
    return image_list

#build your own datasets, inherit from torch.utils.data.Dataset
class my_datasets(data.Dataset):
    def __init__(self, data_root, fileList, transform):
        self.data_root = data_root
        self.transform = transform
        self.image_list = read_image_list(fileList)
    
    def __getitem__(self, index):
        image_profile_name, image_frontal_name, image_label = self.image_list[index]

        image_profile_path = os.path.join(self.data_root, image_profile_name)
        image_frontal_path = os.path.join(self.data_root, image_frontal_name)

        image_profile = Image.open(image_profile_path).convert('RGB')
        image_frontal = Image.open(image_frontal_path).convert('RGB')
        
        image_profile128 = self.transform(image_profile.resize((128, 128)))
        image_profile64 = self.transform(image_profile.resize((64, 64)))
        image_profile32 = self.transform(image_profile.resize((32, 32)))
        image_frontal = self.transform(image_frontal.resize((128, 128)))
        
        return image_profile128, image_profile64, image_profile32, image_frontal, image_label
    
    def __len__(self):
        return len(self.image_list)
