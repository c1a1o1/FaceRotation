import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

#the ImageList class is the dataloader of light cnn, the MyDataset class is the dataloader of tp-gan
#I suggest you better understand what is the function of every line of code for fitting your dataloader
#if you want to convert to gray image, modify the convert function using 'L' instead of 'RGB'
#here I load the path of each image instead of storing it in the memory, you can change it if you want

#light cnn dataloader
class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform = None):
        self.root      = root
        self.transform = transform

        self.imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                imgPath, label = line.strip().split(' ')
                self.imgList.append((imgPath, int(label)))

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(os.path.join(self.root, imgPath)).convert('RGB').resize(128, 128)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

#tp-gan dataloader
class MyDatasets(data.Dataset):
    def __init__(self, root, fileList, transform = None):
        self.root = root
        self.transform = transform

        self.imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                img_profile, img_frontal, img_label = line.strip().split(' ')
                self.imgList.append((img_profile, img_frontal, int(img_label)))
    
    def __getitem__(self, index):
        image_profile_name, image_frontal_name, image_label = self.imgList[index]

        image_profile_path = os.path.join(self.root, image_profile_name)
        image_frontal_path = os.path.join(self.root, image_frontal_name)

        image_profile = Image.open(image_profile_path).convert('RGB')
        image_frontal = Image.open(image_frontal_path).convert('RGB')
        
        image_profile128 = self.transform(image_profile.resize((128, 128)))
        image_profile64 = self.transform(image_profile.resize((64, 64)))
        image_profile32 = self.transform(image_profile.resize((32, 32)))
        image_frontal = self.transform(image_frontal.resize((128, 128)))
        
        return image_profile128, image_profile64, image_profile32, image_frontal, image_label
    
    def __len__(self):
        return len(self.imgList)
 
#change the 4-dim ndarray into a batch of image whose type is ndarray
def recover_image(img):
   return (
       (
           img *
           np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)) +
           np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1))
       ).transpose(0, 2, 3, 1) * 255.
          ).clip(0, 255).astype(np.uint8)


#for calculating the average of loss, precision, time during training
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
