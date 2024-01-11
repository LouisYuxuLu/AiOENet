from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import PIL
import os
import numpy as np
from torch.autograd import Variable
import cv2

# data_transform = transforms.Compose([
#     transforms.Resize(size=(244, 244)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# im1 = Image.open("1.jpg")
# # print(im1.size())
# x = data_transform(im1)
# print(x)

def laplacian_filter(img):
    h,w,c = img.shape
    img_edge = np.zeros((h,w))
    img_cat = np.zeros((h,w,4))
    
    img_edge = img[:,:,0]*0.2989 + img[:,:,0]*0.5870 + img[:,:,0]*0.1140
    kernel = np.array([[-1,-1,-1], [-1,4,-1], [-1,-1,-1]]) # Laplacian核心
    filtered_img = cv2.filter2D(img_edge, -1, kernel)
    
    img_cat[:,:,0:3] = img
    img_cat[:,:,3]   = filtered_img   
    
    return img_cat


class myDataSet(Dataset):
    def __init__(self, root, transform):
        self.image_files = np.array([x.path for x in os.scandir(root)
                                     if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.transform = transform

    def __getitem__(self, item):
        x = Image.open(self.image_files[item]).resize((244,244))
        x_cat= laplacian_filter(np.asarray(x))
        
        print(self.image_files[item])
        
        x = self.transform(x_cat)
        thisLabel = []
        
        #print(self.image_files[item])
        
        if "clear" in self.image_files[item]:
            #print('111111')
            thisLabel = 0#Variable(torch.tensor([1,0,0,0,0])).float()
        elif "haze" in self.image_files[item]:
            #print('222222')
            thisLabel = 1#Variable(torch.tensor([0,1,0,0,0])).float()
        elif "low" in self.image_files[item]:
            #print('333333')
            thisLabel = 2#Variable(torch.tensor([0,0,1,0,0])).float()
        elif "rain" in self.image_files[item]:
            #print('444444')
            thisLabel = 3#Variable(torch.tensor([0,0,0,1,0])).float()
# =============================================================================
#         else:
#             print('555555')
#             thisLabel = Variable(torch.tensor([0,0,0,0,1])).float()
# =============================================================================

        elif "snow" in self.image_files[item]:
            #print('555555')
            thisLabel = 4#Variable(torch.tensor([0,0,0,0,1])).float()
        return x, thisLabel

    def __len__(self):
        return len(self.image_files)
    '''
    def __getitem__(self,index):
        img_path=self.imgs[index]
        if self.test:
            label=int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label=1 if 'dog' in img_path.split('/')[-1] else 0
        data=Image.open(img_path)
        data=self.transform(data)
        return data,label
    '''
    
