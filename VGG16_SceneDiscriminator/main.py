from dataset import myDataSet
from VGG16 import my_vgg16

from torch.utils.data import Dataset
from torchvision import transforms

import torch
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchsize = 16
epoch = 40


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
])


net = my_vgg16(numClass=5).to(device)

CatAndDog = myDataSet("./td/", data_transform)
cadSet = torch.utils.data.DataLoader(CatAndDog, batch_size=batchsize, shuffle=True, num_workers=0)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
loss = torch.nn.CrossEntropyLoss()

"""print(len(cadSet))
for batch_idx, (img, label) in enumerate(cadSet):
    print(batch_idx,"his label is ", label)"""

for epoch in range(100):
    ally = 0
    for batch_idx, (img, label) in enumerate(cadSet):
        img = img.to(torch.float32).to(device)

        label = label.to(device)
        optimizer.zero_grad()
        testlable = net(img)
        
        lossData = loss(testlable, label)
        lossData.backward()
        optimizer.step()
        print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(cadSet), lossData))
    print(testlable,label)   
        

torch.save(net.state_dict(), 'ckp/model.pth')

'''
batch_size=5 batch_idx=4999

'''
