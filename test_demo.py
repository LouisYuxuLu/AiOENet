# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn

import numpy as np
import cv2
import time
import os
from AiOENet import *
from VGG16_SceneDiscriminator import Get_Type

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir):
    
	model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
	net = AiOENet()
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	model.load_state_dict(model_info['state_dict'])
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(model_info['optimizer'])
	cur_epoch = model_info['epoch']

	return model, optimizer,cur_epoch


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])


if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './dataset/Test_Test'
	result_dir = './Test_result'    
	testfiles = os.listdir(test_dir)
    
	IsGPU = 1    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir,IsGPU)

	for f in range(len(testfiles)):
		model.eval()
		with torch.no_grad():
			img_low = cv2.imread(test_dir + '/' + testfiles[f])
			img_lap = cv2.Laplacian(img_low,cv2.CV_8U)    #Get Laplacian Edge
            

			input_img_low = torch.from_numpy(hwc_to_chw(img_low).copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
			input_img_lap = torch.from_numpy(img_lap.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

			input_var = torch.cat((input_img_low,input_img_lap),1)
			Type = Get_Type(input_var)
            
			s = time.time()
			E_out = model(input_var,Type)
			e = time.time()   
			print(input_var.shape)       
			print('Time:%.4f'%(e-s))    
			E_out = chw_to_hwc(E_out.squeeze().cpu().detach().numpy())		               
			cv2.imwrite(result_dir + '/' + testfiles[f][:-4] + '_AiOENet.png',np.clip(E_out*255,0.0,255.0))

                
				
			
			

