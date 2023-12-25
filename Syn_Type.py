# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:37:30 2022

@author: Administrator
"""
import cv2
import numpy as np
import random
import math
import torch


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def LowLight(img):

    g = np.random.uniform(0.1,0.8)
    img_l = img*g
    
    return img_l
    

def Hazey(img):
    a = np.random.uniform(0.70,0.95)       
    t = np.random.uniform(0.1,0.5)
    img_l = img*t +a*(1-t)
    return img_l

def SandStorm(img):
    img = chw_to_hwc(img)
    Ar = np.random.uniform(0.60,0.75)
    Ag = Ar-np.random.uniform(0.15,0.20)
    Ab = Ag-np.random.uniform(0.15,0.20)
    t = np.random.uniform(0.2,0.5)
    
    
    img[:,:,0] =(img[:,:,0]-(1-Ab))*t+ Ab
    img[:,:,1] =(img[:,:,1]-(1-Ag))*t+ Ag
    img[:,:,2] =(img[:,:,2]-(1-Ar))*t+ Ar
    
    return hwc_to_chw(img)

degtorad = 0.01745329252
radtodeg = 1/degtorad;

def rand_lines(w,h,a,l,nrs):
    lines=[]
    
    for i in range(nrs):
        # randomize starting and ending points for 2D lines
        sx = random.randint(0,w-1)
        sy = random.randint(0,h-1)
        
        le = random.randint(1,l)
        ang = a + random.randint(0,10)
        ex = sx + int(le * math.sin(ang * degtorad))
        ey = sy + int(le * math.cos(ang * degtorad))
        
        # move the endpoints inside the image frame
        if ex<0: ex = 0
        if ex>w-1: ex=w-1
        if ey<0: ey = 0
        if ey>w-1: ey=h-1
        
        # add line to list
        lines.append({'sx':sx,'sy':sy,'ex':ex,'ey':ey})
        
    return lines

def add_rain(img, angle, drop_length, drop_thickness, drop_nrs, blur=4, intensity = 100):
    # create placehplder for rain
    rain=np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype='uint16')
    
    # generate random lines
    lines=rand_lines(rain.shape[1],rain.shape[0],angle,drop_length,drop_nrs)
    
    # draw lines to the image
    for l in lines:
        cv2.line(rain,(l['sx'],l['sy']),(l['ex'],l['ey']),(intensity,intensity,intensity),drop_thickness)
    
    # add blur to the lines
    rain = cv2.blur(rain,(blur,blur))
    
    return torch.from_numpy(rain/1.0).type(torch.FloatTensor)+img



def Rainy(img):    
    img = np.clip(chw_to_hwc(img)*255,0.0,255.0)
    angle = 100
    length = -1
    thickness = -1
    drop_nrs = -1
    blur = np.random.randint(4,6)
    intensity = 150
    
 
    # add rain to all images

    # if one/some of the parameters were not given: randomize for each image
    if angle == 100:
        rangle = random.randint(-90,90)
    else:
        rangle=angle
    if length == -1:
        rlength = random.randint(10,30)
    else:
        rlength = length
    if thickness == -1:
        rthickness = random.randint(1,2)
    else:
        rthickness = thickness
    if drop_nrs == -1:
        rdrop_nrs = random.randint(50,300)
    else:
        rdrop_nrs = drop_nrs


    rainy = add_rain(img,rangle,rlength,rthickness,rdrop_nrs,blur,intensity)

    return hwc_to_chw(rainy)/255.0

def Snowy(img):
    img = np.clip(chw_to_hwc(img)*255,0.0,255.0)

    snow_list = []
    h,w,c = img.shape
    
    snow=np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype='uint16')
    Snow=np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype='uint16')
    count = np.random.randint(50,300)
    
    for i in range(count):
        x = random.randrange(0, h)
        y = random.randrange(0, w)
        radius = random.randrange(1,5)
        snow_list.append([x, y, radius])
        
        
    for j in range(len(snow_list)):
           # 绘制雪花，颜色、位置、大小
        snow=np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype='uint16')
        intensity = np.random.uniform(0.1,0.85)
        cv2.circle(snow, (snow_list[j][1], snow_list[j][0]),snow_list[j][2],(255, 255, 255), -1)
        Snow = intensity*snow+Snow 
        
    snowy = torch.from_numpy(Snow/1.0).type(torch.FloatTensor)+img
    
    return hwc_to_chw(snowy)/255.0
