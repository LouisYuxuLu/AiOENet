# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 16:14:37 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pdb import set_trace as stx
from einops import rearrange

import numbers

class AiOENet(nn.Module):
	def __init__(self,channel = 16):
		super(AiOENet,self).__init__()

		self.Haze_E = Encoder(channel)
		self.Low_E  = Encoder(channel)
		self.Rain_E = Encoder(channel)
		self.Snow_E = Encoder(channel)
        
		self.Share  =SNet(channel)
		#self.Share_D  = Decoder(channel)
        
		self.Haze_D = Decoder(channel)
		self.Low_D  = Decoder(channel)
		self.Rain_D = Decoder(channel)
		self.Snow_D = Decoder(channel)
        
		self.Haze_in = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.Haze_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)   

		self.Low_in = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.Low_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)  

		self.Rain_in = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.Rain_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False) 

		self.Snow_in = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)        
		self.Snow_out = nn.Conv2d(channel,3,kernel_size=1,stride=1,padding=0,bias=False)         
        
	def forward(self,x,Type):
        

		if   Type == 0:
			x_in = self.Haze_in(x)            
			L,M,S,SS = self.Haze_E(x_in)
			Share = self.Share(SS)
			x_out = self.Haze_D(Share,SS,S,M,L)
			out = self.Haze_out(x_out)# + x
            
		elif Type == 2:
			x_in = self.Low_in(x)             
			L,M,S,SS = self.Low_E(x_in)
			Share = self.Share(SS)
			x_out = self.Low_D(Share,SS,S,M,L)
			out = self.Low_out(x_out)# + x

		elif Type == 4:
			x_in = self.Rain_in(x)             
			L,M,S,SS = self.Rain_E(x_in)
			Share = self.Share(SS)
			x_out = self.Rain_D(Share,SS,S,M,L)
			out = self.Rain_out(x_out)# + x

		else:
			x_in = self.Snow_in(x)             
			L,M,S,SS = self.Snow_E(x_in)
			Share = self.Share(SS)
			x_out = self.Snow_D(Share,SS,S,M,L)
			out = self.Snow_out(x_out)#+ x            
            
		return out

class Encoder(nn.Module):
	def __init__(self,channel):
		super(Encoder,self).__init__()    

		self.el = ResidualBlock(channel)
		self.em = ResidualBlock(channel*2)
		self.es = ResidualBlock(channel*4)
		self.ess = ResidualBlock(channel*8)
		self.esss = ResidualBlock(channel*16)
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)  
		self.conv_esstesss = nn.Conv2d(8*channel,16*channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def forward(self,x):
        
		elout = self.el(x)
		x_emin = self.conv_eltem(self.maxpool(elout))
		emout = self.em(x_emin)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin)
		x_esin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_esin)

        
		return elout,emout,esout,essout#,esssout


class Decoder(nn.Module):
	def __init__(self,channel):
		super(Decoder,self).__init__()    

		#self.dsss = ResidualBlock(channel*16)
		self.dss = ResidualBlock(channel*8)
		self.ds = ResidualBlock(channel*4)
		self.dm = ResidualBlock(channel*2)
		self.dl = ResidualBlock(channel)
        
		self.conv_dssstdss = nn.Conv2d(16*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)           
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def _upsample(self,x):
		_,_,H,W = x.size()
		return F.upsample(x,size=(2*H,2*W),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):


		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout))
		dlout = self.dl(x_dlin+l)
        
		return dlout
    


class ResidualBlock(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(ResidualBlock,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel,  channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.sig= nn.Sigmoid()

		self.norm =nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1_1(x)))
		x_2 = self.act(self.norm(self.conv_2_1(x_1)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)


		return	x_out        
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        #print(x.shape,mu.shape)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
     
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class SNet(nn.Module):
	def __init__(self,channel):
		super(SNet,self).__init__()
  
		self.t1 = TransformerBlock(dim=int(channel*8), num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
		self.t2 = TransformerBlock(dim=int(channel*8), num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
		self.t3 = TransformerBlock(dim=int(channel*8), num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 
		self.t4 = TransformerBlock(dim=int(channel*8), num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') 

        
	def forward(self, x):
		T1 = self.t1(x)
		T2 = self.t2(T1)            
		T3 = self.t3(T2)
		T4 = self.t4(T3)


		return T4