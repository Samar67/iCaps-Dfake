#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import torch.nn as nn
import torch.nn.functional as F
import torch

import layers
import cls_hrnet

import cv2
import numpy as np
from skimage import feature

def get_lbP(paths):
    numPoints_p = 8
    radius_r = 1
    color_spaces = ["hsv", "yCrCb"]
    #hists = np.zeros((1,6,256,1))
    hists = np.zeros((1,6,14,14))
    for image in paths:
        img = cv2.imread(image)
        #internal_hists = np.zeros((1,256,1))
        internal_hists = np.zeros((1,14,14))
        for space in color_spaces:
            if space == "hsv":
                imgg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif space == "yCrCb":
                imgg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            for i in range(imgg.shape[2]):
                lbp = feature.local_binary_pattern(imgg[:,:,i], numPoints_p,radius_r, method="uniform")
                #hist,bins = np.histogram(lbp.ravel(),256,[0,256])   #(256,)
                hist,bins = np.histogram(lbp.ravel(),196,[0,196])   #(256,)
                hist = np.expand_dims(hist, axis=1)     #(256,1)
                hist = hist.reshape(14,14)
                hist = np.expand_dims(hist, axis=0)     #(1,256,1)
                internal_hists = np.concatenate((internal_hists,hist),axis=0)   #(N,256,1)
        internal_hists = np.delete(internal_hists,0,axis=0)
        internal_hists = np.expand_dims(internal_hists, axis=0) #(1,6,256,1)
        hists = np.concatenate((hists,internal_hists),axis=0) #(N,6,256,1)
    hists = np.delete(hists,0,axis=0)
    hists = torch.from_numpy(hists).float().to('cuda')
    
    return(hists) 

# Capsule model
class HRCapsModel(nn.Module):
    def __init__(self,  cfg, **kwargs):   
        super(HRCapsModel, self).__init__()
        
        #### Parameters
        self.sequential_routing = cfg['CapsNet']['seq_routing']
        ## General
        self.num_routing = cfg['CapsNet']['num_routing'] # >3 may cause slow converging

        #from_confg_file
        self.primary = cfg['CapsNet']['extra']['primary_capsules']
        self.capsules = cfg['CapsNet']['extra']['capsules']
        self.classs = cfg['CapsNet']['extra']['class_capsules']

        #Normalizing Histograms
        self.hist_norm = nn.BatchNorm2d(6, affine=False)

        ## Primary Capsule Layer
        self.pc_num_caps = self.primary['num_caps']
        self.pc_caps_dim = self.primary['caps_dim']
        self.pc_output_dim = self.primary['out_img_size']
        
        #### Building Networks
        ## Backbone (before capsule)
        self.pre_caps = cls_hrnet.HighResolutionNet(cfg)

        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(  in_channels=self.primary['input_dim'],
                                    out_channels=self.primary['num_caps'] *\
                                                 self.primary['caps_dim'],
                                    kernel_size=self.primary['kernel_size'],
                                    stride=self.primary['stride'],
                                    padding=self.primary['padding'],
                                    bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(self.primary['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(self.capsules)):
            if self.capsules[i]['type'] == 'CONV':
                in_n_caps = self.primary['num_caps'] if i==0 else \
                                                               self.capsules[i-1]['num_caps']
                in_d_caps = self.primary['caps_dim'] if i==0 else \
                                                               self.capsules[i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.CapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=self.capsules[i]['num_caps'],
                                out_d_capsules=self.capsules[i]['caps_dim'],
                                kernel_size=self.capsules[i]['kernel_size'], 
                                stride=self.capsules[i]['stride'], 
                                matrix_pose=self.capsules[i]['matrix_pose'], 
                                dp=cfg['CapsNet']['dp'],
                                coordinate_add=False
                            )
                )
            elif self.capsules[i]['type'] == 'FC':
                if i == 0:
                    in_n_caps =self.primary['num_caps'] * self.primary['out_img_size'] * self.primary['out_img_size']
                    in_d_caps = self.primary['caps_dim']
                elif self.capsules[i-1]['type'] == 'FC':
                    in_n_caps = self.capsules[i-1]['num_caps']
                    in_d_caps = self.capsules[i-1]['caps_dim']                                           
                elif self.capsules[i-1]['type'] == 'CONV':
                    in_n_caps = self.capsules[i-1]['num_caps'] * self.capsules[i-1]['out_img_size'] *\
                                                                                        self.capsules[i-1]['out_img_size']  
                    in_d_caps = self.capsules[i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.CapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=self.capsules[i]['num_caps'], 
                          out_d_capsules=self.capsules[i]['caps_dim'], 
                          matrix_pose=self.capsules[i]['matrix_pose'],
                          dp=cfg['CapsNet']['dp']
                          )
                )
                                                               
        ## Class Capsule Layer
        if not len(self.capsules)==0:
            if self.capsules[-1]['type'] == 'FC':
                in_n_caps = self.capsules[-1]['num_caps']
                in_d_caps = self.capsules[-1]['caps_dim']
            elif self.capsules[-1]['type'] == 'CONV':    
                in_n_caps = self.capsules[-1]['num_caps'] * self.capsules[-1]['out_img_size'] *\
                                                                                   self.capsules[-1]['out_img_size']
                in_d_caps = self.capsules[-1]['caps_dim']
        else:
            in_n_caps = self.primary['num_caps'] * self.primary['out_img_size'] * self.primary['out_img_size']
            in_d_caps = self.primary['caps_dim']
        self.capsule_layers.append(
            layers.CapsuleFC(in_n_capsules= in_n_caps, 
                  in_d_capsules= in_d_caps, 
                  out_n_capsules= self.classs['num_caps'], 
                  out_d_capsules= self.classs['caps_dim'], 
                  matrix_pose= self.classs['matrix_pose'],
                  dp=cfg['CapsNet']['dp']
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(self.classs['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(self.classs['num_caps'], self.classs['caps_dim']))

    def forward(self, x, path, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)    
        #print("after HR output shape --> ", c.shape)
        hists = self.hist_norm(get_lbP(path))
        c = torch.cat([c,hists],dim=1)
        #print("after Hist output shape --> ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        #print("after Primary capsule shape --> ", u.shape)
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
        #print("before routing shape --> ", init_capsule_value.shape)

        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        #print(out)
        out = self.final_fc(out) # fixed classifier for all capsules
        #print(out)
        out = out.squeeze() # fixed classifier for all capsules
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        
        return out 