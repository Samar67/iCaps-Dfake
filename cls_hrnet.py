# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.1
#logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        #print(out.size())
        return out


class HighResolutionModule(nn.Module):
    #2nd->(2,basic,[2,2],[16,32],[16,32],sum,true)
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            #logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            #logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            #logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches #2nd->2
        num_inchannels = self.num_inchannels #2nd->[16,32]
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1): #0,1
            fuse_layer = []
            for j in range(num_branches): #0,1
                if j > i: #i=0,j=1
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i: #0,0
                    fuse_layer.append(None)
                else: #i=1,j=0
                    conv3x3s = []
                    for k in range(i-j): #0
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches): #0,1
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)): #0,1
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False) # out_image = 112
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False) #out_image = 56
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        #print("Stage 1: ")
        self.stage1_cfg = cfg['HRNet']['extra']['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]   #32
        block = blocks_dict[self.stage1_cfg['block']]       #Bottleneck
        num_blocks = self.stage1_cfg['num_blocks'][0]       #1
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks) # Residual network 3 conv not-strided & kernel-> 1,3,1 out=in=128*56*56
        stage1_out_channel = block.expansion*num_channels # 4*32=128
        #print(f"{stage1_out_channel}")
        #print(self.layer1)

        #print("Stage 2: ")
        self.stage2_cfg = cfg['HRNet']['extra']['stage2']
        num_channels = self.stage2_cfg['num_channels'] #[16,32]
        block = blocks_dict[self.stage2_cfg['block']] #Basic
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))] # [16,32]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels) #parameter-> ([128],[16,32]) #3 conv2d (128,16,1),(128,128,2)&(128,128,2)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels) #(cfg,[16,32])
        #print(f"pre_stage_2_channels: {pre_stage_channels}")
        #print("Transition Layers:")
        #print(self.transition1)
        #print("Stage:")
        #print(self.stage2)


        #print("Stage 3: ")
        self.stage3_cfg = cfg['HRNet']['extra']['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block = blocks_dict[self.stage3_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)
        #print(f"pre_stage_3_channels: {pre_stage_channels}")
        #print("Transition Layers:")
        #print(self.transition2)
        #print("Stage:")
        #print(self.stage3)



        #print("Stage 4: ")
        self.stage4_cfg = cfg['HRNet']['extra']['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block = blocks_dict[self.stage4_cfg['block']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        #print(f"pre_stage_3_channels: {pre_stage_channels}")

        #print("Classification Head")
        # Classification Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)
        #print("inre_modules:")
        #print(self.incre_modules)
        #print("downsamp_modules:")
        #print(self.downsamp_modules)
        #self.classifier = nn.Linear(2048, 2)

    def _make_head(self, pre_stage_channels):
        #print("Making Head")
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]
        #print(f"prestage channels inside making the head: {pre_stage_channels}")
        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer( #2nd->([128],[16,32])
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer) #2nd->2
        num_branches_pre = len(num_channels_pre_layer) #2nd->1 

        transition_layers = []
        for i in range(num_branches_cur): #2nd->i=0,1
            if i < num_branches_pre: #2nd->mara if w mara else
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]: 
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):#2nd->0,1
                    inchannels = num_channels_pre_layer[-1]#2nd->128
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1): #1st->(1,64,32,1)
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        #print(downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        #print("in make layer")
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
            #print("first layer for loop")

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True): #2nd->[16,32]
        num_modules = layer_config['num_modules'] #2nd->1
        num_branches = layer_config['num_branches'] #2nd->2
        num_blocks = layer_config['num_blocks'] #2nd->[2,2]
        num_channels = layer_config['num_channels'] #2nd->[16,32]
        block = blocks_dict[layer_config['block']] #basic
        fuse_method = layer_config['fuse_method'] #sum

        modules = []
        for i in range(num_modules): #2nd->0
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output) #2nd->(2,basic,[2,2],[16,32],[16,32],sum,true)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print(f"before stage 1: x size= {x.size()}")
        #print(self.layer1(x))
        #print("inside layer 1")
        x = self.layer1(x)
        #print("outside layer 1")
        
        
        #print(f"after stage 1: x size= {x.size()}")
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        # print("eli tal3 mn stage 2")
        # for el in y_list:
        #     print(el.shape)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        # print("eli tal3 mn stage 3")
        # for el in y_list:
        #     print(el.shape)


        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        #print("eli tal3 mn stage 4")
        #for el in y_list:
        #    print(el.shape)
        # print("Classification head")
        #print(f"Before Classification Head: len = {y_list[0].shape} ")
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        #print("Before for in Final Layer", y.shape) #32*128*56*56
        
        #####################ana eli mzawdahaa tab3 fusion2.yaml
        for i in range(len(self.downsamp_modules)-1):
            #print(i, " Inside for in Final Layer- eli hyt3mlo down y=", y.shape)
            #print(i, " Inside for in Final Layer- eli hyt3mlo up y=", y_list[i+1].shape)
            k = self.incre_modules[i+1](y_list[i+1]) 
            #print(f"{i} incre moduele of [i+1] {self.incre_modules[i+1]}")
            l = self.downsamp_modules[i](y)
            y = k + l
            #print(f"iteration: {i}, k={k.shape}, l={l.shape}")
            #y = self.incre_modules[i+1](y_list[i+1]) + \
            #            self.downsamp_modules[i](y)
            #print(i, " Inside for in Final Layer", y.shape)  #32*256*28*28, 32*512*14*14

        ######################

        # for i in range(len(self.downsamp_modules)):
        #     y = self.incre_modules[i+1](y_list[i+1]) + \
        #                 self.downsamp_modules[i](y)
        #     #print(i, " Inside for in Final Layer", y.shape)  #32*256*28*28, 32*512*14*14, 32*1024*7*7

        # #print("Before Final Layer", y.shape)   #32*1024*7*7
        # y = self.final_layer(y)

        # if torch._C._get_tracing_state():
        #     y = y.flatten(start_dim=2).mean(dim=2)
        # else:
        #     y = F.avg_pool2d(y, kernel_size=y.size()
        #                          [2:]).view(y.size(0), -1)

        # y = self.classifier(y)

        return y

    def init_weights(self, pretrained='',):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            #logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #     logger.info('=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model
