"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch.nn as nn
import torch


class RotHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                 num_regions=8, rot_output_dim=3, mask_output_dim=1, depth_output_dim=1, weight_output_dim = 2, freeze=False):
        super(RotHeadNet, self).__init__()

        self.freeze = freeze

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, 'Only support kenerl 1 and 3'
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.rot_output_dim = rot_output_dim
        self.mask_output_dim = mask_output_dim
        self.region_output_dim = num_regions + 1  # add one channel for bg
        self.depth_output_dim = depth_output_dim
        self.weight_output_dim = weight_output_dim
        output_dim = self.mask_output_dim + self.rot_output_dim + self.region_output_dim + self.depth_output_dim + self.weight_output_dim if output_dim is None else output_dim

        self.out_layer = nn.Conv2d(num_filters, output_dim, kernel_size=output_kernel_size, padding=pad, bias=True)

        self.scale_branch = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        # if self.freeze:
        #     with torch.no_grad():
        #         for i, l in enumerate(self.features):
        #             x = l(x)
        #         x3d, w2d = self.out_layer(x).split([3, 2], dim=1)
        #         scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
        # else:
        #     for i, l in enumerate(self.features):
        #         x = l(x)
        #     x3d, w2d = self.out_layer(x).split([3, 2], dim=1)
        #     scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
        # return x3d, w2d, scale
        if self.concat:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        if i == 3:
                            x = l(torch.cat([x, x_f16], 1))
                        elif i == 12:
                            x = l(torch.cat([x, x_f32], 1))
                        elif i == 21:
                            x = l(torch.cat([x, x_f64], 1))
                        x = l(x)
                        
                    return x.detach()
            else:
                for i, l in enumerate(self.features):
                    if i == 3:
                        x = torch.cat([x, x_f16], 1)
                    elif i == 12:
                        x = torch.cat([x, x_f32], 1)
                    elif i == 21:
                        x = torch.cat([x, x_f64], 1)
                    x = l(x)
                return x
        else:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        x = l(x)
                    scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
                    mask = x[:, : self.mask_output_dim, :, :]
                    norm = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
                    region = x[:, self.mask_output_dim + self.rot_output_dim : self.mask_output_dim + self.rot_output_dim + self.region_output_dim, :, :]
                    depth = x[:, self.mask_output_dim + self.rot_output_dim + self.refion_output_dim : self.mask_output_dim + self.rot_output_dim + self.refion_output_dim + self.weight_output_dim, :, :] # zty #TODO
                    weight = x[:, self.mask_output_dim + self.rot_output_dim + self.refion_output_dim + self.mask_output_dim + self.weight_output_dim: , :, :]
                    bs, c, h, w = norm.shape
                    norm = norm.view(bs, 6, self.rot_output_dim // 6, h, w)
                    norm_stc_x = norm[:, 0, :, :, :]
                    norm_stc_y = norm[:, 1, :, :, :]
                    norm_stc_z = norm[:, 2, :, :, :]
                    norm_dyn_x = norm[:, 3, :, :, :]
                    norm_dyn_y = norm[:, 4, :, :, :]
                    norm_dyn_z = norm[:, 5, :, :, :]
                    return (mask.detach(), norm_stc_x.detach(), norm_stc_y.detach(), norm_stc_z.detach(), norm_dyn_x.detach(), norm_dyn_y.detach(), norm_dyn_z.detach(), region.detach(), depth.detach(), weight.detach(), scale.detach())
            else:
                for i, l in enumerate(self.features):
                    x = l(x)
                    
                scale = self.scale_branch(x.flatten(2).mean(dim=-1)).exp()
                mask = x[:, : self.mask_output_dim, :, :]
                norm = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
                region = x[:, self.mask_output_dim + self.rot_output_dim : self.mask_output_dim + self.rot_output_dim + self.region_output_dim, :, :]
                depth = x[:, self.mask_output_dim + self.rot_output_dim + self.refion_output_dim : self.mask_output_dim + self.rot_output_dim + self.refion_output_dim + self.weight_output_dim, :, :] # zty #TODO
                weight = x[:, self.mask_output_dim + self.rot_output_dim + self.refion_output_dim + self.mask_output_dim + self.weight_output_dim: , :, :]
                bs, c, h, w = norm.shape
                norm = norm.view(bs, 6, self.rot_output_dim // 6, h, w)
                norm_stc_x = norm[:, 0, :, :, :]
                norm_stc_y = norm[:, 1, :, :, :]
                norm_stc_z = norm[:, 2, :, :, :]
                norm_dyn_x = norm[:, 3, :, :, :]
                norm_dyn_y = norm[:, 4, :, :, :]
                norm_dyn_z = norm[:, 5, :, :, :]
                return mask, norm_stc_x, norm_stc_y, norm_stc_z, norm_dyn_x, norm_dyn_y, norm_dyn_z, region, depth, weight, scale # zty #TODO

