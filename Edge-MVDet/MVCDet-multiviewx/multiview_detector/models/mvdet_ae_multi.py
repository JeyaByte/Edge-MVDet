import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
from torchvision.models.vgg import vgg11
# from multiview_detector.models.resnet import resnet18
import cv2
from multiview_detector.datasets.MultiviewX import MultiviewX
from multiview_detector.utils import projection
import DCNv2

import math
from multiview_detector.models.transformer import TransformerEncoderLayer, TransformerEncoder

# from multiview_detector.models.block import *


from torchvision.models.resnet import ResNet18_Weights  # 适配新版PyTorch
from torchvision.models import resnet18


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.pronums = 5
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.worldgrid2worldcoord_mat, dataset.base)

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))

        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam * self.pronums)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            # base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, False, False]).children())[:-2])
            self.base_pt = ResNetMultiScaleFusion(out_channels=512).to('cuda:0')

            # self.base_pt = base.to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')


        self.feat_classifier = nn.Sequential(
                                            # nn.Conv2d(out_channel, 8, 1), nn.ReLU(),
                                            nn.Conv2d(out_channel, 1, 1, bias=False)
                                            ).to('cuda:0')

        self.C = 256
        
        self.feat_decoder = nn.Sequential(
                                            # nn.Conv2d(out_channel, 8, 1), nn.ReLU(),
                                            nn.Conv2d(1, self.C, 1, bias=False),
                                            # nn.GroupNorm(1, 1),
                                            ).to('cuda:0')
        
        self.compress_conv = nn.Sequential(
                                            # nn.Conv2d(out_channel, 8, 1), nn.ReLU(),
                                            nn.Conv2d(self.pronums*self.C, self.C, 1, bias=False),
                                            # nn.GroupNorm(1, 1),
                                            ).to('cuda:0')



        hidden_dim = 128
        dropout = 0.1
        nhead = 8
        dim_feedforward = 512
        # downsample_in_channels = self.num_cam * self.pronums * 1 + 2
        downsample_in_channels = self.num_cam * self.pronums * 1

        self.se_module = LightweightSE(downsample_in_channels).to('cuda:0')

        self.downsample = nn.Sequential(
                                        # nn.Conv2d(downsample_in_channels, hidden_dim, 1, 1, bias=False),
                                        nn.Conv2d(self.C, hidden_dim, 3, 2, 1), nn.ReLU(),
                                        # nn.Conv2d(downsample_in_channels, hidden_dim, 3, 2, 1), nn.ReLU(),
                                        nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(),
                                        ).to('cuda:0')

        self.pos_embedding = create_pos_embedding(np.ceil(np.array(self.reducedgrid_shape) / 4).astype(int),
                                                  hidden_dim // 2)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, dropout=dropout, nhead=nhead,
                                                dim_feedforward=dim_feedforward).to('cuda:0')
        self.encoder = TransformerEncoder(encoder_layer, 3).to('cuda:0')

        self.upsample = nn.Sequential(
                                      nn.Upsample(np.ceil(np.array(self.reducedgrid_shape) / 2).astype(int).tolist(),
                                                  mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(),
                                      nn.Upsample(self.reducedgrid_shape, mode='bilinear'),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1), nn.ReLU(),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1), nn.ReLU(),
                                      nn.Conv2d(hidden_dim, 1, 1)
                                      ).to('cuda:0')


        # self.map_classifier = nn.Sequential(
        #                                     nn.Conv2d(self.C, 512, 3, padding=1), nn.ReLU(),
        #                                     DCNv2.DeformConv2d(512, 512, 2, padding=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
        #                                     nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)
        #                                     ).to('cuda:0')

        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        print_features = []

        for cam in range(self.num_cam):
            img_feature = self.base_pt(imgs[:, cam].to('cuda:0'))

            img_feature = self.feat_classifier(img_feature)

            # print(img_feature.size())

            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')

            # print_features.append(img_feature)

            img_feature = self.feat_decoder(img_feature)

            height_proj_feats = []

            for ih in range(self.pronums):
                proj_mat = self.proj_mats[ih * self.num_cam + cam].repeat([B, 1, 1]).float().to('cuda:0')
                if ih == 0:
                    world_feature = warp_perspective(img_feature.to('cuda:0'), proj_mat,
                                                            self.reducedgrid_shape)
                    # world_features.append(world_feature.to('cuda:0'))
                    height_proj_feats.append(world_feature.to('cuda:0'))

                else:
                    world_feature = warp_perspective(img_feature.to('cuda:0'), proj_mat,
                                                            self.reducedgrid_shape)
                    # world_features.append(world_feature.to('cuda:0'))
                    height_proj_feats.append(world_feature.to('cuda:0'))

            # 新加内容，池化方法聚合
            height_concat = torch.cat(height_proj_feats, dim=1)  # B×(C×5)×H_bev×W_bev
            cam_proj = self.compress_conv(height_concat)  # B×C×H_bev×W_bev
            world_features.append(cam_proj)

        # world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        # world_features = torch.cat(world_features, dim=1)

        # world_features = self.se_module(world_features)  # 特征加权

        cam_feats_stack = torch.stack(world_features, dim=1)  # B×num_cam×C×H×W
        world_features = cam_feats_stack.max(dim=1)[0]  # B×C×H×W（沿摄像头维度取max）

        # for cam in range(self.num_cam):
        #     img_feature = self.base_pt(imgs[:, cam].to('cuda:0'))

        #     img_feature = self.feat_classifier(img_feature)

        #     img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')

        #     for ih in range(self.pronums):
        #         proj_mat = self.proj_mats[ih * self.num_cam + cam].repeat([B, 1, 1]).float().to('cuda:0')
        #         if ih == 0:
        #             world_feature = warp_perspective(img_feature.to('cuda:0'), proj_mat,
        #                                                     self.reducedgrid_shape)
        #             world_features.append(world_feature.to('cuda:0'))

        #         else:
        #             world_feature = warp_perspective(img_feature.to('cuda:0'), proj_mat,
        #                                                     self.reducedgrid_shape)
        #             world_features.append(world_feature.to('cuda:0'))

        # # world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        # world_features = torch.cat(world_features, dim=1)

        # world_features = self.se_module(world_features)  # 特征加权

        x = self.downsample(world_features)
        _, _, H, W = x.shape
        # H*W,B,C*N
        pos_embedding = self.pos_embedding.repeat(B, 1, 1, 1).flatten(2).permute(2, 0, 1).to(x.device)
        x = self.encoder(x.flatten(2).permute(2, 0, 1), pos=pos_embedding)
        merged_feat = self.upsample(x.permute(1, 2, 0).view(B, -1, H, W))
        map_result = F.interpolate(merged_feat, self.reducedgrid_shape, mode='bilinear')

        # map_result = self.map_classifier(world_features.to('cuda:0'))
        # map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        return map_result, print_features

    def get_imgcoord2worldgrid_matrices(self, worldgrid2worldcoord_mat, base_dataset=None):
        dataset = base_dataset or MultiviewX('./Data/MultiviewX')
        height = [0, 0.3, 0.6, 0.9, 0.15]
        projection_matrices = {}
        count = -1
        for iih in range(self.pronums):
            count += 1
            for cam in range(self.num_cam):
                xi = np.arange(0, 640, 40)
                yi = np.arange(0, 1000, 40)
                world_grid = np.stack(np.meshgrid(xi, yi, indexing='ij')).reshape([2, -1])
                world_coord = dataset.get_worldcoord_from_worldgrid(world_grid)
                img_coord = projection.get_imagecoord_from_worldcoord(world_coord, dataset.intrinsic_matrices[cam],
                                                                      dataset.extrinsic_matrices[cam], height[iih])

                img_n = []
                world_n = []
                for j in range(img_coord.shape[1]):
                    if 0 < img_coord[0, j] < 1920 and 0 < img_coord[1, j] < 1080:
                        img_n.append((img_coord[0, j], img_coord[1, j]))
                        world_n.append((world_coord[0, j], world_coord[1, j]))

                Homo, mask = cv2.findHomography(np.float32(world_n), np.float32(img_n), cv2.RANSAC, 5)
                worldcoord2imgcoord_mat = np.float32(Homo)

                worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat

                imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)

                permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                projection_matrices[count * 6 + cam] = permutation_mat @ imgcoord2worldgrid_mat
                pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def create_pos_embedding(img_size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    H, W = img_size
    not_mask = torch.ones([1, H, W])
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos




# 新增：多尺度特征融合模块
class ResNetMultiScaleFusion(nn.Module):
    """
    提取ResNet18的1/8、1/16、1/32特征，下采样对齐到1/32后融合
    """
    def __init__(self, out_channels=512, pretrain=True):
        super().__init__()
        # 拆分ResNet18，提取不同尺度特征
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        # resnet = resnet18(replace_stride_with_dilation=[False, False, False])
        # 基础层：到maxpool后（1/4尺度）
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # 1/8尺度特征层（layer2）
        self.layer2 = resnet.layer2  # 输出1/8
        # 1/16尺度特征层（layer3）
        self.layer3 = resnet.layer3  # 输出1/16
        # 1/32尺度特征层（layer4）
        self.layer4 = resnet.layer4  # 输出1/32

        # 下采样模块：将1/8→1/16→1/32，1/16→1/32
        self.downsample_8_to_16 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/8→1/16
        self.downsample_16_to_32 = nn.MaxPool2d(kernel_size=2, stride=2)  # 1/16→1/32

        # 1×1卷积调整通道数（统一为256，便于融合）
        # self.conv_1x1_8 = nn.Conv2d(128, 256, kernel_size=1, bias=False)  # layer2输出128通道
        # self.conv_1x1_16 = nn.Conv2d(256, 256, kernel_size=1, bias=False) # layer3输出256通道
        # self.conv_1x1_32 = nn.Conv2d(512, 256, kernel_size=1, bias=False) # layer4输出512通道

        # 融合用1×1卷积：拼接后256*3=768通道 → 目标512通道
        self.fusion_conv = nn.Conv2d(896, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 提取各尺度特征
        x_base = self.base(x)          # 1/4尺度
        x_8 = self.layer2(x_base)      # 1/8尺度, [B, 128, H/8, W/8]
        x_16 = self.layer3(x_8)        # 1/16尺度, [B, 256, H/16, W/16]
        x_32 = self.layer4(x_16)       # 1/32尺度, [B, 512, H/32, W/32]

        return x_32
'''
        # 1/8特征下采样到1/32
        x_8_down = self.downsample_8_to_16(x_8)  # 1/8→1/16
        x_8_down = self.downsample_16_to_32(x_8_down)  # 1/16→1/32
        # x_8_down = self.conv_1x1_8(x_8_down)     # 128→256通道

        # 1/16特征下采样到1/32
        x_16_down = self.downsample_16_to_32(x_16)  # 1/16→1/32
        # x_16_down = self.conv_1x1_16(x_16_down)     # 256→256通道

        # 1/32特征调整通道
        # x_32_adjust = self.conv_1x1_32(x_32)       # 512→256通道

        # ========== 核心修改：强制对齐尺寸 ==========
        # 以x_32_adjust的尺寸为基准，插值对齐其他特征图
        target_h, target_w = x_32.shape[2], x_32.shape[3]
        # 对齐x_8_down
        if x_8_down.shape[2:] != (target_h, target_w):
            x_8_down = F.interpolate(
                x_8_down,
                size=(target_h, target_w),
                mode='bilinear',  # 双线性插值，保留特征信息
                align_corners=False  # 避免边缘失真
            )
        # 对齐x_16_down
        if x_16_down.shape[2:] != (target_h, target_w):
            x_16_down = F.interpolate(
                x_16_down,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

        # 拼接融合 + 1×1卷积降维
        fused = torch.cat([x_8_down, x_16_down, x_32], dim=1)  # [B, 768, H/32, W/32]
        fused = self.relu(self.fusion_conv(fused))  # [B, 512, H/32, W/32]

        return fused
'''



# 第一步：定义轻量化SE模块（无通道数增加，仅加权）
class LightweightSE(nn.Module):
    def __init__(self, channels, reduction=4):  # reduction=4保证轻量化
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.fc(weight)
        return x * weight  # 特征加权，无维度变化