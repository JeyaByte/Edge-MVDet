import torch
from torch import nn
import torch.nn.functional as F


class GaussianMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, target, kernel):
        target = self._traget_transform(x, target, kernel)
        return F.mse_loss(x, target)

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target


class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        """
        论文指定的Focal Loss（用于占用图M_l的损失计算）
        :param alpha: 平衡因子（论文未指定，默认1.0，可根据数据调整）
        :param gamma: 聚焦参数（论文未指定，默认2.0，符合Focal Loss原始设置）
        :param reduction: 损失聚合方式（默认mean，适配批量训练）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-5


    def forward(self, x, target, kernel):
        """
        Args:
            x: 模型原始 logits [B, C, H, W]（未经过 sigmoid）
            target: 原始目标热力图 [B, C, H_gt, W_gt]
            kernel: 高斯卷积核 [1, 1, K, K]
        Returns:
            标量损失值
        """
        # 1. 对 target 进行高斯预处理（与原逻辑一致）
        target = self._traget_transform(x, target, kernel)  # [B, C, H, W], 值域 [0, 1]

        # 2. 将预测 logits 转为概率（数值稳定）
        pred = torch.sigmoid(x)
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

        # 3. 计算软标签 Focal Loss（CenterNet 风格）
        # 正样本部分: - (1 - pred)^α * log(pred) * target
        pos_loss = - (1 - pred) ** self.alpha * torch.log(pred) * target

        # 负样本部分: - (1 - target)^β * pred^α * log(1 - pred)
        neg_weights = (1 - target) ** self.beta
        neg_loss = - neg_weights * (pred ** self.alpha) * torch.log(1 - pred)

        # 4. 按有效目标区域归一化（避免背景主导）
        loss = (pos_loss + neg_loss).sum()
        normalization = target.sum() + self.eps  # 防止除零
        return loss / normalization

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target






