import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image,img_color_denormalize

from torch.cuda.amp import autocast, GradScaler
import cv2


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.to('cuda')
            map_gt = map_gt.to('cuda')

            map_res = self.model(data)
            if isinstance(map_res, (tuple, list)):
                map_res = map_res[0]

            t_f = time.time()
            t_forward += t_f - t_b
            loss = self.criterion(map_res, map_gt, data_loader.dataset.map_kernel)

            loss.backward()

            optimizer.step()

            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, \tTime: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / batch_idx, t_backward / batch_idx, map_res.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):

            data = data.to('cuda')
            map_gt = map_gt.to('cuda')

            with torch.no_grad():

                map_res = self.model(data)
                if isinstance(map_res, (tuple, list)):
                    map_res = map_res[0]

            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss = self.criterion(map_res, map_gt, data_loader.dataset.map_kernel)

            losses += loss.item()

            pred = (map_res > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            plt.tight_layout()
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)


        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, 20, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                        data_loader.dataset.base.__name__)
            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda

    def eval(self, data_loader, visualize=True):
        """
        简化版测试函数：仅推理第一个batch，绘制并保存map_res、imgs_res、feature列表的第一个通道图
        Args:
            data_loader: 数据加载器
            visualize: 是否可视化（固定为True，核心功能）
        """
        # 1. 模型切换为评估模式
        self.model.eval()
        # 确保保存目录存在
        os.makedirs(self.logdir, exist_ok=True)

        # 2. 仅遍历第一个batch
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
            if batch_idx != 0:  # 只处理第一个batch，后续直接跳出
                break

            with torch.no_grad():
                map_res, features = self.model(data)

            self._plot_and_save_map_res(map_res, save_path=os.path.join(self.logdir, 'map_res.jpg'))
            self._plot_and_save_features(features, save_dir=self.logdir)

            print(f"已完成第一个batch的推理，结果已保存至：{self.logdir}")
            break

        return map_res, features


    # ------------------------------ 新增辅助绘图函数 ------------------------------
    def _plot_and_save_map_res(self, map_res, save_path):
        """绘制并保存map_res（地面平面热力图，去除坐标轴/边框，JET色彩）"""
        # 转换为numpy并去除冗余维度
        map = map_res.detach().cpu().squeeze().numpy()
        map = map - map.min()
        map = map / (map.max() + 1e-8)
        map = np.uint8(255 * map)
        map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        map = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
        map.save(save_path)

        # 归一化（和add_heatmap_to_image保持一致）
        # map_res_np = map_res_np - map_res_np.min()
        # map_res_np = map_res_np / (map_res_np.max() + 1e-8)

        # 绘制热力图（无坐标轴、无边框）
        # plt.figure(figsize=(8, 6))
        # plt.imshow(map_res_np)  # JET色彩风格
        # plt.axis('off')  # 关闭坐标轴
        # # plt.tight_layout(pad=0)  # 去除留白
        # # 保存：无边框、无留白、高分辨率
        # plt.savefig(
        #     save_path,
        #     # dpi=150,
        #     # bbox_inches='tight',
        #     # pad_inches=0,
        #     # frameon=False  # 关闭图像边框
        # )
        # plt.close()
        print(f"map_res已保存：{save_path}")


    def _plot_and_save_imgs_res(self, imgs_res, data, save_dir):
        """绘制并保存imgs_res（各视角的head/foot热力图，叠加到原始图像）"""
        # 反归一化原始图像（用于叠加热力图）
        denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # 遍历每个视角的imgs_res
        for view_idx, img_res in enumerate(imgs_res):
            # img_res结构：[batch, 2, H, W]（2=head/foot）
            img_res_np = img_res.detach().cpu().squeeze()
            # 原始图像：[batch, 3, H, W] → 转换为PIL图像
            raw_img = denormalize(data[0, view_idx]).cpu().numpy().squeeze().transpose([1, 2, 0])
            raw_img = Image.fromarray((raw_img * 255).astype('uint8'))

            # 绘制并保存head热力图（第一个通道）
            head_heatmap = img_res_np[0]  # 第一个通道=head
            head_save_path = os.path.join(save_dir, f'imgs_res_view{view_idx}_head.jpg')
            head_cam = add_heatmap_to_image(head_heatmap, raw_img)  # 叠加热力图（自带JET色彩）
            head_cam.save(head_save_path)

            # 绘制并保存foot热力图（第二个通道）
            foot_heatmap = img_res_np[1]  # 第二个通道=foot
            foot_save_path = os.path.join(save_dir, f'imgs_res_view{view_idx}_foot.jpg')
            foot_cam = add_heatmap_to_image(foot_heatmap, raw_img)
            foot_cam.save(foot_save_path)

            print(f"视角{view_idx}的imgs_res已保存：{head_save_path}, {foot_save_path}")


    def _plot_and_save_features(self, features, save_dir):
        """遍历feature列表，绘制每个feature的第一个通道（JET色彩+无坐标轴/边框）"""
        # 遍历feature列表中的每个特征图
        for feat_idx, feat in enumerate(features):
            # feat结构：[batch, C, H, W] → 取第一个通道（C=0）
            print(feat.size())
            feat_first_channel = feat.detach().cpu().squeeze().numpy()
            print(feat_first_channel.shape)
            # feat_first_channel = Image.fromarray(cv2.applyColorMap(np.uint8(255 * feat_first_channel), cv2.COLORMAP_JET))

            # 关键：仿照add_heatmap_to_image做归一化（保证色彩范围一致）
            # feat_first_channel = feat_first_channel - feat_first_channel.min()
            # feat_first_channel = feat_first_channel / (feat_first_channel.max() + 1e-8)

            # 绘制特征图（JET色彩+无坐标轴/边框）
            plt.figure()
            plt.imshow(feat_first_channel)  # 替换灰度图为JET色彩
            plt.axis('off')  # 关闭所有坐标轴
            plt.tight_layout(pad=0)  # 去除图像周围留白
            # # # 保存：无边框、无留白、无坐标轴
            #
            save_path = os.path.join(save_dir, f'feature_{feat_idx}_first_channel.jpg')
            # map = feat.detach().cpu().squeeze()[0].numpy()
            # map = map - map.min()
            # map = map / (map.max() + 1e-8)
            # map = np.uint8(255 * map)
            # map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
            # map = Image.fromarray(cv2.cvtColor(map, cv2.COLOR_BGR2RGB))
            # map.save(save_path)

            plt.savefig(
                save_path,
                bbox_inches='tight',  # 裁剪到图像内容边界
                pad_inches=0,  # 裁剪后无额外留白
                # frameon=False,  # 关闭画布边框
            #     # dpi=150,
            #     # bbox_inches='tight',
            #     # pad_inches=0,
            #     # frameon=False  # 关闭图像边框
            )
            plt.close()

            print(f"Feature {feat_idx}第一个通道已保存：{save_path}")


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
