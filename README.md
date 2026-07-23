# MVCDet — 多视角行人检测训练

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA（训练必需）

## 安装

```bash
pip install -r MVCDet-multiviewx/requirements.txt
```

## 数据集下载

### MultiviewX（合成数据集）

```bash
git clone https://bgithub.xyz/hou-yz/MultiviewX.git
```

下载后目录结构：
```
MultiviewX/
├── Image_subsets/
├── calibrations/
└── annotations_positions/
```

### Wildtrack（真实场景数据集）

从 EPFL 官网下载：
https://www.epfl.ch/labs/cvlab/data/data-wildtrack/

下载后目录结构：
```
Wildtrack_dataset/
├── Image_subsets/
├── calibrations/
└── annotations_positions/
```

将数据集放到 `./Data/` 目录下（可自定义路径）：

```bash
mv MultiviewX ./Data/MultiviewX
mv Wildtrack_dataset ./Data/Wildtrack_dataset
```

## 训练

### MultiviewX

```bash
cd MVCDet-multiviewx
python main.py -d multiviewx --data_path ./Data/MultiviewX
```

### Wildtrack

```bash
cd MVCDet-wildtrack
python main.py -d wildtrack --data_path ./Data/Wildtrack_dataset
```

## 仅验证（使用预训练权重）

```bash
cd MVCDet-multiviewx
python eval.py -d multiviewx --model_path MultiviewDetector.pth --data_path ./Data/MultiviewX
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d / --dataset` | `multiviewx` / `wildtrack` | 数据集：`multiviewx` / `wildtrack` / `terrace` |
| `--data_path` | `./Data/{dataset_name}` | 数据集根目录 |
| `--arch` | `resnet18` | 骨干网络：`resnet18` / `vgg11` |
| `--variant` | `default` | 模型变体：`default` / `img_proj` / `res_proj` / `no_joint_conv` |
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `1` | 批次大小 |
| `--lr` | `0.1` | 学习率 |
| `--cls_thres` | `0.4` | 分类阈值 |
| `--seed` | `1` | 随机种子 |

## 许可证

仅供学术研究使用。
