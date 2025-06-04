# util/datasets.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from util.quadatasetgpu import QuaternionWaveletNoise


class AddNoise:
    """
    图像加噪变换类。
    下游微调/评估阶段用此类在每个图像或图像批量上注入噪声，
    根据配置可以选择高斯噪声或四元数小波噪声。
    可直接在 torchvision.transforms.Compose 中使用，也可在 DataLoader 里对 batch 进行处理。
    """

    def __init__(self,
                 sigma: float,
                 use_quaternion_noise: bool = True,
                 levels: int = 1,
                 ratio: float = 3.0):
        """
        参数：
          sigma (float): 噪声标准差。若 sigma<=0，则不加噪。
          use_quaternion_noise (bool): 是否启用四元数小波噪声；否则使用像素级高斯噪声。
          levels (int): 小波分解层数，仅当 use_quaternion_noise=True 时生效。
          ratio (float): 高频与低频噪声方差比例，仅当 use_quaternion_noise=True 时生效。
        """
        self.sigma = sigma
        self.use_quaternion_noise = use_quaternion_noise
        self.levels = levels
        self.ratio = ratio

    def __call__(self, img):
        """
        对输入 img 应用噪声。支持以下几种格式：
          - PIL.Image: 先转为 [C,H,W] 的 Tensor([0,1])，再加噪后返回 Tensor([C,H,W])。
          - torch.Tensor:
            * 如果形状是 [C,H,W]（单张图像），先加 batch 维度变 [1,C,H,W]，加噪后再 squeeze 成 [C,H,W]；
            * 如果形状是 [B,C,H,W]（图像批量），直接对批量调用四元数小波噪声或像素级高斯噪声，返回同样形状 [B,C,H,W]。
        返回：同样格式（Tensor）的加噪结果。
        """
        # 如果输入是 PIL.Image，则转换成 Tensor([C,H,W]) 范围 [0,1]
        if not torch.is_tensor(img):
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

        # 此时 img 是 Tensor
        # 如果 sigma <= 0，则不加噪，直接返回原 img
        if not self.sigma or self.sigma <= 0:
            return img

        # 判断是否要使用四元数小波噪声
        if self.use_quaternion_noise:
            # 如果 img 已经是 [B, C, H, W] 批量格式
            if img.dim() == 4:
                # 直接将整个批量传给 apply_noise
                # apply_noise 将返回同样形状 [B, C, H, W]
                noisy = QuaternionWaveletNoise.apply_noise(
                    img,
                    sigma=self.sigma,
                    filter_name='haar',
                    levels=self.levels,
                    ratio=self.ratio,
                    device=img.device
                )
                print("quanoise")
                # 确保数值在 [0,1]
                if noisy.dtype.is_floating_point:
                    noisy = noisy.clamp(0.0, 1.0)
                return noisy

            # 如果 img 是单张图像 [C, H, W]
            elif img.dim() == 3:
                # 扩展到 [1, C, H, W]
                img_batch = img.unsqueeze(0)  # [1, C, H, W]
                noisy_batch = QuaternionWaveletNoise.apply_noise(
                    img_batch,
                    sigma=self.sigma,
                    filter_name='haar',
                    levels=self.levels,
                    ratio=self.ratio,
                    device=img.device
                )  # [1, C, H, W]
                noisy_img = noisy_batch.squeeze(0)  # [C, H, W]
                if noisy_img.dtype.is_floating_point:
                    noisy_img = noisy_img.clamp(0.0, 1.0)
                return noisy_img

            else:
                raise ValueError(f"AddNoise: 输入张量维度应为 3 或 4, 当前为 {img.dim()}")

        else:
            # 使用像素级高斯噪声
            # 如果是批量 [B, C, H, W]
            if img.dim() == 4:
                noise = torch.randn_like(img) * self.sigma  # [B, C, H, W]
                noisy = img + noise
                if noisy.dtype.is_floating_point:
                    noisy = noisy.clamp(0.0, 1.0)
                return noisy

            # 如果是单张 [C, H, W]
            elif img.dim() == 3:
                noise = torch.randn_like(img) * self.sigma  # [C, H, W]
                noisy_img = img + noise
                if noisy_img.dtype.is_floating_point:
                    noisy_img = noisy_img.clamp(0.0, 1.0)
                return noisy_img

            else:
                raise ValueError(f"AddNoise: 输入张量维度应为 3 或 4, 当前为 {img.dim()}")


class NoisyImageDataset(Dataset):
    """
    含加噪功能的数据集类，用于下游微调/评估阶段。
    支持 CIFAR-10 数据集 + 可选的四元数小波噪声或高斯噪声注入。
    """

    def __init__(self,
                 data_root: str,
                 train: bool,
                 input_size: int,
                 batch_size: int,
                 use_quaternion_noise: bool = False,
                 noise_sigma: float = 0.0,
                 levels: int = 1,
                 ratio: float = 3.0):
        """
        参数:
          data_root (str): CIFAR-10 根路径，会自动下载到该路径下。
          train (bool): True 表示训练集，False 表示测试/验证集。
          input_size (int): 模型期望的输入尺寸（正方形边长），图像会被 Resize 到 (input_size, input_size)。
          batch_size (int): DataLoader 的 batch_size，仅作构造时传参，可由外部覆盖。
          use_quaternion_noise (bool): 是否使用四元数小波噪声；否则使用像素级高斯噪声。
          noise_sigma (float): 噪声标准差 σ；若为 0 或负数，则不注入噪声。
          levels (int): 四元数小波分解层数，仅当 use_quaternion_noise=True 时生效。
          ratio (float): 高频/低频噪声方差比例，仅当 use_quaternion_noise=True 时生效。
        """
        self.data_root = data_root
        self.train = train
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_quaternion_noise = use_quaternion_noise
        self.sigma = noise_sigma
        self.levels = levels
        self.ratio = ratio

        # 构建 Transform 流水线
        transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),  # 将 PIL.Image -> Tensor([C,H,W])
        ]
        if self.sigma and self.sigma > 0:
            # 在 Tensor 阶段注入噪声
            transform_list.append(
                AddNoise(
                    sigma=self.sigma,
                    use_quaternion_noise=self.use_quaternion_noise,
                    levels=self.levels,
                    ratio=self.ratio
                )
            )
        # （可在外部添加 Normalize 等变换）

        self.transform = transforms.Compose(transform_list)

        # 加载 CIFAR-10 数据集
        self.dataset = CIFAR10(
            root=self.data_root,
            train=self.train,
            transform=self.transform,
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        返回 (noisy_img, label)：
          - noisy_img: Tensor([C, input_size, input_size])，已应用 Resize + ToTensor + AddNoise
          - label: 对应的整数标签 0~9
        """
        img, label = self.dataset[idx]
        return img, label

    def get_dataloader(self, shuffle: bool = True, num_workers: int = 4):
        """
        返回一个 DataLoader，便于训练/评估循环调用。
        shuffle: 是否打乱数据（训练时 True，验证时 False）。
        num_workers: 数据加载进程数，根据硬件情况调整。
        """
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


def build_dataset(is_train: str, args):
    """
    构建下游微调/评估用的 CIFAR-10 数据集（不含 DataLoader）。
    通常在 finetune_cifar10.py 中这样调用：
        train_dataset = build_dataset('train', args)
        val_dataset   = build_dataset('val', args)

    参数:
      is_train (str): 'train' 或 'val'/'test'，区分训练集与测试/验证集。
      args: 参数对象，需包含以下字段：
        - data_path (str): CIFAR-10 数据集根目录
        - input_size (int): 模型期望的输入尺寸
        - batch_size (int): DataLoader 的 batch_size
        - sigma (float): 噪声标准差
        - use_quaternion_noise (bool): 是否使用四元数小波噪声
        - levels (int): 四元数小波分解层数
        - ratio (float): 高频/低频噪声比例
    返回:
      torch.utils.data.Dataset 实例，可直接传给 DataLoader 使用。
    """
    train_flag = (is_train.lower() == 'train')
    dataset = NoisyImageDataset(
        data_root=args.data_path,
        train=train_flag,
        input_size=args.input_size,
        batch_size=args.batch_size,
        use_quaternion_noise=args.use_quaternion_noise,
        noise_sigma=args.sigma,
        levels=args.levels,
        ratio=args.ratio
    )
    return dataset


def build_dataset_with_interval(is_train: str, args):
    """
    构建带“区间划分”的 CIFAR-10 数据集，示例等价于 build_dataset，
    可根据后续需要自行实现更复杂的区间划分逻辑。
    """
    return build_dataset(is_train, args)
