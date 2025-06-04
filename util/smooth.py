# util/smooth.py

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import beta, norm

# 直接从 util/quadatasetgpu.py 中导入 QuaternionWaveletNoise
from util.quadatasetgpu import QuaternionWaveletNoise


class Smooth:
    """
    随机平滑分类器：通过四元数小波噪声（QuaternionWaveletNoise）对输入进行多次采样，
    实现鲁棒预测和认证半径估计。

    方法：
      - predict(x, n, batch_size): 对单张输入 x 进行 n 次噪声采样，返回多数投票类别或 ABSTAIN。
      - certify(x, n0, n, alpha, batch_size, y=None): 对单张输入 x 进行随机平滑认证，
         先用 n0 次采样确定预测类别，再用 n 次采样估计置信下界并计算 L2 认证半径。
    """

    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int,
                 sigma: float, levels: int = 1, ratio: float = 3.0, device: str = None):
        """
        参数：
          base_classifier: 已训练好的分类模型 (输出 logits)。
          num_classes (int): 分类类别总数。
          sigma (float): 噪声标准差，用于 QuaternionWaveletNoise。
          levels (int): 四元数小波分解层数。
          ratio (float): 高频/低频噪声比例。
          device (str 或 torch.device): 推理设备 (如 'cuda' 或 'cpu')，若为 None 则自动从 base_classifier 获取。
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.levels = levels
        self.ratio = ratio

        # 冻结模型，设为评估模式
        self.base_classifier.eval()
        # 确定运行设备
        if device is None:
            try:
                self.device = next(base_classifier.parameters()).device
            except Exception:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # 获取模型期望的输入尺寸（假设基于 timm ViT，有 patch_embed.img_size 属性）
        try:
            img_size = self.base_classifier.patch_embed.img_size
            if isinstance(img_size, (tuple, list)):
                self.input_size = img_size[0]
            else:
                self.input_size = img_size
        except Exception:
            # 若无法获取，则假设输入应为 224
            self.input_size = 224

    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        确保输入 x 为形状 [C, H, W] 的单张图像张量，
        并将其移动到 self.device；如果传入 [1, C, H, W]，则先 squeeze。
        否则抛出异常。
        """
        if not torch.is_tensor(x):
            raise TypeError(f"输入必须是 torch.Tensor, 当前类型: {type(x)}")
        if x.dim() == 3:
            return x.to(self.device)
        elif x.dim() == 4 and x.shape[0] == 1:
            return x.squeeze(0).to(self.device)
        else:
            raise ValueError(f"输入张量形状应为 [C,H,W] 或 [1,C,H, W], 当前形状: {x.shape}")

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """
        将单张图像 x ([C,H,W]) 调整到模型期望的输入大小 [C, input_size, input_size]。
        """
        x = x.unsqueeze(0)  # [1, C, H, W]
        x_resized = F.interpolate(x, size=(self.input_size, self.input_size),
                                  mode='bilinear', align_corners=False)
        return x_resized.squeeze(0)  # [C, input_size, input_size]

    def predict(self, x: torch.Tensor, n: int, batch_size: int) -> int:
        """
        平滑分类预测：
          对输入 x 进行 n 次四元数小波噪声采样，统计基础分类器的投票结果，
          返回投票最多的类别索引；如果出现并列最高投票，则返回 ABSTAIN。
        参数：
          x (torch.Tensor): 单张输入图像，形状 [C, H, W] 或 [1, C, H, W]，值在 [0,1]。
          n (int): 噪声采样次数。
          batch_size (int): 每次前向推理的子 batch 大小，便于并行计算。
        返回：
          int: 预测类别索引（0 ~ num_classes-1），或 ABSTAIN (-1)。
        """
        # 确保 x 是正确形状，并移到指定设备
        x_img = self._ensure_tensor(x)  # 形状 [C, H, W]

        # 调整到模型期望的输入大小
        x_img = self._resize(x_img)  # [C, input_size, input_size]

        # 生成 n 个加噪样本：先将 [C, input_size, input_size] 扩展为 [n, C, input_size, input_size]
        x_batch = x_img.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, C, input_size, input_size]
        # 对整个 batch 应用四元数小波噪声
        noised_batch = QuaternionWaveletNoise.apply_noise(
            x_batch,
            sigma=self.sigma,
            filter_name='haar',
            levels=self.levels,
            ratio=self.ratio,
            device=self.device
        )  # 结果形状 [n, C, input_size, input_size]

        # 分批进行推理，将预测结果累积到 counts 中
        counts = np.zeros(self.num_classes, dtype=int)
        noised_batch = noised_batch.to(self.device)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = noised_batch[start:end]  # [batch_size, C, input_size, input_size]
                logits = self.base_classifier(batch)  # [batch_size, num_classes]
                preds = logits.argmax(dim=1).cpu().numpy()  # [batch_size]
                for p in preds:
                    counts[p] += 1

        # 找到投票最多的类别
        max_votes = counts.max()
        top_classes = np.nonzero(counts == max_votes)[0]
        if len(top_classes) != 1:
            return Smooth.ABSTAIN
        else:
            return int(top_classes[0])

    def certify(self, x: torch.Tensor, n0: int, n: int, alpha: float, batch_size: int, y=None) -> (int, float):
        """
        随机平滑认证：
          1. 使用 n0 次噪声采样确定临时预测类别 cAhat。
          2. 使用 n 次噪声采样统计该类别的出现次数 countA。
          3. 计算 Clopper-Pearson 置信下界 p_lower。
          4. 如果 p_lower < 0.5，则返回 (ABSTAIN, 0.0)；否则返回 (cAhat, sigma * Φ^{-1}(p_lower))。
        参数：
          x (torch.Tensor): 单张输入图像，形状 [C,H,W] 或 [1,C,H,W]，值在 [0,1]。
          n0 (int): 初始采样次数，用于确定 cAhat。
          n  (int): 后续采样次数，用于估计置信下界。
          alpha (float): 置信水平（如 0.001 对应 99.9%）。
          batch_size (int): 子 batch 大小，用于分类推理。
          y: 真实标签，仅用于兼容调用，不做内部计算。
        返回：
          (预测类别 int, 认证半径 float); 如果无法认证则返回 (ABSTAIN, 0.0)。
        """
        # 确保 x 形状正确并移到设备
        x_img = self._ensure_tensor(x)  # [C, H, W]

        # 调整到模型期望的输入大小
        x_img = self._resize(x_img)  # [C, input_size, input_size]

        # -------- 步骤 1: n0 次采样确定 cAhat --------
        x_batch0 = x_img.unsqueeze(0).repeat(n0, 1, 1, 1)  # [n0, C, input_size, input_size]
        noised_batch0 = QuaternionWaveletNoise.apply_noise(
            x_batch0,
            sigma=self.sigma,
            filter_name='haar',
            levels=self.levels,
            ratio=self.ratio,
            device=self.device
        )  # [n0, C, input_size, input_size]
        print("quanoise")
        counts0 = np.zeros(self.num_classes, dtype=int)
        noised_batch0 = noised_batch0.to(self.device)
        with torch.no_grad():
            for start in range(0, n0, batch_size):
                end = min(start + batch_size, n0)
                batch0 = noised_batch0[start:end]  # [batch_size, C, input_size, input_size]
                logits0 = self.base_classifier(batch0)  # [batch_size, num_classes]
                preds0 = logits0.argmax(dim=1).cpu().numpy()  # [batch_size]
                for p in preds0:
                    counts0[p] += 1

        cAhat = int(np.argmax(counts0))
        if counts0[cAhat] == 0:
            return Smooth.ABSTAIN, 0.0

        # -------- 步骤 2: n 次采样估计置信下界 --------
        x_batch = x_img.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, C, input_size, input_size]
        noised_batch = QuaternionWaveletNoise.apply_noise(
            x_batch,
            sigma=self.sigma,
            filter_name='haar',
            levels=self.levels,
            ratio=self.ratio,
            device=self.device
        )  # [n, C, input_size, input_size]
        countA = 0
        noised_batch = noised_batch.to(self.device)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batchN = noised_batch[start:end]  # [batch_size, C, input_size, input_size]
                logitsN = self.base_classifier(batchN)  # [batch_size, num_classes]
                predsN = logitsN.argmax(dim=1).cpu().numpy()  # [batch_size]
                countA += int((predsN == cAhat).sum())

        if countA == 0:
            p_lower = 0.0
        else:
            p_lower = beta.ppf(alpha / 2, countA, n - countA + 1)

        if p_lower < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(p_lower)
            return cAhat, float(radius)
