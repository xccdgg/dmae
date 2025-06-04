# util/quadatasetgpu.py

import torch
import torch.nn.functional as F
import math

class QuaternionWavelet:
    """
    正交四元数离散小波（Haar）实现。
    支持彩色(3通道)与灰度(1通道)图像的 DWT/IDWT。
    图像映射到四元数域 (r,i,j,k)，使用可分离 2D Haar 小波。
    """

    def __init__(self, filter_name='haar', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self._init_filters(filter_name)

    def _init_filters(self, filter_name):
        if filter_name != 'haar':
            raise NotImplementedError(f'Filter {filter_name} 未实现')
        sqrt2 = math.sqrt(2.0)
        # 分解滤波器
        self.h = torch.tensor([1/sqrt2, 1/sqrt2], dtype=torch.float32, device=self.device)
        self.g = torch.tensor([1/sqrt2, -1/sqrt2], dtype=torch.float32, device=self.device)
        # 重构滤波器是分解滤波器的翻转
        self.hr = self.h.flip(0)
        self.gr = self.g.flip(0)

    def _conv2d_sep(self, t, fr, fc):
        """
        可分离 2D 卷积: 先对行方向 fr 下采样卷积，再对列方向 fc 下采样卷积
        t: [N, H, W]
        fr: [K]  (行滤波器)
        fc: [K]  (列滤波器)
        返回: [N, H//2, W//2]
        """
        # 先做行方向的下采样卷积
        pad_r = fr.numel() // 2
        # 在行方向 padding pad_r，列方向不 padding
        out = F.conv2d(
            t.unsqueeze(1),                   # [N,1,H,W]
            fr.view(1,1,-1,1),                # [1,1,Kh,1]
            stride=(2,1),
            padding=(pad_r, 0)
        ).squeeze(1)  # [N,H//2, W]
        # 再做列方向的下采样卷积
        pad_c = fc.numel() // 2
        out = F.conv2d(
            out.unsqueeze(1),                 # [N,1,H//2, W]
            fc.view(1,1,1,-1),                # [1,1,1,Kw]
            stride=(1,2),
            padding=(0, pad_c)
        ).squeeze(1)  # [N, H//2, W//2]
        return out

    def _up_conv2d_sep(self, t, fr, fc):
        """
        可分离 2D 逆卷积: 先对行方向 fr 上采样逆卷积，再对列方向 fc 上采样逆卷积
        t: [N, H, W]
        fr: [K]  (行逆卷积核)
        fc: [K]  (列逆卷积核)
        返回: [N, H*2, W*2]
        """
        # 行方向上采样逆卷积
        pad_r = fr.numel() // 2
        out = F.conv_transpose2d(
            t.unsqueeze(1),                   # [N,1,H,W]
            fr.view(1,1,-1,1),                # [1,1,Kh,1]
            stride=(2,1),
            padding=(pad_r, 0)
        ).squeeze(1)  # [N, H*2, W]
        # 列方向上采样逆卷积
        pad_c = fc.numel() // 2
        out = F.conv_transpose2d(
            out.unsqueeze(1),                 # [N,1, H*2, W]
            fc.view(1,1,1,-1),                # [1,1,1,Kw]
            stride=(1,2),
            padding=(0, pad_c)
        ).squeeze(1)  # [N, H*2, W*2]
        return out

    def decompose(self, images, levels=1, sigma_low=0.0, sigma_high=0.0):
        """
        对输入图像进行四元数小波分解，并返回各子带系数。
        images: Tensor, 形状可为 [H, W] 或 [N, H, W, C]
                - 灰度图时 C=1
                - 彩色图时 C=3
        levels: 分解层数
        sigma_low, sigma_high: 本函数只负责分解，不实际使用 sigma；噪声注入在外部完成
        返回: coeffs 字典，包含：
            - 'LL': [N, H/2^levels, W/2^levels, 4]  低频系数（四元数表示）
            - 'subbands': 列表长度为 levels，每层包含 (LH, HL, HH)，每个子带形状 [N, H/2^i, W/2^i, 4]
        """
        # 判断输入是否是灰度单通道 H,W
        is_gray = (images.ndim == 2)
        if is_gray:
            images = images.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]

        N, H, W, C = images.shape
        # 转为四元数表示的通道：四元数有 4 个分量 (r,i,j,k)，彩色图用 (0,R,G,B) 为 i,j,k 分量
        x = images.to(self.device).permute(0,3,1,2).float()  # [N, C, H, W]
        if C == 3:
            zeros = torch.zeros(N, 1, H, W, device=self.device)
            x = torch.cat([zeros, x], dim=1)  # [N,4,H,W]
        else:
            zeros = torch.zeros_like(x)
            x = torch.cat([x, zeros, zeros, zeros], dim=1)  # [N,4,H,W]

        coeffs = {'subbands': []}
        current = x  # 每次迭代保持 [N,4,H_i,W_i]

        for _ in range(levels):
            a, b, c1, d = current[:,0], current[:,1], current[:,2], current[:,3]
            # 低频 LL
            al = self._conv2d_sep(a, self.h, self.h)
            bl = self._conv2d_sep(b, self.h, self.h)
            cl = self._conv2d_sep(c1, self.h, self.h)
            dl = self._conv2d_sep(d, self.h, self.h)
            LL = torch.stack([al, bl, cl, dl], dim=-1)  # [N, H/2, W/2, 4]

            # 高频 LH
            ah = self._conv2d_sep(a, self.h, self.g)
            bh = self._conv2d_sep(b, self.h, self.g)
            ch = self._conv2d_sep(c1, self.h, self.g)
            dh = self._conv2d_sep(d, self.h, self.g)
            LH = torch.stack([ah, bh, ch, dh], dim=-1)  # [N, H/2, W/2, 4]

            # 高频 HL
            al_ = self._conv2d_sep(a, self.g, self.h)
            bl_ = self._conv2d_sep(b, self.g, self.h)
            cl_ = self._conv2d_sep(c1, self.g, self.h)
            dl_ = self._conv2d_sep(d, self.g, self.h)
            HL = torch.stack([al_, bl_, cl_, dl_], dim=-1)  # [N, H/2, W/2, 4]

            # 高频 HH
            ah_ = self._conv2d_sep(a, self.g, self.g)
            bh_ = self._conv2d_sep(b, self.g, self.g)
            ch_ = self._conv2d_sep(c1, self.g, self.g)
            dh_ = self._conv2d_sep(d, self.g, self.g)
            HH = torch.stack([ah_, bh_, ch_, dh_], dim=-1)  # [N, H/2, W/2, 4]

            coeffs['LL'] = LL
            coeffs['subbands'].append((LH, HL, HH))
            # 下一层在 LL 上继续分解：需要将 LL [N, H/2, W/2, 4] 转为 [N,4,H/2,W/2]
            current = LL.permute(0, 3, 1, 2)  # [N,4,H/2,W/2]

        return coeffs

    def reconstruct(self, coeffs):
        """
        使用一层重构将小波系数重建到图像空间。
        仅支持一层：输入 coeffs 需含 'LL' 和 'subbands' 里的( LH,HL,HH ) 三个子带。
        coeffs['LL']: [N, H, W, 4]
        coeffs['subbands'][0]: LH, HL, HH，每个形状 [N, H, W, 4]
        返回: 重构后的图像张量，形状 [N, H*2, W*2, C]，其中 C=3（RGB）或 C=1（灰度）
        """
        LL = coeffs['LL']          # [N, H, W, 4]
        LH, HL, HH = coeffs['subbands'][0]  # 各为 [N, H, W, 4]
        print("1")

        comps = []
        for d in range(4):
            ll = LL[..., d]    # [N, H, W]
            lh = LH[..., d]
            hl = HL[..., d]
            hh = HH[..., d]
            rec = (
                self._up_conv2d_sep(ll, self.hr, self.hr) +
                self._up_conv2d_sep(lh, self.hr, self.gr) +
                self._up_conv2d_sep(hl, self.gr, self.hr) +
                self._up_conv2d_sep(hh, self.gr, self.gr)
            )  # [N, H*2, W*2]
            comps.append(rec)

        rec = torch.stack(comps, dim=-1)  # [N, H*2, W*2, 4]
        # 丢弃实部 (第 0 分量)，保留 i,j,k 三个分量作为 RGB 或 灰度
        img = rec[..., 1:4]  # [N, H*2, W*2, 3]
        if img.shape[-1] == 1:
            img = img[..., 0]  # 灰度情况 [N, H*2, W*2]

        return img.clamp(0.0, 1.0)

class QuaternionWaveletNoise:
    """
    四元数小波噪声模块。
    对输入图像张量在四元数小波系数域分别对低频子带和各高频子带加高斯噪声，
    满足随机平滑所需的加性独立同分布噪声条件。
    """

    def __init__(self, sigma, device=None, filter_name='haar', levels=1, ratio=3.0):
        """
        参数:
          sigma (float): 总体噪声标准差。
          device (str 或 torch.device): 运算设备。
          filter_name (str): 小波基名称，当前仅支持 'haar'。
          levels (int): 小波分解层数。
          ratio (float): 高频与低频方差比例，高频 σ_high = σ * (ratio/(1+ratio))，低频 σ_low = σ/(1+ratio)。
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.sigma_low  = sigma / (1.0 + ratio)
        self.sigma_high = sigma * (ratio / (1.0 + ratio))
        self.levels = levels
        self.qwt = QuaternionWavelet(filter_name=filter_name, device=self.device)

    def _apply_quat_add_noise(self, coeffs):
        """
        对小波系数添加四元数高斯噪声：对每个通道分量添加 N(0, (σ_low/2)^2 ) 或 N(0, (σ_high/2)^2)。
        coeffs: 字典，含 'LL': [N,H,W,4] 和 'subbands': [(LH,HL,HH), ...]
        返回: 添加噪声后的 coeffs 字典（就地修改）。
        """
        def add_noise_to_Q(Q, sigma):
            # Q 可以是形状 [N, H, W, 4] 或 [N, H, W]
            return Q + torch.randn_like(Q, device=self.device) * (sigma / 2.0)

        # 对低频子带 LL 加噪
        coeffs['LL'] = add_noise_to_Q(coeffs['LL'], self.sigma_low)

        # 对每一层的高频子带加噪
        new_subbands = []
        for (LH, HL, HH) in coeffs['subbands']:
            LHn = add_noise_to_Q(LH, self.sigma_high)
            HLn = add_noise_to_Q(HL, self.sigma_high)
            HHn = add_noise_to_Q(HH, self.sigma_high)
            new_subbands.append((LHn, HLn, HHn))
        coeffs['subbands'] = new_subbands

        return coeffs

    @staticmethod
    def apply_noise(x, sigma, filter_name='haar', levels=1, ratio=3.0, device=None):
        """
        对输入 x 添加四元数小波噪声并返回加噪结果。
        x: Tensor, 形状可以是 [C, H, W] 或 [N, C, H, W]。
        sigma, filter_name, levels, ratio, device: 与构造函数参数一致。
        返回: 加噪后的张量，形状与输入相同，dtype 与输入相同。
        """
        # 确保 x 是张量
        if not torch.is_tensor(x):
            raise TypeError(f"apply_noise 要求输入为 torch.Tensor, 得到: {type(x)}")
        single = (x.dim() == 3)  # 单张图像情况
        if single:
            x = x.unsqueeze(0)  # [1, C, H, W]

        N, C, H, W = x.shape
        # 初始化工具对象
        qwn = QuaternionWaveletNoise(sigma=sigma, device=device, filter_name=filter_name, levels=levels, ratio=ratio)

        x = x.to(qwn.device)
        orig_dtype = x.dtype

        # 仅支持 1 通道或 3 通道
        if C not in (1, 3):
            raise ValueError("QuaternionWaveletNoise 仅支持 1 或 3 通道图像，当前通道数: {}".format(C))

        # 将 x 转换为 NHWC 或 HWC 形式
        x_float = x.float()
        if C == 3:
            imgs = x_float.permute(0, 2, 3, 1)  # [N, H, W, 3]
        else:  # C == 1
            imgs = x_float[:, 0, :, :].unsqueeze(-1)  # [N, H, W, 1]

        # 小波分解
        coeffs = qwn.qwt.decompose(imgs, levels=levels, sigma_low=0.0, sigma_high=0.0)
        # 添加噪声
        coeffs = qwn._apply_quat_add_noise(coeffs)
        # 小波重构
        rec = qwn.qwt.reconstruct(coeffs)  # [N, H, W, C] 或 [N, H, W] (灰度)

        # 将重构结果转回 NCHW 或 CHW
        if rec.dim() == 4:  # 彩色 [N, H, W, 3]
            out = rec.permute(0, 3, 1, 2)  # [N, 3, H, W]
        else:  # 灰度 [N, H, W]
            out = rec.unsqueeze(3).permute(0, 3, 1, 2)  # [N, 1, H, W]

        out = out.clamp(0.0, 1.0).to(orig_dtype)

        if single:
            out = out.squeeze(0)  # 返回 [C, H, W]

        return out
