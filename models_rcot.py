import torch
import torch.nn as nn
from typing import Optional


class ResidualEncoder(nn.Module):
    """Encode residual images into a conditioning embedding."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embed_dim)
        self.act = nn.GELU()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(r)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).view(x.size(0), -1)
        e = self.act(self.fc(x))
        return e


class FiLMBlock(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feat_dim)
        self.beta_fc = nn.Linear(cond_dim, feat_dim)

    def forward(self, cond: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond)
        beta = self.beta_fc(cond)
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return features * gamma + beta


class ConditionalTransformerBlock(nn.Module):
    """Transformer block with FiLM conditioning."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: Optional[int] = None):
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.film = FiLMBlock(cond_dim, embed_dim)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        if cond is not None:
            x = self.film(cond, x)
        return x


class ConditionalDecoder(nn.Module):
    """Transformer decoder that uses FiLM conditioning from residual embedding."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        cond_dim: Optional[int] = None,
        patch_size: int = 16,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(embed_dim, num_heads, mlp_ratio, cond_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, latent_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = latent_tokens + self.pos_embed[:, : latent_tokens.size(1), :]
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        patch_pixels = self.out_proj(x)
        B, N, _ = patch_pixels.shape
        patches_per_side = int(N ** 0.5)
        out = patch_pixels.view(B, patches_per_side, patches_per_side, self.patch_size, self.patch_size, 3)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out_image = out.view(B, 3, patches_per_side * self.patch_size, patches_per_side * self.patch_size)
        return out_image


class TwoStageDMAE(nn.Module):
    """Wrap DMAE with residual conditioned second stage decoder."""

    def __init__(self, base_model: nn.Module, decoder2: ConditionalDecoder, res_encoder: ResidualEncoder) -> None:
        super().__init__()
        self.base = base_model
        self.decoder2 = decoder2
        self.res_encoder = res_encoder
        self.mean = self.base.mean
        self.std = self.base.std

    def forward(self, x_noisy: torch.Tensor, x_clean: Optional[torch.Tensor] = None):
        if self.mean.device != x_noisy.device:
            self.mean = self.mean.to(x_noisy.device)
            self.std = self.std.to(x_noisy.device)
        x_noisy = (x_noisy - self.mean) / self.std
        if x_clean is not None:
            x_clean = (x_clean - self.mean) / self.std
        latent, _, ids_restore = self.base.forward_encoder(x_noisy, mask_ratio=0.0)
        pred = self.base.forward_decoder(latent, ids_restore)
        x_hat = self.base.unpatchify(pred)
        residual = (x_clean if x_clean is not None else x_noisy) - x_hat
        e = self.res_encoder(residual)
        refined = self.decoder2(latent[:, 1:, :], e)
        x_hat = x_hat * self.std + self.mean
        refined = refined * self.std + self.mean
        return x_hat, refined
