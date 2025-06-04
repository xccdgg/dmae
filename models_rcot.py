import torch
import torch.nn as nn

class ResidualEncoder(nn.Module):
    """Encode residual image into a vector."""
    def __init__(self, in_channels=3, embed_dim=768):
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
    """Feature-wise Linear Modulation."""
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
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int = None):
        super().__init__()
        self.embed_dim = embed_dim
        cond_dim = embed_dim if cond_dim is None else cond_dim
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        if cond is not None:
            x = self.film(cond, x)
        return x

class ConditionalDecoder(nn.Module):
    """Transformer decoder conditioned on residual embedding."""
    def __init__(self, embed_dim=768, num_layers=8, num_heads=8, mlp_ratio=4.0, cond_dim=None, patch_size=16, image_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(embed_dim, num_heads, mlp_ratio, cond_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.out_proj = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, latent_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = latent_tokens + self.pos_embed[:, :latent_tokens.size(1), :]
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


class DMAEEncoderWrapper(nn.Module):
    """Wrap DMAE model's encoder for use in two-stage architecture."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.base.forward_encoder(x, mask_ratio=0.0)
        return latent


class DMAEDecoderWrapper(nn.Module):
    """Wrap DMAE model's decoder to output reconstructed image without masking."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        ids_restore = torch.arange(latent.size(1), device=latent.device)
        ids_restore = ids_restore.unsqueeze(0).repeat(latent.size(0), 1)
        pred = self.base.forward_decoder(latent, ids_restore)
        img = self.base.unpatchify(pred)
        return img

class TwoStageDMAE(nn.Module):
    """Combine encoder, first decoder, residual encoder and conditional decoder."""
    def __init__(self, encoder: nn.Module, decoder1: nn.Module, decoder2: nn.Module, res_encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.res_encoder = res_encoder

    def forward(self, x_noisy: torch.Tensor, x_clean: torch.Tensor = None):
        z = self.encoder(x_noisy)
        x_hat = self.decoder1(z)
        if x_clean is not None:
            r = x_clean - x_hat
        else:
            r = x_noisy - x_hat
        e = self.res_encoder(r)
        x_refined = self.decoder2(z, e)
        return x_hat, x_refined

