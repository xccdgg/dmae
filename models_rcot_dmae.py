import torch
import torch.nn as nn


class ResidualEncoder(nn.Module):
    """Encode residual image into an embedding."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
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

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int | None = None):
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        if cond is not None:
            x = self.film(cond, x)
        return x


class ConditionalDecoder(nn.Module):
    """Decoder with FiLM conditioned Transformer blocks."""

    def __init__(self, *, embed_dim: int = 768, num_layers: int = 8, num_heads: int = 8, mlp_ratio: float = 4.0,
                 cond_dim: int | None = None, patch_size: int = 16, image_size: int = 224):
        super().__init__()
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


class TwoStageDMAE(nn.Module):
    """DMAE with residual-conditioned second-stage decoder."""

    def __init__(self, base_model: nn.Module, decoder2: nn.Module, res_encoder: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.decoder2 = decoder2
        self.res_encoder = res_encoder

    def forward(self, x_noisy: torch.Tensor, x_clean: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z, _, ids_restore = self.base_model.forward_encoder(x_noisy, mask_ratio=0.0)
        pred1 = self.base_model.forward_decoder(z, ids_restore)
        x_hat = self.base_model.unpatchify(pred1)
        dec_tokens = self.base_model.decoder_embed(z)
        dec_tokens = dec_tokens[:, 1:, :]  # remove cls token
        if x_clean is not None:
            r = x_clean - x_hat
        else:
            r = x_noisy - x_hat
        e = self.res_encoder(r)
        x_refined = self.decoder2(dec_tokens, e)
        return x_hat, x_refined


def two_stage_dmae_vit_base(base_model: nn.Module) -> TwoStageDMAE:
    """Build a two-stage model from a pretrained DMAE base model."""
    res_encoder = ResidualEncoder(in_channels=3, embed_dim=base_model.cls_token.shape[-1])
    img_sz = getattr(base_model.patch_embed, 'img_size', 224) if hasattr(base_model, 'patch_embed') else 224
    if isinstance(img_sz, tuple):
        img_sz = img_sz[0]
    patch_size = getattr(base_model, 'patch_embed', None)
    patch_size = patch_size.patch_size[0] if patch_size is not None else 16
    decoder2 = ConditionalDecoder(
        embed_dim=base_model.decoder_embed.out_features if hasattr(base_model, 'decoder_embed') else 768,
        num_layers=getattr(base_model, 'decoder_depth', 8),
        num_heads=getattr(base_model, 'decoder_num_heads', 8),
        cond_dim=res_encoder.fc.out_features,
        patch_size=patch_size,
        image_size=img_sz,
    )
    try:
        decoder2.load_state_dict(base_model.state_dict(), strict=False)
    except Exception:
        pass
    return TwoStageDMAE(base_model, decoder2, res_encoder)
