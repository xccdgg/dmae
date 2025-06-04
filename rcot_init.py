import torch
from models_dmae import dmae_vit_base_patch16_dec512d8b
from models_rcot import (
    ResidualEncoder,
    ConditionalDecoder,
    TwoStageDMAE,
    DMAEEncoderWrapper,
    DMAEDecoderWrapper,
)


def build_two_stage_model(checkpoint_path=None):
    base = dmae_vit_base_patch16_dec512d8b()
    encoder = DMAEEncoderWrapper(base)
    decoder1 = DMAEDecoderWrapper(base)
    res_encoder = ResidualEncoder(in_channels=3, embed_dim=768)
    decoder2 = ConditionalDecoder(
        embed_dim=768,
        num_layers=8,
        num_heads=16,
        cond_dim=768,
        patch_size=16,
        image_size=224,
    )
    decoder2.load_state_dict(base.state_dict(), strict=False)
    model = TwoStageDMAE(encoder, decoder1, decoder2, res_encoder)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return model

if __name__ == "__main__":
    model = build_two_stage_model()
    x = torch.randn(1, 3, 224, 224)
    x_hat, x_refined = model(x)
    print(x_hat.shape, x_refined.shape)
