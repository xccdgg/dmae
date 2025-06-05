import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import models_dmae
from models_rcot_dmae import two_stage_dmae_vit_base

if __name__ == "__main__":
    # load pretrained DMAE (example using base architecture)
    base_model = models_dmae.dmae_vit_base_patch16()
    base_model.eval()

    # build two-stage model
    two_stage_model = two_stage_dmae_vit_base(base_model)
    two_stage_model.eval()

    # dummy input
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        x_hat, x_refined = two_stage_model(x)
    print("Stage1 output shape:", x_hat.shape)
    print("Stage2 output shape:", x_refined.shape)
