import torch
from utils.image_utils import process_display_input
import torch.nn.functional as F
from models.bald_proxy.networks.level_mapper import LevelMapper

class BaldProxy(torch.nn.Module):
    def __init__(self, generator, bald_model_path):
        super(BaldProxy, self).__init__()
        self.generator = generator
        self.mapper_for_bald, self.alpha = self.load_bald_model(bald_model_path)

    def load_bald_model(self, bald_model_path):
        bald_ckpt = torch.load(bald_model_path)
        alpha = float(bald_ckpt['alpha']) * 1.2
        mapper_for_bald = LevelMapper(input_dim=512).eval().cuda()
        mapper_for_bald.load_state_dict(bald_ckpt['state_dict'], strict=True)
        return mapper_for_bald, alpha

    def forward(self, src_latent):
        visual_list = []

        mapper_input_tensor = src_latent.clone().detach()
        latent_infer = src_latent.clone().detach()
        with torch.no_grad():
            latent_infer[:, :8, :] += self.alpha * self.mapper_for_bald(mapper_input_tensor)
            bald_target_img, _ = self.generator([latent_infer], input_is_latent=True, return_latents=False)
            inv_source_img, _ = self.generator([src_latent], input_is_latent=True, return_latents=False)

        visual_list.append(process_display_input(inv_source_img))
        visual_list.append(process_display_input(bald_target_img))
        return latent_infer, visual_list