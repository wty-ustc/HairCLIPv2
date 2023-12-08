import torch
from utils.image_utils import process_display_input
import torch.nn.functional as F
from models.sketch_proxy.encoders import backbone_encoders
from utils.sketch_ui import draw_sketch
from PIL import Image
import numpy as np
from torchvision import transforms
from argparse import Namespace
from utils.image_utils import dliate_erode
class SketchProxy(torch.nn.Module):
    def __init__(self, generator, mean_latent_code, sketch_model_path):
        super(SketchProxy, self).__init__()
        self.generator = generator
        self.mean_latent_code = mean_latent_code
        self.sketch_translation_encoder = self.load_sketch_translator(sketch_model_path)
        self.pre_process = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])

    def get_keys(self, d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
        return d_filt

    def load_sketch_translator(self, sketch_model_path):
        sketch_ckpt = torch.load(sketch_model_path, map_location='cpu')
        sketch_opts = sketch_ckpt['opts']
        sketch_opts = Namespace(**sketch_opts)

        sketch_translation_encoder = backbone_encoders.BackboneEncoderFirstStage(50, 'ir_se', sketch_opts)
        sketch_translation_encoder.load_state_dict(self.get_keys(sketch_ckpt, 'encoder_firststage'), strict=True)
        sketch_translation_encoder.eval()
        sketch_translation_encoder.cuda()
        return sketch_translation_encoder   

    def forward(self, visual_mask):
        visual_list = []

        sketch_by_user, _ = draw_sketch(visual_mask)
        sketch_for_visual = sketch_by_user.copy()
        sketch_for_visual = np.stack([sketch_for_visual,sketch_for_visual,sketch_for_visual], axis=2)

        sketch_mask = sketch_for_visual.copy()
        sketch_mask[sketch_mask>0] = 1
        local_blending_mask = dliate_erode(sketch_mask, 30)

        img_sketch = Image.fromarray(sketch_by_user)
        img_sketch = self.pre_process(img_sketch).unsqueeze(0).cuda().float()

        with torch.no_grad():
            latent_infer = self.sketch_translation_encoder(img_sketch) + self.mean_latent_code
            sketch_target_img, _ = self.generator([latent_infer], input_is_latent=True, randomize_noise=False)

        visual_list.append(process_display_input(sketch_for_visual))
        visual_list.append(process_display_input(255*local_blending_mask))
        visual_list.append(process_display_input(sketch_target_img))
        return latent_infer, local_blending_mask, visual_list