import torch
from criteria.clip_loss import CLIPLoss, AugCLIPLoss
from criteria import bg_loss, average_lab_color_loss
from utils.image_utils import process_display_input
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from criteria import lpips
from scripts.feature_blending import color_feature_blending
from torchvision import transforms

class ColorProxy(torch.nn.Module):
    def __init__(self, generator, seg, aug_clip_loss=False):
        super(ColorProxy, self).__init__()
        self.clip_lambda = 1
        self.bg_lambda = 10
        self.avg_color_lambda = 0.01

        self.hair_mse_lambda = 500
        self.not_hair_mse_lambda = 10
        self.hair_lpips_lambda = 1

        self.lr=0.01
        self.step=100
        self.visual_num = 10

        self.clip_loss = self.get_clip_loss(aug_clip_loss=aug_clip_loss)
        self.average_color_loss = average_lab_color_loss.AvgLabLoss(seg).cuda().eval()
        self.background_loss = bg_loss.BackgroundLoss(seg).cuda().eval()
        self.mse_loss = torch.nn.MSELoss()
        self.lpips_loss = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True).eval()

        self.generator = generator
        self.seg = seg


    def get_clip_loss(self, aug_clip_loss=False):
        if aug_clip_loss:
            return AugCLIPLoss()
        else:
            return CLIPLoss()

    def get_color_edit_mode(self, input_data):
        if isinstance(input_data, str):
            if input_data.endswith('.jpg') or input_data.endswith('.png'):
                return 'ref'
            else:
                return 'text'
        elif isinstance(input_data, tuple) and len(input_data) == 3:
            return 'rgb'
        else:
            raise ValueError('Invalid input. Unsupported data type or format.')

    def pre_process_edit_cond(self, editing_mode, color_cond):
        color_image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if editing_mode == 'text':
            return color_cond
        elif editing_mode == 'ref':
            color_ref_img = Image.open(f'test_images/ref_img/{color_cond}')
            color_cond = color_image_transform(color_ref_img).unsqueeze(0).cuda()
            self.avg_color_lambda = 0.01
        elif editing_mode == 'rgb':
            given_rgb_np = np.reshape(np.array(color_cond), (1,1,-1))
            color_cond = color_image_transform(Image.fromarray(np.uint8(given_rgb_np))).unsqueeze(0).cuda()
            self.avg_color_lambda = 0.003
        return color_cond


    def setup_color_optimizer(self, src_latent, src_feature):
        color_latent = []
        for i in range(18):
            tmp = src_latent[0][i].clone().detach().cuda()
            if (i>=9) and (i < 13):
                tmp.requires_grad = True
            else:
                tmp.requires_grad = False
            color_latent.append(tmp)
        color_optimizer = torch.optim.Adam(color_latent[9:13], lr=self.lr)
        latent_F = src_feature.clone().detach().requires_grad_(False)
        return color_optimizer, color_latent, latent_F

    def setup_final_optimizer(self, src_latent, color_feature):
        final_latent = []
        for i in range(18):
            tmp = src_latent[0][i].clone().detach().cuda()
            if (i>=11) and (i < 18):
                tmp.requires_grad = True
            else:
                tmp.requires_grad = False
            final_latent.append(tmp)
        latent_F_final = color_feature.clone().detach().requires_grad_(True)
        final_optimizer = torch.optim.Adam(final_latent[11:18] + [latent_F_final], lr=self.lr)
        return final_optimizer, final_latent, latent_F_final



    def forward(self, color_cond, edited_hairstyle_img, src_latent, src_feature):
        editing_mode = self.get_color_edit_mode(color_cond)
        color_cond = self.pre_process_edit_cond(editing_mode, color_cond)
        color_optimizer, color_latent, latent_F = self.setup_color_optimizer(src_latent, src_feature)
        visual_interval = self.step // self.visual_num
        visual_color_list = []

        pbar = tqdm(range((self.step)))
        for i in pbar:
            color_latent_in = torch.stack(color_latent).unsqueeze(0)
            color_img_gen, _ = self.generator([color_latent_in], input_is_latent=True, randomize_noise=False, start_layer=4, end_layer=8, layer_in=latent_F)

            bg_loss = self.background_loss(edited_hairstyle_img.detach(), color_img_gen)
            color_loss = self.bg_lambda * bg_loss
            if editing_mode == 'text':
                c_loss = self.clip_loss(color_img_gen, color_cond)
                color_loss += self.clip_lambda * c_loss
            else:
                avg_color_loss = self.average_color_loss(color_cond, color_img_gen)
                color_loss += self.avg_color_lambda * avg_color_loss

            color_optimizer.zero_grad()
            color_loss.backward(retain_graph=True)
            color_optimizer.step()
            pbar.set_description((f"color_loss: {color_loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.step-1)):
                with torch.no_grad():
                    color_img_gen, _ = self.generator([color_latent_in], input_is_latent=True, randomize_noise=False, start_layer=4, end_layer=8, layer_in=latent_F)
                    visual_color_list.append(process_display_input(color_img_gen))

        color_feature, final_hair_mask = color_feature_blending(self.generator, self.seg, edited_hairstyle_img, src_latent, color_latent_in, latent_F)
        final_optimizer, final_latent, latent_F_final = self.setup_final_optimizer(src_latent, color_feature)
        visual_final_list = []

        pbar = tqdm(range((self.step)))
        for i in pbar:
            final_latent_in = torch.stack(final_latent).unsqueeze(0)
            final_img, _ = self.generator([final_latent_in], input_is_latent=True, randomize_noise=False, start_layer=7, end_layer=8, layer_in=latent_F_final)

            hair_mse_loss = self.mse_loss(final_img * final_hair_mask, (color_img_gen * final_hair_mask).detach())
            not_hair_mse_loss = self.mse_loss(final_img * (1-final_hair_mask), (edited_hairstyle_img* (1-final_hair_mask)).detach())
            hair_lpips_loss = self.lpips_loss(final_img * final_hair_mask, (color_img_gen * final_hair_mask).detach()).sum()
            final_loss = self.hair_mse_lambda * hair_mse_loss + self.not_hair_mse_lambda * not_hair_mse_loss + self.hair_lpips_lambda * hair_lpips_loss
            
            final_optimizer.zero_grad()
            final_loss.backward()
            final_optimizer.step()
            pbar.set_description((f"final_loss: {final_loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.step-1)):
                with torch.no_grad():
                    final_img, _ = self.generator([final_latent_in], input_is_latent=True, randomize_noise=False, start_layer=7, end_layer=8, layer_in=latent_F_final)
                    visual_final_list.append(process_display_input(final_img))

        return visual_color_list, visual_final_list