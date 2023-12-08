import torch
from criteria.clip_loss import CLIPLoss, AugCLIPLoss
import face_alignment
from utils.image_utils import process_display_input
import torch.nn.functional as F
from tqdm import tqdm
class TextProxy(torch.nn.Module):
    def __init__(self, generator, seg, mean_latent_code, aug_clip_loss=True):
        super(TextProxy, self).__init__()
        self.generator = generator
        self.seg = seg
        self.clip_lambda = 1
        self.hair_mask_lambda = 1
        self.landmark_lambda = 200
        self.lr=0.01
        self.step=200
        self.visual_num = 10
        self.mask_loss = self.weighted_ce_loss()
        self.clip_loss = self.get_clip_loss(aug_clip_loss=aug_clip_loss)
        self.landmark_loss = torch.nn.MSELoss()
        self.mean_latent_code = mean_latent_code
        self.kp_extractor = self.load_kp_extractor()

    def weighted_ce_loss(self):
        weight_tmp = torch.zeros(16).cuda()
        weight_tmp[10] = 1
        weight_tmp[1] = 1
        weight_tmp[6] = 1
        weight_tmp[0] = 1
        return torch.nn.CrossEntropyLoss(weight=weight_tmp).cuda()

    def get_clip_loss(self, aug_clip_loss=True):
        if aug_clip_loss:
            return AugCLIPLoss()
        else:
            return CLIPLoss()

    def load_kp_extractor(self):
        kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda')
        for param in kp_extractor.face_alignment_net.parameters():
            param.requires_grad = False
        return kp_extractor

    def setup_optimizer(self, from_mean=True):
        if not from_mean:
            truncation_mean_latent = self.mean_latent_code[:,0]
            latent_code_init_not_trunc = torch.randn(1, 512).cuda()
            with torch.no_grad():
                _, latent_code_init_truncated = self.generator([latent_code_init_not_trunc], return_latents=True, truncation=0.3, truncation_latent=truncation_mean_latent)
            random_latent_with_trunc = latent_code_init_truncated[0][0]

        latent = []
        for i in range(18):
            if from_mean:
                tmp = self.mean_latent_code[0,0].clone().detach().cuda()
            else:
                tmp = random_latent_with_trunc.clone().detach()
            if i < 7:
                tmp.requires_grad = True
            else:
                tmp.requires_grad = False
            latent.append(tmp)
        optimizer = torch.optim.Adam(latent[0:7], lr=self.lr)
        return optimizer, latent

    def inference_on_kp_extractor(self, input_image):
        return self.kp_extractor.face_alignment_net(((F.interpolate(input_image, size=(256, 256)) + 1) / 2).clamp(0, 1))

    def forward(self, tar_description, src_image, from_mean=True, painted_mask=None):
        optimizer, latent = self.setup_optimizer(from_mean=from_mean)
        src_kp = self.inference_on_kp_extractor(src_image).clone().detach()
        visual_list = []
        visual_interval = self.step // self.visual_num
        pbar = tqdm(range(self.step))
        for i in pbar:
            latent_in = torch.stack(latent).unsqueeze(0)
            img_gen, _ = self.generator([latent_in], input_is_latent=True, randomize_noise=False)

            c_loss = self.clip_loss(img_gen, tar_description)

            gen_kp = self.inference_on_kp_extractor(img_gen)
            kp_loss = self.landmark_loss(src_kp[:, 0:17], gen_kp[:, 0:17]) + self.landmark_loss(src_kp[:, 27:36], gen_kp[:, 27:36])

            loss = self.clip_lambda * c_loss + self.landmark_lambda * kp_loss

            if painted_mask is not None:
                down_seg = self.seg(img_gen)[1]
                hair_mask_loss = self.mask_loss(down_seg, painted_mask)
                loss += self.hair_mask_lambda * hair_mask_loss
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            pbar.set_description((f"text_loss: {loss.item():.4f};"))
            if (i % visual_interval == 0) or (i == (self.step-1)):
                with torch.no_grad():
                    img_gen, _ = self.generator([latent_in], input_is_latent=True, randomize_noise=False)
                    visual_list.append(process_display_input(img_gen))
        return latent_in, visual_list