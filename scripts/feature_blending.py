import torch
from utils.image_utils import dliate_erode
import torch.nn.functional as F

def hairstyle_feature_blending(generator, seg, src_latent, src_feature, visual_mask, latent_bald, latent_global=None, latent_local=None, local_blending_mask=None):

    if latent_global is not None:
        bald_feature, _ = generator([latent_bald], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
        global_feature, _ = generator([latent_global], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
        global_proxy, _ = generator([latent_global], input_is_latent=True, randomize_noise=False)
        global_proxy_seg = torch.argmax(seg(global_proxy)[1], dim=1).unsqueeze(1).long()

        ear_mask = torch.where(visual_mask==6, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_mask = torch.where(visual_mask==10, torch.ones_like(visual_mask), torch.zeros_like(visual_mask))[0].cpu().numpy()
        hair_ear_mask = ear_mask + hair_mask
        bald_blending_mask = dliate_erode(hair_ear_mask.astype('uint8'), 30)
        bald_blending_mask = torch.from_numpy(bald_blending_mask).unsqueeze(0).unsqueeze(0).cuda()
        bald_blending_mask_down = F.interpolate(bald_blending_mask.float(), size=(32, 32), mode='bicubic')
        src_feature = bald_feature * bald_blending_mask_down + src_feature * (1-bald_blending_mask_down)

        global_hair_mask = torch.where(global_proxy_seg==10, torch.ones_like(global_proxy_seg), torch.zeros_like(global_proxy_seg))
        global_hair_mask_down = F.interpolate(global_hair_mask.float(), size=(32, 32), mode='bicubic')
        src_feature = global_feature * global_hair_mask_down + src_feature * (1-global_hair_mask_down)

    if latent_local is not None:
        local_feature, _ = generator([latent_local], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
        local_blending_mask = torch.from_numpy(local_blending_mask[:,:,0]).unsqueeze(0).unsqueeze(0).long().cuda()
        local_blending_mask = torch.where(local_blending_mask==1, torch.ones_like(local_blending_mask), torch.zeros_like(local_blending_mask))
        local_blending_mask_down = F.interpolate(local_blending_mask.float(), size=(32, 32), mode='bicubic')
        src_feature = local_feature * local_blending_mask_down + src_feature * (1-local_blending_mask_down)

    with torch.no_grad():
        img_gen_blend, _ = generator([src_latent], input_is_latent=True, randomize_noise=False, start_layer=4, end_layer=8, layer_in=src_feature)
    return src_feature, img_gen_blend

def color_feature_blending(generator, seg, edited_hairstyle_img, src_latent, color_latent_in, latent_F):
    hair_seg = torch.argmax(seg(edited_hairstyle_img)[1], dim=1).unsqueeze(1).long()
    hari_mask = torch.where(hair_seg==10, torch.ones_like(hair_seg), torch.zeros_like(hair_seg))
    enlarged_hair_mask_np = dliate_erode(hari_mask[0][0].cpu().numpy().astype('uint8'), 30)
    enlarged_hair_mask = torch.from_numpy(enlarged_hair_mask_np).unsqueeze(0).unsqueeze(0).cuda()
    final_hair_mask = F.interpolate(enlarged_hair_mask.float(), size=(1024, 1024)).long().clone().detach()

    source_feature, _ = generator([src_latent], input_is_latent=True, randomize_noise=False, return_latents=False, start_layer=4, end_layer=6, layer_in=latent_F)
    color_feature, _ = generator([color_latent_in], input_is_latent=True, randomize_noise=False, return_latents=False, start_layer=4, end_layer=6, layer_in=latent_F)
    final_hair_mask_down = F.interpolate(final_hair_mask.float(), size=(256, 256), mode='bicubic')
    color_feature = color_feature * final_hair_mask_down + source_feature * (1-final_hair_mask_down)
    return color_feature, final_hair_mask