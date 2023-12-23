import torch
from models.stylegan2.model import Generator
from models.face_parsing.model import BiSeNet

def load_base_models(opts):
    ckpt = opts.stylegan_path
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    mean_latent = torch.load(ckpt)["latent_avg"].unsqueeze(0).unsqueeze(0).repeat(1,18,1).clone().detach().cuda()

    seg_pretrained_path = opts.seg_path
    seg = BiSeNet(n_classes=16)
    seg.load_state_dict(torch.load(seg_pretrained_path), strict=False)
    for param in seg.parameters():
        param.requires_grad = False
    seg.eval()
    seg = seg.cuda()

    return g_ema, mean_latent, seg