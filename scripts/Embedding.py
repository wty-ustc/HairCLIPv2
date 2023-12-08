import torch
from torch import nn
import numpy as np
import os
from typing import Tuple
from functools import partial
from utils.bicubic import BicubicDownSample
from criteria.embedding_loss import EmbeddingLossBuilder
from tqdm import tqdm
import PIL
import torchvision
from torchvision import transforms
from skimage import io
import cv2
from PIL import Image
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, generator, mean_latent_code, W_steps=1100, FS_steps=250):
        super(Embedding, self).__init__()
        self.generator = generator
        self.mean_latent_code = mean_latent_code
        self.W_steps = W_steps
        self.FS_steps = FS_steps
        self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.load_PCA_model()
        self.load_downsampling()
        self.setup_embedding_loss_builder()

    def load_PCA_model(self):
        PCA_path = "pretrained_models/ffhq_PCA.npz"
        if not os.path.isfile(PCA_path):
            print("Can not find the PCA_PATH for FFHQ!")

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().cuda()
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().cuda()
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().cuda()

    def load_downsampling(self):
        factor = 1024 // 256
        self.downsample = BicubicDownSample(factor=factor, cuda=True)

    def setup_W_optimizer(self):
        latent = []
        for i in range(18):
            tmp = self.mean_latent_code.clone().detach().cuda()
            tmp.requires_grad = True
            latent.append(tmp)
        optimizer_W = torch.optim.Adam(latent, lr=0.01)

        return optimizer_W, latent

    def setup_FS_optimizer(self, latent_W, F_init):
        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        for i in range(18):
            tmp = latent_W[0, i].clone()
            if i < 7:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True
            latent_S.append(tmp)
        optimizer_FS = torch.optim.Adam(latent_S[7:] + [latent_F], lr=0.01)
        return optimizer_FS, latent_F, latent_S

    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder()

    def invert_image_in_W(self, image_path=None):
        ref_im = Image.open(image_path).convert('RGB')
        ref_im_L = self.image_transform(ref_im.resize((256, 256), PIL.Image.LANCZOS)).unsqueeze(0)
        ref_im_H = self.image_transform(ref_im.resize((1024, 1024), PIL.Image.LANCZOS)).unsqueeze(0)

        optimizer_W, latent = self.setup_W_optimizer()
        pbar = tqdm(range(self.W_steps), desc='Embedding', leave=False)
        for step in pbar:
            optimizer_W.zero_grad()
            latent_in = torch.stack(latent).unsqueeze(0)
            gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False)
            im_dict = {
                'ref_im_H': ref_im_H.cuda(),
                'ref_im_L': ref_im_L.cuda(),
                'gen_im_H': gen_im,
                'gen_im_L': self.downsample(gen_im)
            }
            loss, loss_dic = self.cal_loss(im_dict, latent_in)
            loss.backward()
            optimizer_W.step()
            pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'\
                .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))
        return latent_in

    def invert_image_in_FS(self, image_path=None):
        ref_im = Image.open(image_path).convert('RGB')
        ref_im_L = self.image_transform(ref_im.resize((256, 256), PIL.Image.LANCZOS)).unsqueeze(0)
        ref_im_H = self.image_transform(ref_im.resize((1024, 1024), PIL.Image.LANCZOS)).unsqueeze(0)

        latent_W = self.invert_image_in_W(image_path=image_path).clone().detach()
        F_init, _ = self.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
        optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)

        pbar = tqdm(range(self.FS_steps), desc='Embedding', leave=False)
        for step in pbar:
            optimizer_FS.zero_grad()
            latent_in = torch.stack(latent_S).unsqueeze(0)
            gen_im, _ = self.generator([latent_in], input_is_latent=True, return_latents=False,
                                           start_layer=4, end_layer=8, layer_in=latent_F)
            im_dict = {
                'ref_im_H': ref_im_H.cuda(),
                'ref_im_L': ref_im_L.cuda(),
                'gen_im_H': gen_im,
                'gen_im_L': self.downsample(gen_im)
            }
            loss, loss_dic = self.cal_loss(im_dict, latent_in)
            loss.backward()
            optimizer_FS.step()
            pbar.set_description(
                'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))
        return latent_in, latent_F


    def cal_p_norm_loss(self, latent_in, p_norm_lambda=0.001):
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss

    def cal_l_F(self, latent_F, F_init, l_F_lambda=0.1):
        return l_F_lambda * (latent_F - F_init).pow(2).mean()

    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic


