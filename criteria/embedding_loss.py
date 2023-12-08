import torch
from criteria import lpips
import PIL
import os

class EmbeddingLossBuilder(torch.nn.Module):
    def __init__(self, l2_lambda=1.0, percept_lambda=1.0):
        super(EmbeddingLossBuilder, self).__init__()
        self.l2_lambda = l2_lambda
        self.percept_lambda = percept_lambda
        self.parsed_loss = [[self.l2_lambda, 'l2'], [self.percept_lambda, 'percep']]
        self.l2 = torch.nn.MSELoss()
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=False).cuda()
        self.percept.eval()


    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)

    def _loss_lpips(self, gen_im, ref_im, **kwargs):

        return self.percept(gen_im, ref_im).sum()

    def forward(self, ref_im_H,ref_im_L, gen_im_H, gen_im_L):
        loss = 0
        loss_fun_dict = {
            'l2': self._loss_l2,
            'percep': self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'l2':
                var_dict = {
                    'gen_im': gen_im_H,
                    'ref_im': ref_im_H,
                }
            elif loss_type == 'percep':
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                }

            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        return loss, losses