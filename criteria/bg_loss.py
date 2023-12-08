import torch
from torch import nn

class BackgroundLoss(nn.Module):
    def __init__(self, parsenet):
        super(BackgroundLoss, self).__init__()
        self.parsenet = parsenet
        self.bg_mask_l2_loss = torch.nn.MSELoss()

    def gen_bg_mask(self, input_image):
        labels_predict = torch.argmax(self.parsenet(input_image)[0], dim=1).unsqueeze(1).long().detach()
        bg_mask = (labels_predict!=10).float()
        return bg_mask

    def forward(self, x, x_hat):
        x_bg_mask = self.gen_bg_mask(x)
        x_hat_bg_mask = self.gen_bg_mask(x_hat)
        bg_mask = ((x_bg_mask+x_hat_bg_mask)==2).float()

        loss = self.bg_mask_l2_loss(x * bg_mask, x_hat * bg_mask)
        return loss

