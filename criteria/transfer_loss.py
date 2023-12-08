import torch
from criteria.style.style_loss import StyleLoss

class TransferLossBuilder(torch.nn.Module):
    def __init__(self):
        super(TransferLossBuilder, self).__init__()

        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).cuda()
        self.style.eval()


    def style_loss(self, im1, im2, mask1, mask2):
        loss = self.style(im1 * mask1, im2 * mask2, mask1=mask1, mask2=mask2)
        return loss


