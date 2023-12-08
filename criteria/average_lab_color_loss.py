import torch
from torch import nn

class AvgLabLoss(nn.Module):
    def __init__(self, parsenet):
        super(AvgLabLoss, self).__init__()
        self.parsenet = parsenet
        self.criterion = nn.L1Loss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])

    def gen_hair_mask(self, input_image):
        labels_predict = torch.argmax(self.parsenet(input_image)[0], dim=1).unsqueeze(1).long().detach()
        hair_mask = (labels_predict==10).float()
        return hair_mask

    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[~mask] = 7.787 * input[~mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output

    def cal_hair_avg(self, input, mask):
        x = input * mask
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        return avg

    def forward(self, target, result):
        if target.shape[-1] != 1:
            mask_target = self.gen_hair_mask(target)
            target_Lab = self.xyz2lab(self.rgb2xyz((target + 1) / 2.0))
            target_Lab_avg = self.cal_hair_avg(target_Lab, mask_target)
        else:
            target_Lab_avg = self.xyz2lab(self.rgb2xyz((target + 1) / 2.0))

        mask_result = self.gen_hair_mask(result)
        result_Lab = self.xyz2lab(self.rgb2xyz((result + 1) / 2.0))
        result_Lab_avg = self.cal_hair_avg(result_Lab, mask_result)

        loss = self.criterion(target_Lab_avg, result_Lab_avg)
        return loss