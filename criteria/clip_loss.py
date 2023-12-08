import torch
import clip
from torchvision import transforms
import kornia.augmentation as K
import torch.nn.functional as F

class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.clip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=1024 // 32)

    def forward(self, image, text):
        image = self.clip_normalize(((image+1)/2))
        image = self.avg_pool(self.upsample(image))
        text_token = torch.cat([clip.tokenize(text)]).cuda()
        similarity = 1 - self.model(image, text_token)[0] / 100
        return similarity[0,0]


class ImageAugmentations(torch.nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number

        self.augmentations = torch.nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=p),
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input):
        resized_images = self.avg_pool(input)
        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))
        batch_size = input.shape[0]
        # We want at least one non augmented image
        non_augmented_batch = resized_images[:batch_size]
        augmented_batch = self.augmentations(resized_images[batch_size:])
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)

        return updated_batch


class AugCLIPLoss(torch.nn.Module):
    def __init__(self):
        super(AugCLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.clip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.image_augmentations = ImageAugmentations(224, 8)

    def d_clip_loss(self, x, y, use_cosine=False):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        if use_cosine:
            distance = 1 - (x @ y.t()).squeeze()
        else:
            distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        return distance

    def forward(self, image, text):
        text_embed = self.model.encode_text(clip.tokenize(text).cuda()).float()
        clip_loss_value = torch.tensor(0)
        augmented_input = self.image_augmentations(image).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.model.encode_image(clip_in).float()
        dists = self.d_clip_loss(image_embeds, text_embed)
        # We want to sum over the averages
        clip_loss_value = dists.mean()
        return clip_loss_value