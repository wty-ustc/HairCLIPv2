import sys
sys.path.append(".")
sys.path.append("..")
import dlib
from pathlib import Path
import argparse
import torchvision
from utils.shape_predictor import align_face
import PIL

parser = argparse.ArgumentParser(description='Align_face')

parser.add_argument('-unaligned_dir', type=str, default='test_images/unaligned_img/', help='directory with unaligned images')
parser.add_argument('-output_dir', type=str, default='test_images/aligned_img', help='output directory')
parser.add_argument('-output_size', type=int, default=1024, help='size to downscale the input images to, must be power of 2')
###############

args = parser.parse_args()
predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")

for im in Path(args.unaligned_dir).glob("*.*"):
    faces = align_face(str(im),predictor)
    for i,face in enumerate(faces):
        if(args.output_size):
            factor = 1024//args.output_size
            assert args.output_size*factor == 1024
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)
            if factor != 1:
                face = face.resize((args.output_size, args.output_size), PIL.Image.LANCZOS)
        if len(faces) > 1:
            face.save(Path(args.output_dir) / (im.stem+f"_{i}.jpg"))
        else:
            face.save(Path(args.output_dir) / (im.stem + f".jpg"))