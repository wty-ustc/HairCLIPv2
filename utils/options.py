from argparse import ArgumentParser


class Options:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for pretrained model path
		self.parser.add_argument('--stylegan_path', default="pretrained_models/ffhq.pt", type=str, help='Path to StyleGAN model checkpoint')
		self.parser.add_argument('--seg_path', default="pretrained_models/seg.pth", type=str, help='Path to face parsing model checkpoint')
		self.parser.add_argument('--bald_path', default="pretrained_models/bald_proxy.pt", type=str, help='Path to balding model checkpoint')
		self.parser.add_argument('--sketch_path', default="pretrained_models/sketch_proxy.pt", type=str, help='Path to sketch2hair model checkpoint')
		self.parser.add_argument('--ffhq_pca_path', default="pretrained_models/ffhq_PCA.npz", type=str, help='Path to FFHQ PCA')

		# arguments for image and latent dir path
		self.parser.add_argument('--src_img_dir', default="test_images/src_img", type=str, help='Folder of source image')
		self.parser.add_argument('--src_latent_dir', default="test_images/src_F", type=str, help='Folder of source latent')
		self.parser.add_argument('--ref_img_dir', default="test_images/ref_img", type=str, help='Folder of reference image')
		self.parser.add_argument('--ref_latent_dir', default="test_images/ref_latent", type=str, help='Folder of reference latent')

		# arguments for embedding
		self.parser.add_argument('--W_steps', default=1100, type=int, help='Step for W plus inversion')
		self.parser.add_argument('--FS_steps', default=250, type=int, help='Step for F inversion')
		self.parser.add_argument('--lr_embedding', default=0.01, type=float, help='Learning rate for embedding')
		self.parser.add_argument('--l2_lambda_embedding', default=1.0, type=float, help='L2 loss lambda for embedding')
		self.parser.add_argument('--percept_lambda_embedding', default=1.0, type=float, help='LPIPS loss lambda for embedding')
		self.parser.add_argument('--p_norm_lambda_embedding', default=0.001, type=float, help='P norm loss lambda for embedding')
		self.parser.add_argument('--l_F_lambda_embedding', default=0.1, type=float, help='F norm loss lambda for embedding')

		# arguments for text proxy
		self.parser.add_argument('--no_aug_clip_loss_text', default=False, action="store_true")
		self.parser.add_argument('--clip_lambda_text', default=1.0, type=float, help='CLIP loss lambda for text proxy')
		self.parser.add_argument('--hair_mask_lambda_text', default=1.0, type=float, help='Hair mask loss lambda for text proxy')
		self.parser.add_argument('--landmark_lambda_text', default=200.0, type=float, help='Landmark loss lambda for text proxy')
		self.parser.add_argument('--lr_text', default=0.01, type=float, help='Learning rate for text proxy')
		self.parser.add_argument('--steps_text', default=200, type=int, help='Step for text proxy optimization')
		self.parser.add_argument('--visual_num_text', default=10, type=int, help='Visual image number during text proxy optimization')

		# arguments for reference proxy
		self.parser.add_argument('--style_lambda_ref', default=40000.0, type=float, help='Style loss lambda for reference proxy')
		self.parser.add_argument('--delta_w_lambda_ref', default=1000.0, type=float, help='Delta W loss lambda for reference proxy')
		self.parser.add_argument('--hair_mask_lambda_ref', default=1.0, type=float, help='Hair mask loss lambda for reference proxy')
		self.parser.add_argument('--landmark_lambda_ref', default=1000.0, type=float, help='Landmark loss lambda for reference proxy')
		self.parser.add_argument('--lr_ref', default=0.01, type=float, help='Learning rate for reference proxy')
		self.parser.add_argument('--steps_ref', default=200, type=int, help='Step for reference proxy optimization')
		self.parser.add_argument('--visual_num_ref', default=10, type=int, help='Visual image number during reference proxy optimization')

		# arguments for color proxy
		self.parser.add_argument('--aug_clip_loss_color', default=False, action="store_true")
		self.parser.add_argument('--clip_lambda_color', default=1.0, type=float, help='CLIP loss lambda for color proxy')
		self.parser.add_argument('--bg_lambda_color', default=10.0, type=float, help='Background loss lambda for color proxy')
		self.parser.add_argument('--avg_color_lambda_color', default=0.01, type=float, help='Average color loss lambda for color proxy')
		self.parser.add_argument('--hair_mse_lambda_color', default=500.0, type=float, help='Hair L2 loss lambda for color proxy')
		self.parser.add_argument('--not_hair_mse_lambda_color', default=10.0, type=float, help='Not Hair L2 loss lambda for color proxy')
		self.parser.add_argument('--hair_lpips_lambda_color', default=1.0, type=float, help='Hair LPIPS loss lambda for color proxy')
		self.parser.add_argument('--lr_color', default=0.01, type=float, help='Learning rate for color proxy')
		self.parser.add_argument('--edit_steps_color', default=100, type=int, help='Editing step for color proxy optimization')
		self.parser.add_argument('--final_steps_color', default=100, type=int, help='Final blending step for color proxy optimization')
		self.parser.add_argument('--visual_num_color', default=10, type=int, help='Visual image number during color proxy optimization')


	def parse(self, jupyter=False):
		if jupyter:
			opts = self.parser.parse_args(args=[])
		else:
			opts = self.parser.parse_args()
		return opts