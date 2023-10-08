from .base_options import BaseOptions


class DebloomingGANOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--name', type=str, default='debloomingGAN', help='name of the experiment. It decides where to store samples and models')
		
		self.parser.add_argument('--save_latest_freq', 	type=int, 	default=200,	help='frequency of saving the latest results')
		self.parser.add_argument('--save_epoch_freq', 	type=int, 	default=20, 	help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--continue_train', 	action='store_true', 		help='continue training: load the latest model')
		self.parser.add_argument('--phase', 			type=str, 	default='train', help='train, val, test, etc')
		self.parser.add_argument('--which_epoch', 		type=str, 	default='latest', help='which epoch to load? set to latest to use latest cached model')
		self.parser.add_argument('--niter', 			type=int, 	default=100, 	help='# of iter at starting learning rate')
		self.parser.add_argument('--niter_decay', 		type=int, 	default=50, 	help='# of iter to linearly decay learning rate to zero')
		self.parser.add_argument('--beta1', 			type=float, default=0.5, 	help='momentum term of adam')
		self.parser.add_argument('--lr', 				type=float, default=0.001, help='initial learning rate for adam')
		self.parser.add_argument('--lr_stable', 		type=float, default=0.001, help='record the learning rate for update')
		self.parser.add_argument('--GAN_G_LossW', 		type=float, default=1.0, 	help='weight for GAN-G loss')
		self.parser.add_argument('--image_LossW',	 	type=float, default=100.0, 	help='weight for image loss')
		self.parser.add_argument('--pixel_LossW', 		type=float, default=200.0, 	help='weight for pixel loss')
		self.parser.add_argument('--identity', 			type=float, default=0.0, 	help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
		self.parser.add_argument('--pool_size', 		type=int, 	default=50, 	help='the size of image buffer that stores previously generated images')
		self.parser.add_argument('--no_html', 			action='store_true', 		help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		self.isTrain = True
