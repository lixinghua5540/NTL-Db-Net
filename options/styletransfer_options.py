from .base_options import BaseOptions

class StyleTransferOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch',          type=int, default=100,     help='The number of epoch')
        self.parser.add_argument('--content_weight', type=int, default=1,       help='The weight of content loss内容损失权重')
        self.parser.add_argument('--style_weight',   type=int, default=1000,    help='The weight of style loss风格损失权重')
        self.parser.add_argument('--cuda',           type=bool,default=True,    help='use cuda?')
        self.parser.add_argument('--psf_size',       type=int, default=15,      help='size of psf kernel')
        self.parser.add_argument('--psf_parameter',  type=list,default=[],      help='stores psf parameters')
        self.parser.add_argument('--psf_type',       type=str, default='Moffat',help='type of psf model')
        self.isTrain = True
