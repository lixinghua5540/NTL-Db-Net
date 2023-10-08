from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--result_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--test_img_name', type=str, default='./dataset/test/aomen.jpg', help='the full image of the test image.')
        self.parser.add_argument('--test_dir', type=str, default='./dataset/test/test_cut/', help='test dataset.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--num_row', type=int, default=0, help='number of landscape images')
        self.parser.add_argument('--num_col', type=int, default=0, help='number of portrait images')
        self.parser.add_argument('--step', type=int, default=300, help='Image cropping step(px)')

        self.isTrain = False
