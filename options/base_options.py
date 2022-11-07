import argparse
import os
import utils

class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        parser.add_argument('--train_video_root', required=True, help='path to train videos')
        parser.add_argument('--train_list_root', required=True, help='path to train videos list')
        parser.add_argument('--test_video_root', required=True, help='path to test videos')
        parser.add_argument('--test_list_root', required=True, help='path to test videos list')
        parser.add_argument('--permutation_root',default='./datasets/permutation_10.npy')
        parser.add_argument('--dataset_name', required=True,type=str,default='MMI',help='BU3D, AFEW, MMI, DFEW')
        parser.add_argument('--name',type=str,default='experiment_name',help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids',type=str,default='0',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_threads',default=4,type=int,help='# threads for loading data')
        parser.add_argument('--batch_size',type=int,default=8,help='input batch size')
        parser.add_argument('--seed',type=int,default=3456,help='random seed')
        parser.add_argument('--checkpoints_dir',type=str,default='./checkpoints',help='models are saved here')
        parser.add_argument('--phase',type=str,default='train',help='train,test')
        parser.add_argument('--epoch', default=0, type=int, help='start epoch count')
        parser.add_argument('--epochs_count', default=160, type=int)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum　(default: 0.9)')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=20, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--lamb', default=0.1428, type=float, help='control permutation')
        parser.add_argument('--warm_up',default=10,type=int)
        parser.add_argument('--print_freq',default=20,type=int)

        parser.add_argument('--snippets', type=int, default=7, help='the number of snippets')
        parser.add_argument('--per_snippets', type=int, default=5, help='the number of per snippets')
        parser.add_argument('--use_norm', action='store_false')
        parser.add_argument('--d_model',type=int,default=512)
        parser.add_argument('--nhead',type=int,default=4)
        parser.add_argument('--encoder_nums',type=int,default=3)
        parser.add_argument('--decoder_nums',type=int,default=3)
        parser.add_argument('--permutation_classes',type=int,default=10)
        parser.add_argument('--parameterDir',type=str,default='./parameters/Resnet18_FER+_pytorch.pth.tar')

        ###########Continue######################
        parser.add_argument('--continue_train',action='store_true')
        parser.add_argument('--pre_train_model_path',type=str)
        parser.add_argument('--heat_map_path',type=str,default='./heat_map')
        ###########Evaluation###################
        parser.add_argument('--eval_model_path',type=str)
        parser.add_argument('--draw_weight',action='store_true')
        parser.add_argument('--test_first',type=int,default=15)
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        return parser.parse_args()

    def print_options(self,opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        utils.mkdirs(opt.checkpoints_dir)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        str_time = utils.get_time()
        file_name = os.path.join(expr_dir, '{}_{}_opt.txt'.format(opt.phase,str_time))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = True if opt.phase == 'train' else False

        self.print_options(opt)
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        self.opt = opt
        return self.opt