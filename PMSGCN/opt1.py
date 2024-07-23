import argparse
import os
import math



class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        # ===============================================================
        #                     Dataset options
        # ===============================================================
        self.parser.add_argument('--dataset', type=str, default='h36m', help='dataset')
        self.parser.add_argument('--crop_uv', type=int, default=0, help='if crop_uv to center and do normalization')
        self.parser.add_argument('--root_path', type=str, default='/home/chuanjiang/Projects/PMSGCN_SSE/PMSGCN/data/dataset/', help='dataset root path')
        self.parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                            help='actions to train/test on, separated by comma, or * for all')
        self.parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                            help='downsample frame rate by factor (semi-supervised)')
        self.parser.add_argument('--subset', default=1, type=float, metavar='FRACTION',
                            help='reduce dataset size by fraction')
        self.parser.add_argument('-s', '--stride', default=1, type=int, metavar='N',
                            help='chunk size to use during training')
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False, help='if reverse the video to augment data')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--pro_train', type=int, default=1, help='if start train process')
        self.parser.add_argument('--pro_test', type=int, default=1, help='if start test process')
        self.parser.add_argument('--nepoch', type=int, default=200, help='number of epochs')
        self.parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
        self.parser.add_argument('--large_decay_epoch', type=int, default=5, help='give a large lr decay after how manys epochs')
        self.parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
        self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer for SGD')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='SGD or Adam')
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float, metavar='LR',
                            help='learning rate decay per epoch')
        self.parser.add_argument('--save_skl', type=bool, default=False)#

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--show_protocol2', action='store_true')#
        self.parser.add_argument('--model_doc', type=str, default='pms_gcn', help='current model document name')
        self.parser.add_argument('--layout', type=str, default='hm36_gt', help='dataset used')
        self.parser.add_argument('--strategy', type=str, default='spatial', help='dataset used')
        self.parser.add_argument('--save_model', type=int, default=1, help='if save model')
        self.parser.add_argument('--save_out_type', type=str, default='xyz', help='xyz/uvd/post/time')
        self.parser.add_argument('--stgcn_reload', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--post_refine_reload', type=int, default=0, help='if continue from last time')
        self.parser.add_argument('--previous_dir', type=str,
                                 default='/home/chuanjiang/Projects/PMSGCN_SSE/PMSGCN/results/pms_gcn/no_pose_refine/gt/',
                                 help='previous output folder')
        self.parser.add_argument('--pmsgcn_model', type=str, default='',
                                 help='model name')
        self.parser.add_argument('--post_refine_model', type=str, default='',
                                 help='model name')
        self.parser.add_argument('--n_joints', type=int, default=17, help='number of joints')
        self.parser.add_argument('--out_joints', type=int, default=17, help='number of joints')
        self.parser.add_argument('--out_all', type=bool, default=True, help='output 1 frame or all frames')
        self.parser.add_argument('--out_channels', type=int, default=3, help='expected input channels here 2')
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf,
                            help='threshold data:reg_RGB_3D/reg_3D_3D')
        self.parser.add_argument('-previous_pms_gcn_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('-previous_post_refine_name', type=str, default='', help='save last saved model name')


        self.parser.add_argument('--data_augmentation', type=bool, default=True, help='disable train-time flipping')
        self.parser.add_argument('--test_augmentation', type=bool, default=True, help='flip and fuse the output result')
        self.parser.add_argument('--cal_uvd', type=bool, default=True, help='calculate uvd error as well')
        self.parser.add_argument('--learning_rate', type=float, default=1.5e-3)
        self.parser.add_argument('--sym_penalty', type=int, default=0, help='if add sym penalty add on train_multi')
        self.parser.add_argument('--pad', type=int, default=1)
        self.parser.add_argument('--post_refine', action='store_true', help='if use post_refine model')

        self.parser.add_argument('--in_channels', type=int, default=2, help='expected input channels here 2')
        self.parser.add_argument('--input_inverse_intrinsic', type=bool, default=False, help='if input_inverse_intrinsic')
        self.parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME',
                                  help='2D detections to use {gt||cpn_ft_h36m_dbb}')

        self.parser.add_argument('--boneAngle_define', type=str, default='cosa_|1+cosb|', help='how to compute boneAngle')
        self.parser.add_argument('--boneLength_penalty', type=int, default=1, help='if add bone_length_penalty on train_multi')
        self.parser.add_argument('--boneAngle_penalty', type=int, default=1, help='if add bone_angle_penalty on train_multi')
        self.parser.add_argument('--neighbourBoneAngle_penalty', type=int, default=1, help='if add neighbourBoneAngle_penalty on train_multi')
        self.parser.add_argument('--twoStepBoneAngle_penalty', type=int, default=1, help='if add two-step bone_angle_penalty on train_multi')
        self.parser.add_argument('--threeStepBoneAngle_penalty', type=int, default=1, help='if add three-step bone_angle_penalty on train_multi')
        self.parser.add_argument('--fourStepBoneAngle_penalty', type=int, default=1, help='if add four-step bone_angle_penalty on train_multi')
        self.parser.add_argument('--fiveStepBoneAngle_penalty', type=int, default=1, help='if add five-step bone_angle_penalty on train_multi')
        self.parser.add_argument('--sixStepBoneAngle_penalty', type=int, default=1, help='if add six-step bone_angle_penalty on train_multi')
        self.parser.add_argument('--sevenStepBoneAngle_penalty', type=int, default=1, help='if add seven-step bone_angle_penalty on train_multi')

        self.parser.add_argument('--co_diff', type=float, default=0)
        self.parser.add_argument('--co_boneLength', type=float, default=0.066)#1
        self.parser.add_argument('--co_boneAngle', type=float, default=0.066)#1
        self.parser.add_argument('--co_neighbourBoneAngle', type=float, default=0.059)#0.1
        self.parser.add_argument('--co_twostep_neighbourBoneAngle', type=float, default=0.059)
        self.parser.add_argument('--co_threestep_neighbourBoneAngle', type=float, default=0.055)
        self.parser.add_argument('--co_fourstep_neighbourBoneAngle', type=float, default=0.048)
        self.parser.add_argument('--co_fivestep_neighbourBoneAngle', type=float, default=0.041)
        self.parser.add_argument('--co_sixstep_neighbourBoneAngle', type=float, default=0.035)
        self.parser.add_argument('--co_sevenstep_neighbourBoneAngle', type=float, default=0.027)


        self.parser.add_argument('--show_boneError', type=bool, default=False, help='if show bone error')

        self.parser.add_argument('--twoStepNeigh', type=bool, default=True, help='if add partial two step neighbour pairs')
        self.parser.add_argument('--partialSym', type=bool, default=False, help='if add partial sym pairs')
        self.parser.add_argument('--tcnlayer', type=bool, default=True, help='if fuse features after gcn layer')
        self.parser.add_argument('--dynamic_correlation_weights', type=bool, default=True, help='if correlation weights between nodes are dynamic')
        self.parser.add_argument('--channel_sharedweights', type=bool, default=True, help='if multi-channel input features share one learnable correlation weight matrix')
        self.parser.add_argument('--pmsgcn_non_local', type=bool, default=False, help='if pmsgcn apply non_local layer')
        self.parser.add_argument('--framework', default='pmsgcn', type=str, metavar='NAME', help='framework to run')
        self.parser.add_argument('--learn_adjacency', type=bool, default=False, help='if set adjacency matrix learnable')
        self.parser.add_argument('--set_loss3d_weights', type=bool, default=False, help='if set or learn weights of 3 dimensions of loss-XYZ')
        self.parser.add_argument('--x15_to_x16', type=str, default='conv128-512-512-conv9Res31', help='layers from x15 to x16=gcn_conv||gcn_conv_noRes||conv128-512-512-conv12Res||conv128-512-512-conv9Res31||conv128-512-512-conv9Res11')
        self.parser.add_argument('--use_projected_2dgt', type=bool, default=False, help='if use projected 2d gt as input')
        self.parser.add_argument('--error_rule', default='F1', type=str, metavar='NAME', help='error:absF1/mseF2')
        self.parser.add_argument('--vis_keypoints', type=bool, default=False, help='if use projected 2d gt as input')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

        self.opt.save_dir = './results/'+'%d_frame/'%(self.opt.pad*2+1) +self.opt.model_doc + '/'+ \
        '%spose_refine/'%('' if self.opt.post_refine else 'no_')    #'./results/3_frame/st_gcn/pose_refine/'


        if self.opt.dataset == 'h36m':
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'



        if self.opt.keypoints == 'cpn_ft_h36m_dbb':
            self.opt.save_dir += 'cpn/'  #'./results/3_frame/st_gcn/pose_refine/cpn/'
            self.opt.layout = 'hm36_gt'

        elif self.opt.keypoints == 'gt':
            self.opt.save_dir += 'gt/'
            self.opt.layout = 'hm36_gt'


        if not os.path.exists(self.opt.save_dir):
            os.makedirs(self.opt.save_dir)
        file_name = os.path.join(self.opt.save_dir, 'opt.txt')  #'./results/3_frame/st_gcn/pose_refine/cpn/opt.txt'


        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        print(self.opt)
        return self.opt
