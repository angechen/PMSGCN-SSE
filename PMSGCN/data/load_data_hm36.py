"""
fuse training and testing

"""
import os

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import math

from data.common.camera import *
from data.common.utils import deterministic_random
from data.common.generator import ChunkedGenerator



class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset
        self.train = train    #true ;false
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')   #none
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.input_inverse_intrinsic = opt.input_inverse_intrinsic
        if self.train: #true
            self.keypoints = self.prepare_data(dataset, self.train_list, opt)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,  #[4,5,6,11,12,13],[1,2,3,14,15,16]
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list, opt)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list, opt):
        print('Preparing data...')
        for subject in folder_list:
            print('load %s' % subject)
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                multiViewUV = []

                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    uv1 = project_to_2d(torch.tensor(pos_3d),torch.tensor(np.tile(cam['intrinsic'],(pos_3d.shape[0],1))))
                    multiViewUV.append(uv1)
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
                anim['multiView_uv'] = multiViewUV

        print('Loading 2D detections...')
        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()


        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):   #4

                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

        for subject in folder_list:
            for action in keypoints[subject]:
                if opt.use_projected_2dgt == False:
                    a = keypoints[subject][action]
                else:
                    a = dataset._data[subject][action]['multiView_uv']
                for cam_idx, kps in enumerate(a):
                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]
                    kps1 = np.zeros((kps.shape[0], kps.shape[1], 3), dtype=float)
                    kps1[:, :, 0:2] = kps
                    kps1[:, :, 2:3] = np.ones((kps.shape[0], kps.shape[1], 1))
                    cam_intrinsic_matrix = np.concatenate([cam['intrinsic_matrix'], np.array([[0,0,1]])], axis=0)
                    kps1 = np.matmul(np.linalg.inv(np.tile(cam_intrinsic_matrix, (kps.shape[0], kps.shape[1], 1, 1))),np.expand_dims(kps1, axis=3))
                    if self.crop_uv == 0:
                        if opt.use_projected_2dgt == True:
                            kps = kps.numpy()
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

                    if self.input_inverse_intrinsic:
                        if cam_idx < 4:
                            keypoints[subject][action][cam_idx] = kps1.reshape((kps.shape[0], kps.shape[1], 3))
                        else:
                            keypoints[subject][action].append(kps1.reshape((kps.shape[0], kps.shape[1], 3)))
                    else:
                        if cam_idx < 4:
                            keypoints[subject][action][cam_idx] = kps
                        else:
                            keypoints[subject][action].append(kps)
        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        """

        :param dataset:
        :param subjects:
        :param subset:
        :param parse_3d_poses:
        :return: for each pose dict it has key(subject,action,cam_index)
        """
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None: #none
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]


                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):  # 4
                        out_poses_3d[(subject, action, i)] = poses_3d[i]


        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample #1
        if subset < 1:  #1,false
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1: #false
            # Downsample as requested

            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        "Figure our how many sequences we have"

        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        if self.train == False and self.test_aug:

            _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)
        if flip == True:
            flip = 1
        else:
            flip = 0

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind, index, start_3d, flip








