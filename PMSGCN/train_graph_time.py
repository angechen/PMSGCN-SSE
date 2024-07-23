from tqdm import tqdm
from utils.utils1 import *
import data.common.eval_cal as eval_cal
import data.common.eval_cal2 as eval_cal2
import torch
import math
import data.visual_image.vis as vis




def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):

    # initialize some definitions
    num_data_all = 0

    loss_all_sum = {'loss_gt':AccumLoss(), 'loss_diff':AccumLoss(), 'loss_sum':AccumLoss(), 'loss_boneLength':AccumLoss(), 'loss_boneAngle':AccumLoss(), 'loss_neighbourBoneAngle': AccumLoss(), 'loss_neighbourBoneAngle_a': AccumLoss(),'loss_neighbourBoneAngle_b': AccumLoss(),'loss_two_step_neighbourBoneAngle': AccumLoss(), 'loss_two_step_neighbourBoneAngle_a': AccumLoss(),'loss_two_step_neighbourBoneAngle_b': AccumLoss(),'loss_three_step_neighbourBoneAngle': AccumLoss(), 'loss_three_step_neighbourBoneAngle_a': AccumLoss(),'loss_three_step_neighbourBoneAngle_b': AccumLoss(),'loss_four_step_neighbourBoneAngle': AccumLoss(), 'loss_four_step_neighbourBoneAngle_a': AccumLoss(),'loss_four_step_neighbourBoneAngle_b': AccumLoss(),'loss_five_step_neighbourBoneAngle': AccumLoss(),'loss_five_step_neighbourBoneAngle_a': AccumLoss(),'loss_five_step_neighbourBoneAngle_b': AccumLoss(),'loss_six_step_neighbourBoneAngle': AccumLoss(),'loss_six_step_neighbourBoneAngle_a': AccumLoss(),'loss_six_step_neighbourBoneAngle_b': AccumLoss(),'loss_seven_step_neighbourBoneAngle': AccumLoss(),'loss_seven_step_neighbourBoneAngle_a': AccumLoss(),'loss_seven_step_neighbourBoneAngle_b': AccumLoss(),'loss_boneAngle_cosa':AccumLoss(),'loss_neighbourBoneAngle_cosa': AccumLoss(), 'loss_two_step_neighbourBoneAngle_cosa': AccumLoss(),'loss_three_step_neighbourBoneAngle_cosa': AccumLoss(),'loss_four_step_neighbourBoneAngle_cosa': AccumLoss(), 'loss_five_step_neighbourBoneAngle_cosa': AccumLoss(),'loss_six_step_neighbourBoneAngle_cosa': AccumLoss(),'loss_seven_step_neighbourBoneAngle_cosa': AccumLoss()} #<utils.utils1.AccumLoss

    mean_error = {'xyz': 0.0, 'post': 0.0}
    error_sum = AccumLoss()

    if opt.dataset == 'h36m':
        limb_center = [2, 5, 11, 14] if opt.keypoints.startswith('sh') else [2, 5, 12, 15]
        limb_terminal = [3, 6, 12, 15] if opt.keypoints.startswith('sh') else [3,6,13,16]
        action_error_sum = define_error_list(actions)

        action_error_sum_post_out = define_error_list(actions)
        joints_left =[4, 5, 6, 10, 11, 12]  if opt.keypoints.startswith('sh') else [4,5,6,11,12,13]
        joints_right = [1, 2, 3, 13, 14, 15] if opt.keypoints.startswith('sh') else [1, 2, 3, 14, 15, 16]

    criterion_mse = criterion['MSE']
    model_st_gcn = model['pms_gcn']
    model_post_refine = model['post_refine']


    if split == 'train':
        model_st_gcn.train()
        if opt.out_all: #true
            out_all_frame = True
        else:
            out_all_frame = False

    else:
        model_st_gcn.eval()
        out_all_frame = False

    torch.cuda.synchronize()



    # load data
    for i, data in enumerate(tqdm(dataLoader, 0)):

        if split == 'train': #true
            if optimizer is None: #false
                print("error! No Optimizer")
            else:
                optimizer_all = optimizer #adam

        # load and process data
        batch_cam, gt_3D, input_2D, action, subject, scale , bb_box, cam_ind, index, start_3d, flip = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe (split,[input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)
        num_data_all += N

        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels)
        out_target[:, :, 0] = 0
        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:,opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D


        if opt.test_augmentation and split =='test': #true,train
            input_2D, output_3D = input_augmentation(input_2D, model_st_gcn, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints,opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
            output_3D = model_st_gcn(input_2D, out_all_frame)

        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.n_joints, opt.out_channels)
        if output_3D.size(1) > 1:
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1)
        else:
            output_3D_single = output_3D

        if split =='test' and opt.vis_keypoints == True:
            for batch_i in range(N): #for each frame plot
                vis.visualize_3d_gt_pred(out_target_single, output_3D_single, kind="human36m", batch_index=batch_i)

        output_bone_vector = torch.zeros([N, output_3D.size(1), 16, 3]).cuda()
        target_bone_vector = torch.zeros([N, out_target.size(1), 16, 3]).cuda()
        father_nodes = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        for i in range(0, len(father_nodes)):
            output_bone_vector[:, :, i:i+1, :] = output_3D[:, :, i+1:i+2, :] - output_3D[:, :, father_nodes[i]:father_nodes[i]+1, :]
            target_bone_vector[:, :, i:i+1, :] = out_target[:, :, i+1:i+2, :] - out_target[:, :,father_nodes[i]:father_nodes[i]+1, :]
        if target_bone_vector.size(1) > 1:
            target_bone_vector_single = target_bone_vector[:, opt.pad].unsqueeze(1)
            gt_bone_vector_single = target_bone_vector_single
        else:
            target_bone_vector_single = target_bone_vector
            gt_bone_vector_single = target_bone_vector

        if output_bone_vector.size(1) > 1:
            output_bone_vector_single = output_bone_vector[:, opt.pad].unsqueeze(1)
        else:
            output_bone_vector_single = output_bone_vector

        if split == 'train':
            output_bone_vector = output_bone_vector # N, T, V, 3
        elif split == 'test':
            output_bone_vector = output_bone_vector_single

        if split == 'train':
            pred_out = output_3D # N, T, V, 3
        elif split == 'test':
            pred_out = output_3D_single

        if not opt.input_inverse_intrinsic:
            # from uvd get xyz
            input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)
            if opt.crop_uv:  #0
                pred_uv = back_to_ori_uv(input_2D, bb_box)
            else:
                pred_uv = input_2D

            uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0

        if opt.post_refine:
            post_out = model_post_refine(output_3D_single, xyz)
            loss_sym = eval_cal.sym_penalty(opt.dataset, opt.keypoints, post_out)
            loss_post_refine = eval_cal.mpjpe(post_out, out_target_single) + 0.01*loss_sym
        else:
            loss_post_refine = 0


        #calculate loss
        if opt.set_loss3d_weights == False:
            loss_gt = eval_cal.mpjpe(pred_out, out_target)
        else:
            loss_gt = eval_cal2.weighted_mpjpe()(pred_out, out_target)

        if opt.sym_penalty:
            loss_gt += 0.005 * eval_cal.sym_penalty(opt.dataset, opt.keypoints, pred_out)

        if opt.pad == 0 or split == 'test':
            loss_diff = torch.zeros(1).cuda()
        else:
            weight_diff = 4 * torch.ones(output_3D[:, :-1, :].size()).cuda()
            weight_diff[:, :, limb_center] = 2.5
            weight_diff[:, :, limb_terminal] = 1
            diff = (output_3D[:,1:] - output_3D[:,:-1]) * weight_diff
            loss_diff = criterion_mse(diff, Variable(torch.zeros(diff.size()), requires_grad=False).cuda())

        if opt.boneLength_penalty:
            if opt.error_rule == 'F1':
                loss_boneLength_batch = torch.mean( torch.abs(torch.norm(output_bone_vector, dim=len(output_bone_vector.shape)-1) - torch.norm(target_bone_vector, dim=len(target_bone_vector.shape)-1)), dim=len(target_bone_vector.shape) - 2).cuda()
            elif opt.error_rule == 'F2':
                loss_boneLength_batch = torch.mean(((torch.norm(output_bone_vector, dim=len(output_bone_vector.shape)-1) - torch.norm(target_bone_vector, dim=len(target_bone_vector.shape)-1))**2), dim=len(target_bone_vector.shape) - 2).cuda()
            loss_boneLength = torch.mean(loss_boneLength_batch) #mean(512,3)
        if opt.boneAngle_penalty:
            if opt.boneAngle_define == 'vector_product_cosa':
                loss_boneAngle_batch_cosa = (torch.mean((1-(torch.sum(torch.mul(output_bone_vector, target_bone_vector),dim=(3,)) / (torch.norm(output_bone_vector,dim=len(output_bone_vector.shape)-1) * torch.norm(target_bone_vector, dim=len(target_bone_vector.shape)-1)))), dim=len(target_bone_vector.shape)-2) ).cuda()
                loss_boneAngle_cosa = torch.mean(loss_boneAngle_batch_cosa)

            if opt.boneAngle_define == 'cosa_|1+cosb|':
                if opt.error_rule == 'F1':
                    loss_a_batch = (torch.mean(torch.abs(((output_bone_vector[:, :, :, 2]/(torch.norm(output_bone_vector, dim=len(output_bone_vector.shape)-1))) - (target_bone_vector[:, :, :, 2]/(torch.norm(target_bone_vector, dim=len(target_bone_vector.shape)-1))))), dim=len(target_bone_vector.shape) - 2)).cuda()
                elif opt.error_rule == 'F2':
                    loss_a_batch = (torch.mean((((output_bone_vector[:, :, :, 2]/(torch.norm(output_bone_vector, dim=len(output_bone_vector.shape)-1))) - (target_bone_vector[:, :, :, 2]/(torch.norm(target_bone_vector, dim=len(target_bone_vector.shape)-1))))**2), dim=len(target_bone_vector.shape) - 2)).cuda()
                loss_a = torch.mean(loss_a_batch)

                output_x_pos_or_neg = torch.ones(N, output_3D.size(1), 16).cuda()
                output_x_pos_or_neg[output_bone_vector[:, :, :, 0]<0] = -1 #512,3,16  x>=0-->1, x<0=-->-1
                target_x_pos_or_neg = torch.ones(N, output_3D.size(1), 16).cuda()
                target_x_pos_or_neg[target_bone_vector[:, :, :, 0]<0] = -1 #512,3,16  x>=0-->1, x<0=-->-1

                output_bone_b = (output_x_pos_or_neg * (1+(output_bone_vector[:, :, :, 1] / torch.sqrt(output_bone_vector[:, :, :, 0]**2 + output_bone_vector[:, :, :, 1]**2)))).cuda()
                target_bone_b = (target_x_pos_or_neg * (1+(target_bone_vector[:, :, :, 1] / torch.sqrt(target_bone_vector[:, :, :, 0]**2 + target_bone_vector[:, :, :, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_b_batch = torch.mean(torch.abs((output_bone_b-target_bone_b)), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_b_batch = torch.mean(((output_bone_b-target_bone_b)**2), dim=len(target_bone_vector.shape) - 2)
                loss_b = torch.mean(loss_b_batch)

                loss_boneAngle_batch = loss_a_batch + loss_b_batch
                loss_boneAngle = loss_a + loss_b

        if opt.neighbourBoneAngle_penalty:
            neighbour_bone_pairs = [(0,1),(1,2),(0,3),(3,4),(4,5),(0,6),(3,6),(6,7),(7,8),(8,9),(7,10),(8,10),(10,11),(11,12),(7,13),(8,13),(13,14),(14,15)]
            bone_pairs_diff_direction = [-1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1]
            output_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda() #512,3,18
            target_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()
            output_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda() #512,3,18
            target_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()

            output_bone_bi = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()
            output_bone_bj = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()
            target_bone_bi = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()
            target_bone_bj = torch.zeros(N,output_3D.size(1),len(neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n,(i,j) in enumerate(neighbour_bone_pairs):
                    output_neighbour_bone_angles[:,:,n] = ((bone_pairs_diff_direction[n]*torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]),dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_neighbour_bone_angles[:,:,n] = ((bone_pairs_diff_direction[n]*torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]),dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_neighbour_bone_angles - target_neighbour_bone_angles), dim=len(output_neighbour_bone_angles.shape)-1).cuda()
                elif opt.error_rule == 'F2':    
                    loss_neighbourBoneAngle_batch_cosa = torch.mean(((output_neighbour_bone_angles - target_neighbour_bone_angles)**2), dim=len(output_neighbour_bone_angles.shape)-1).cuda()
                loss_neighbourBoneAngle_cosa = torch.mean(loss_neighbourBoneAngle_batch_cosa)
            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(neighbour_bone_pairs):
                    output_bone_pairs_a_difference[:, :, n] = ((bone_pairs_diff_direction[n] * output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_bone_pairs_a_difference[:, :, n] = ((bone_pairs_diff_direction[n] * target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_bone_pairs_batch = torch.mean(torch.abs((output_bone_pairs_a_difference - target_bone_pairs_a_difference)), dim=len(output_bone_pairs_a_difference.shape)-1)
                    loss_b_bone_pairs_batch = torch.mean(torch.abs(((output_bone_bi-output_bone_bj)-(target_bone_bi-target_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_bone_pairs_batch = torch.mean(((output_bone_pairs_a_difference - target_bone_pairs_a_difference)**2), dim=len(output_bone_pairs_a_difference.shape)-1)
                    loss_b_bone_pairs_batch = torch.mean((((output_bone_bi-output_bone_bj)-(target_bone_bi-target_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)        
                loss_a_bone_pairs = torch.mean(loss_a_bone_pairs_batch)
                loss_b_bone_pairs = torch.mean(loss_b_bone_pairs_batch)

                loss_neighbourBoneAngle_batch = loss_a_bone_pairs_batch + loss_b_bone_pairs_batch
                loss_neighbourBoneAngle = loss_a_bone_pairs + loss_b_bone_pairs

        if opt.twoStepBoneAngle_penalty:
            two_step_neighbour_bone_pairs = [(0, 2), (0, 4), (0, 7), (1, 3), (1, 6), (3, 5), (3, 7), (4, 6), (6, 8), (6, 10), (6, 13), (7, 9), (7, 11), (7, 14), (8, 11), (8, 14), (9, 10), (9, 13), (10, 12), (10, 14), (13, 15), (11, 13)]
            two_step_bone_pairs_diff_direction = [-1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1]
            output_two_step_neighbour_bone_angles = torch.zeros(N, output_3D.size(1), len(two_step_neighbour_bone_pairs)).cuda()
            target_two_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            output_two_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            target_two_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()

            output_two_step_bone_bi = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            output_two_step_bone_bj = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            target_two_step_bone_bi = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            target_two_step_bone_bj = torch.zeros(N,output_3D.size(1),len(two_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(two_step_neighbour_bone_pairs): #cosa
                    output_two_step_neighbour_bone_angles[:, :, n] = ((two_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_two_step_neighbour_bone_angles[:, :, n] = ((two_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_two_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_two_step_neighbour_bone_angles - target_two_step_neighbour_bone_angles), dim=len(output_two_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_two_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_two_step_neighbour_bone_angles - target_two_step_neighbour_bone_angles)**2), dim=len(output_two_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_two_step_neighbourBoneAngle_cosa = torch.mean(loss_two_step_neighbourBoneAngle_batch_cosa)

            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(two_step_neighbour_bone_pairs):
                    output_two_step_bone_pairs_a_difference[:, :, n] = ((two_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_two_step_bone_pairs_a_difference[:, :, n] = ((two_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_two_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_two_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_two_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_two_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_two_step_bone_pairs_batch = torch.mean(torch.abs((output_two_step_bone_pairs_a_difference - target_two_step_bone_pairs_a_difference)), dim=len(output_two_step_bone_pairs_a_difference.shape)-1)
                    loss_b_two_step_bone_pairs_batch = torch.mean(torch.abs(((output_two_step_bone_bi-output_two_step_bone_bj)-(target_two_step_bone_bi-target_two_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_two_step_bone_pairs_batch = torch.mean(((output_two_step_bone_pairs_a_difference - target_two_step_bone_pairs_a_difference)**2), dim=len(output_two_step_bone_pairs_a_difference.shape)-1)
                    loss_b_two_step_bone_pairs_batch = torch.mean((((output_two_step_bone_bi-output_two_step_bone_bj)-(target_two_step_bone_bi-target_two_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)

                loss_a_two_step_bone_pairs = torch.mean(loss_a_two_step_bone_pairs_batch)
                loss_b_two_step_bone_pairs = torch.mean(loss_b_two_step_bone_pairs_batch)

                loss_two_step_neighbourBoneAngle_batch = loss_a_two_step_bone_pairs_batch + loss_b_two_step_bone_pairs_batch
                loss_two_step_neighbourBoneAngle = loss_a_two_step_bone_pairs + loss_b_two_step_bone_pairs

        if opt.threeStepBoneAngle_penalty:
            three_step_neighbour_bone_pairs = [(0, 5), (0, 8), (0, 10), (0, 13), (1, 4), (1, 7), (2, 3), (2, 6), (3, 8), (3, 10), (3, 13), (4, 7), (5, 6), (6, 9), (6, 11), (6, 14), (7, 12), (7, 15), (8, 12), (8, 15), (9, 11), (9, 14), (10, 15), (11, 14), (12, 13)]
            three_step_bone_pairs_diff_direction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]
            output_three_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            target_three_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            output_three_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            target_three_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()

            output_three_step_bone_bi = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            output_three_step_bone_bj = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            target_three_step_bone_bi = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            target_three_step_bone_bj = torch.zeros(N,output_3D.size(1),len(three_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(three_step_neighbour_bone_pairs): #cosa
                    output_three_step_neighbour_bone_angles[:, :, n] = ((three_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_three_step_neighbour_bone_angles[:, :, n] = ((three_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_three_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_three_step_neighbour_bone_angles - target_three_step_neighbour_bone_angles), dim=len(output_three_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_three_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_three_step_neighbour_bone_angles - target_three_step_neighbour_bone_angles)**2), dim=len(output_three_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_three_step_neighbourBoneAngle_cosa = torch.mean(loss_three_step_neighbourBoneAngle_batch_cosa)

            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(three_step_neighbour_bone_pairs):
                    output_three_step_bone_pairs_a_difference[:, :, n] = ((three_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_three_step_bone_pairs_a_difference[:, :, n] = ((three_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_three_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_three_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_three_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_three_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()

                if opt.error_rule == 'F1':
                    loss_a_three_step_bone_pairs_batch = torch.mean(torch.abs((output_three_step_bone_pairs_a_difference - target_three_step_bone_pairs_a_difference)), dim=len(output_three_step_bone_pairs_a_difference.shape)-1)
                    loss_b_three_step_bone_pairs_batch = torch.mean(torch.abs(((output_three_step_bone_bi-output_three_step_bone_bj)-(target_three_step_bone_bi-target_three_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_three_step_bone_pairs_batch = torch.mean(((output_three_step_bone_pairs_a_difference - target_three_step_bone_pairs_a_difference)**2), dim=len(output_three_step_bone_pairs_a_difference.shape)-1)
                    loss_b_three_step_bone_pairs_batch = torch.mean((((output_three_step_bone_bi-output_three_step_bone_bj)-(target_three_step_bone_bi-target_three_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)
                loss_a_three_step_bone_pairs = torch.mean(loss_a_three_step_bone_pairs_batch)
                loss_b_three_step_bone_pairs = torch.mean(loss_b_three_step_bone_pairs_batch)
                loss_three_step_neighbourBoneAngle_batch = loss_a_three_step_bone_pairs_batch + loss_b_three_step_bone_pairs_batch
                loss_three_step_neighbourBoneAngle = loss_a_three_step_bone_pairs + loss_b_three_step_bone_pairs

        if opt.fourStepBoneAngle_penalty:
            four_step_neighbour_bone_pairs = [(0, 9), (0, 11), (0, 14), (1, 5), (1, 8), (1, 10), (1, 13), (2, 4), (2, 7), (3, 9), (3, 11), (3, 14), (4, 8), (4, 10), (4, 13), (5, 7), (6, 12), (6, 15), (9, 12), (9, 15), (11, 15), (12, 14)]
            four_step_bone_pairs_diff_direction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
            output_four_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            target_four_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            output_four_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            target_four_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()

            output_four_step_bone_bi = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            output_four_step_bone_bj = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            target_four_step_bone_bi = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            target_four_step_bone_bj = torch.zeros(N,output_3D.size(1),len(four_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(four_step_neighbour_bone_pairs): #cosa
                    output_four_step_neighbour_bone_angles[:, :, n] = ((four_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_four_step_neighbour_bone_angles[:, :, n] = ((four_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_four_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_four_step_neighbour_bone_angles - target_four_step_neighbour_bone_angles), dim=len(output_four_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_four_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_four_step_neighbour_bone_angles - target_four_step_neighbour_bone_angles)**2), dim=len(output_four_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_four_step_neighbourBoneAngle_cosa = torch.mean(loss_four_step_neighbourBoneAngle_batch_cosa)

            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(four_step_neighbour_bone_pairs):
                    output_four_step_bone_pairs_a_difference[:, :, n] = ((four_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_four_step_bone_pairs_a_difference[:, :, n] = ((four_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_four_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_four_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_four_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_four_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_four_step_bone_pairs_batch = torch.mean(torch.abs((output_four_step_bone_pairs_a_difference - target_four_step_bone_pairs_a_difference)), dim=len(output_four_step_bone_pairs_a_difference.shape)-1)
                    loss_b_four_step_bone_pairs_batch = torch.mean(torch.abs(((output_four_step_bone_bi-output_four_step_bone_bj)-(target_four_step_bone_bi-target_four_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_four_step_bone_pairs_batch = torch.mean(((output_four_step_bone_pairs_a_difference - target_four_step_bone_pairs_a_difference)**2), dim=len(output_four_step_bone_pairs_a_difference.shape)-1)
                    loss_b_four_step_bone_pairs_batch = torch.mean((((output_four_step_bone_bi-output_four_step_bone_bj)-(target_four_step_bone_bi-target_four_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)
                loss_a_four_step_bone_pairs = torch.mean(loss_a_four_step_bone_pairs_batch)
                loss_b_four_step_bone_pairs = torch.mean(loss_b_four_step_bone_pairs_batch)

                loss_four_step_neighbourBoneAngle_batch = loss_a_four_step_bone_pairs_batch + loss_b_four_step_bone_pairs_batch
                loss_four_step_neighbourBoneAngle = loss_a_four_step_bone_pairs + loss_b_four_step_bone_pairs

        if opt.fiveStepBoneAngle_penalty:
            five_step_neighbour_bone_pairs = [(0, 12), (0, 15), (1, 9), (1, 11), (1, 14), (2, 5), (2, 8), (2, 10), (2, 13), (3, 12), (3, 15), (4, 9), (4, 12), (4, 14), (5, 8), (5, 10), (5, 13), (12, 15)]
            five_step_bone_pairs_diff_direction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            output_five_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            target_five_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            output_five_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            target_five_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()

            output_five_step_bone_bi = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            output_five_step_bone_bj = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            target_five_step_bone_bi = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            target_five_step_bone_bj = torch.zeros(N,output_3D.size(1),len(five_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(five_step_neighbour_bone_pairs): #cosa
                    output_five_step_neighbour_bone_angles[:, :, n] = ((five_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_five_step_neighbour_bone_angles[:, :, n] = ((five_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_five_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_five_step_neighbour_bone_angles - target_five_step_neighbour_bone_angles), dim=len(output_five_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_five_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_five_step_neighbour_bone_angles - target_five_step_neighbour_bone_angles)**2), dim=len(output_five_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_five_step_neighbourBoneAngle_cosa = torch.mean(loss_five_step_neighbourBoneAngle_batch_cosa)
            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(five_step_neighbour_bone_pairs):
                    output_five_step_bone_pairs_a_difference[:, :, n] = ((five_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_five_step_bone_pairs_a_difference[:, :, n] = ((five_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_five_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_five_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_five_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_five_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_five_step_bone_pairs_batch = torch.mean(torch.abs((output_five_step_bone_pairs_a_difference - target_five_step_bone_pairs_a_difference)), dim=len(output_five_step_bone_pairs_a_difference.shape)-1)
                    loss_b_five_step_bone_pairs_batch = torch.mean(torch.abs(((output_five_step_bone_bi-output_five_step_bone_bj)-(target_five_step_bone_bi-target_five_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_five_step_bone_pairs_batch = torch.mean(((output_five_step_bone_pairs_a_difference - target_five_step_bone_pairs_a_difference)**2), dim=len(output_five_step_bone_pairs_a_difference.shape)-1)
                    loss_b_five_step_bone_pairs_batch = torch.mean((((output_five_step_bone_bi-output_five_step_bone_bj)-(target_five_step_bone_bi-target_five_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)
                loss_a_five_step_bone_pairs = torch.mean(loss_a_five_step_bone_pairs_batch)
                loss_b_five_step_bone_pairs = torch.mean(loss_b_five_step_bone_pairs_batch)

                loss_five_step_neighbourBoneAngle_batch = loss_a_five_step_bone_pairs_batch + loss_b_five_step_bone_pairs_batch
                loss_five_step_neighbourBoneAngle = loss_a_five_step_bone_pairs + loss_b_five_step_bone_pairs

        if opt.sixStepBoneAngle_penalty:
            six_step_neighbour_bone_pairs = [(1, 12), (1, 15), (2, 9), (2, 11), (2, 14), (4, 12), (4, 15), (5, 9), (5, 11), (5, 14)]
            six_step_bone_pairs_diff_direction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            output_six_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            target_six_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            output_six_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            target_six_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()

            output_six_step_bone_bi = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            output_six_step_bone_bj = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            target_six_step_bone_bi = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            target_six_step_bone_bj = torch.zeros(N,output_3D.size(1),len(six_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(six_step_neighbour_bone_pairs): #cosa
                    output_six_step_neighbour_bone_angles[:, :, n] = ((six_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_six_step_neighbour_bone_angles[:, :, n] = ((six_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_six_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_six_step_neighbour_bone_angles - target_six_step_neighbour_bone_angles), dim=len(output_six_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_six_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_six_step_neighbour_bone_angles - target_six_step_neighbour_bone_angles)**2), dim=len(output_six_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_six_step_neighbourBoneAngle_cosa = torch.mean(loss_six_step_neighbourBoneAngle_batch_cosa)
            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(six_step_neighbour_bone_pairs):
                    output_six_step_bone_pairs_a_difference[:, :, n] = ((six_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_six_step_bone_pairs_a_difference[:, :, n] = ((six_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_six_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_six_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_six_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_six_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_six_step_bone_pairs_batch = torch.mean(torch.abs((output_six_step_bone_pairs_a_difference - target_six_step_bone_pairs_a_difference)), dim=len(output_six_step_bone_pairs_a_difference.shape)-1)
                    loss_b_six_step_bone_pairs_batch = torch.mean(torch.abs(((output_six_step_bone_bi-output_six_step_bone_bj)-(target_six_step_bone_bi-target_six_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_six_step_bone_pairs_batch = torch.mean(((output_six_step_bone_pairs_a_difference - target_six_step_bone_pairs_a_difference)**2), dim=len(output_six_step_bone_pairs_a_difference.shape)-1)
                    loss_b_six_step_bone_pairs_batch = torch.mean((((output_six_step_bone_bi-output_six_step_bone_bj)-(target_six_step_bone_bi-target_six_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)
                loss_a_six_step_bone_pairs = torch.mean(loss_a_six_step_bone_pairs_batch)
                loss_b_six_step_bone_pairs = torch.mean(loss_b_six_step_bone_pairs_batch)

                loss_six_step_neighbourBoneAngle_batch = loss_a_six_step_bone_pairs_batch + loss_b_six_step_bone_pairs_batch
                loss_six_step_neighbourBoneAngle = loss_a_six_step_bone_pairs + loss_b_six_step_bone_pairs

        if opt.sevenStepBoneAngle_penalty:
            seven_step_neighbour_bone_pairs = [(2, 12), (2, 15), (5, 12), (5, 15)]
            seven_step_bone_pairs_diff_direction = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            output_seven_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            target_seven_step_neighbour_bone_angles = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            output_seven_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            target_seven_step_bone_pairs_a_difference = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()

            output_seven_step_bone_bi = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            output_seven_step_bone_bj = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            target_seven_step_bone_bi = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            target_seven_step_bone_bj = torch.zeros(N,output_3D.size(1),len(seven_step_neighbour_bone_pairs)).cuda()
            if opt.boneAngle_define == 'vector_product_cosa':
                for n, (i, j) in enumerate(seven_step_neighbour_bone_pairs): #cosa
                    output_seven_step_neighbour_bone_angles[:, :, n] = ((seven_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(output_bone_vector[:,:,i,:], output_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(output_bone_vector[:,:,i,:], dim=len(output_bone_vector.shape)-2) * torch.norm(output_bone_vector[:,:,j,:], dim=len(output_bone_vector.shape)-2)))
                    target_seven_step_neighbour_bone_angles[:, :, n] = ((seven_step_bone_pairs_diff_direction[n] * torch.sum(torch.mul(target_bone_vector[:,:,i,:], target_bone_vector[:,:,j,:]), dim=(2,)))/(torch.norm(target_bone_vector[:,:,i,:], dim=len(target_bone_vector.shape)-2) * torch.norm(target_bone_vector[:,:,j,:], dim=len(target_bone_vector.shape)-2)))
                if opt.error_rule == 'F1':
                    loss_seven_step_neighbourBoneAngle_batch_cosa = torch.mean(torch.abs(output_seven_step_neighbour_bone_angles - target_seven_step_neighbour_bone_angles), dim=len(output_seven_step_neighbour_bone_angles.shape)-1 ).cuda()
                elif opt.error_rule == 'F2':
                    loss_seven_step_neighbourBoneAngle_batch_cosa = torch.mean(((output_seven_step_neighbour_bone_angles - target_seven_step_neighbour_bone_angles)**2), dim=len(output_seven_step_neighbour_bone_angles.shape)-1 ).cuda()
                loss_seven_step_neighbourBoneAngle_cosa = torch.mean(loss_seven_step_neighbourBoneAngle_batch_cosa)
            if opt.boneAngle_define == 'cosa_|1+cosb|':
                for n, (i, j) in enumerate(seven_step_neighbour_bone_pairs):
                    output_seven_step_bone_pairs_a_difference[:, :, n] = ((seven_step_bone_pairs_diff_direction[n]*output_bone_vector[:, :, i, 2]/(torch.norm(output_bone_vector[:, :, i, :], dim=len(output_bone_vector.shape)-2))) - (output_bone_vector[:, :, j, 2]/(torch.norm(output_bone_vector[:, :, j, :], dim=len(output_bone_vector.shape)-2)))).cuda()
                    target_seven_step_bone_pairs_a_difference[:, :, n] = ((seven_step_bone_pairs_diff_direction[n]*target_bone_vector[:, :, i, 2]/(torch.norm(target_bone_vector[:, :, i, :], dim=len(target_bone_vector.shape)-2))) - (target_bone_vector[:, :, j, 2]/(torch.norm(target_bone_vector[:, :, j, :], dim=len(target_bone_vector.shape)-2)))).cuda()

                    output_seven_step_bone_bi[:, :, n] = (output_x_pos_or_neg[:, :, i] * (1+(output_bone_vector[:, :, i, 1] / torch.sqrt(output_bone_vector[:, :, i, 0]**2 + output_bone_vector[:, :, i, 1]**2)))).cuda()
                    output_seven_step_bone_bj[:, :, n] = (output_x_pos_or_neg[:, :, j] * (1+(output_bone_vector[:, :, j, 1] / torch.sqrt(output_bone_vector[:, :, j, 0]**2 + output_bone_vector[:, :, j, 1]**2)))).cuda()
                    target_seven_step_bone_bi[:, :, n] = (target_x_pos_or_neg[:, :, i] * (1+(target_bone_vector[:, :, i, 1] / torch.sqrt(target_bone_vector[:, :, i, 0]**2 + target_bone_vector[:, :, i, 1]**2)))).cuda()
                    target_seven_step_bone_bj[:, :, n] = (target_x_pos_or_neg[:, :, j] * (1+(target_bone_vector[:, :, j, 1] / torch.sqrt(target_bone_vector[:, :, j, 0]**2 + target_bone_vector[:, :, j, 1]**2)))).cuda()
                if opt.error_rule == 'F1':
                    loss_a_seven_step_bone_pairs_batch = torch.mean(torch.abs((output_seven_step_bone_pairs_a_difference - target_seven_step_bone_pairs_a_difference)), dim=len(output_seven_step_bone_pairs_a_difference.shape)-1)
                    loss_b_seven_step_bone_pairs_batch = torch.mean(torch.abs(((output_seven_step_bone_bi-output_seven_step_bone_bj)-(target_seven_step_bone_bi-target_seven_step_bone_bj))), dim=len(target_bone_vector.shape) - 2)
                elif opt.error_rule == 'F2':
                    loss_a_seven_step_bone_pairs_batch = torch.mean(((output_seven_step_bone_pairs_a_difference - target_seven_step_bone_pairs_a_difference)**2), dim=len(output_seven_step_bone_pairs_a_difference.shape)-1)
                    loss_b_seven_step_bone_pairs_batch = torch.mean((((output_seven_step_bone_bi-output_seven_step_bone_bj)-(target_seven_step_bone_bi-target_seven_step_bone_bj))**2), dim=len(target_bone_vector.shape) - 2)
                loss_a_seven_step_bone_pairs = torch.mean(loss_a_seven_step_bone_pairs_batch)
                loss_b_seven_step_bone_pairs = torch.mean(loss_b_seven_step_bone_pairs_batch)

                loss_seven_step_neighbourBoneAngle_batch = loss_a_seven_step_bone_pairs_batch + loss_b_seven_step_bone_pairs_batch
                loss_seven_step_neighbourBoneAngle = loss_a_seven_step_bone_pairs + loss_b_seven_step_bone_pairs

        if opt.boneAngle_define == 'cosa_|1+cosb|':
            if opt.boneLength_penalty==0 and opt.boneAngle_penalty==0 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==0 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==1 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle + opt.co_sixstep_neighbourBoneAngle * loss_six_step_neighbourBoneAngle
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==1 and opt.sevenStepBoneAngle_penalty==1:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle + opt.co_sixstep_neighbourBoneAngle * loss_six_step_neighbourBoneAngle + opt.co_sevenstep_neighbourBoneAngle * loss_seven_step_neighbourBoneAngle

        if opt.boneAngle_define == 'vector_product_cosa':
            if opt.boneLength_penalty==0 and opt.boneAngle_penalty==0 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==0 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==0 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==0 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==0 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==0 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==0 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle_cosa + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==0 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle_cosa + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle_cosa + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==1 and opt.sevenStepBoneAngle_penalty==0:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle_cosa + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle_cosa + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle_cosa + opt.co_sixstep_neighbourBoneAngle * loss_six_step_neighbourBoneAngle_cosa
            if opt.boneLength_penalty==1 and opt.boneAngle_penalty==1 and opt.neighbourBoneAngle_penalty==1 and opt.twoStepBoneAngle_penalty==1 and opt.threeStepBoneAngle_penalty==1 and opt.fourStepBoneAngle_penalty==1 and opt.fiveStepBoneAngle_penalty==1 and opt.sixStepBoneAngle_penalty==1 and opt.sevenStepBoneAngle_penalty==1:
                loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine + opt.co_boneLength * loss_boneLength + opt.co_boneAngle * loss_boneAngle_cosa + opt.co_neighbourBoneAngle * loss_neighbourBoneAngle_cosa+ opt.co_twostep_neighbourBoneAngle * loss_two_step_neighbourBoneAngle_cosa + opt.co_threestep_neighbourBoneAngle * loss_three_step_neighbourBoneAngle_cosa + opt.co_fourstep_neighbourBoneAngle * loss_four_step_neighbourBoneAngle_cosa + opt.co_fivestep_neighbourBoneAngle * loss_five_step_neighbourBoneAngle_cosa + opt.co_sixstep_neighbourBoneAngle * loss_six_step_neighbourBoneAngle_cosa + opt.co_sevenstep_neighbourBoneAngle * loss_seven_step_neighbourBoneAngle_cosa

        loss_all_sum['loss_gt'].update(loss_gt.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_diff'].update(loss_diff.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_sum'].update(loss.detach().cpu().numpy() * N, N)
        if opt.boneLength_penalty:
            loss_all_sum['loss_boneLength'].update(loss_boneLength.detach().cpu().numpy() * N, N)
        if opt.boneAngle_define == 'cosa_|1+cosb|':
            if opt.boneAngle_penalty:
                loss_all_sum['loss_boneAngle'].update(loss_boneAngle.detach().cpu().numpy() * N, N)
            if opt.neighbourBoneAngle_penalty:
                loss_all_sum['loss_neighbourBoneAngle'].update(loss_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_neighbourBoneAngle_a'].update(loss_a_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_neighbourBoneAngle_b'].update(loss_b_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.twoStepBoneAngle_penalty:
                loss_all_sum['loss_two_step_neighbourBoneAngle'].update(loss_two_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_two_step_neighbourBoneAngle_a'].update(loss_a_two_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_two_step_neighbourBoneAngle_b'].update(loss_b_two_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.threeStepBoneAngle_penalty:
                loss_all_sum['loss_three_step_neighbourBoneAngle'].update(loss_three_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_three_step_neighbourBoneAngle_a'].update(loss_a_three_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_three_step_neighbourBoneAngle_b'].update(loss_b_three_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.fourStepBoneAngle_penalty:
                loss_all_sum['loss_four_step_neighbourBoneAngle'].update(loss_four_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_four_step_neighbourBoneAngle_a'].update(loss_a_four_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_four_step_neighbourBoneAngle_b'].update(loss_b_four_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.fiveStepBoneAngle_penalty:
                loss_all_sum['loss_five_step_neighbourBoneAngle'].update(loss_five_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_five_step_neighbourBoneAngle_a'].update(loss_a_five_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_five_step_neighbourBoneAngle_b'].update(loss_b_five_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.sixStepBoneAngle_penalty:
                loss_all_sum['loss_six_step_neighbourBoneAngle'].update(loss_six_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_six_step_neighbourBoneAngle_a'].update(loss_a_six_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_six_step_neighbourBoneAngle_b'].update(loss_b_six_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.sevenStepBoneAngle_penalty:
                loss_all_sum['loss_seven_step_neighbourBoneAngle'].update(loss_seven_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_seven_step_neighbourBoneAngle_a'].update(loss_a_seven_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_seven_step_neighbourBoneAngle_b'].update(loss_b_seven_step_bone_pairs.detach().cpu().numpy() * N, N)
        if opt.boneAngle_define == 'vector_product_cosa':
            if opt.boneAngle_penalty:
                loss_all_sum['loss_boneAngle_cosa'].update(loss_boneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_boneAngle'].update(loss_boneAngle.detach().cpu().numpy() * N, N)
            if opt.neighbourBoneAngle_penalty:
                loss_all_sum['loss_neighbourBoneAngle_cosa'].update(loss_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_neighbourBoneAngle'].update(loss_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_neighbourBoneAngle_a'].update(loss_a_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_neighbourBoneAngle_b'].update(loss_b_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.twoStepBoneAngle_penalty:
                loss_all_sum['loss_two_step_neighbourBoneAngle_cosa'].update(loss_two_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_two_step_neighbourBoneAngle'].update(loss_two_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_two_step_neighbourBoneAngle_a'].update(loss_a_two_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_two_step_neighbourBoneAngle_b'].update(loss_b_two_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.threeStepBoneAngle_penalty:
                loss_all_sum['loss_three_step_neighbourBoneAngle_cosa'].update(loss_three_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_three_step_neighbourBoneAngle'].update(loss_three_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_three_step_neighbourBoneAngle_a'].update(loss_a_three_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_three_step_neighbourBoneAngle_b'].update(loss_b_three_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.fourStepBoneAngle_penalty:
                loss_all_sum['loss_four_step_neighbourBoneAngle_cosa'].update(loss_four_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_four_step_neighbourBoneAngle'].update(loss_four_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_four_step_neighbourBoneAngle_a'].update(loss_a_four_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_four_step_neighbourBoneAngle_b'].update(loss_b_four_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.fiveStepBoneAngle_penalty:
                loss_all_sum['loss_five_step_neighbourBoneAngle_cosa'].update(loss_five_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_five_step_neighbourBoneAngle'].update(loss_five_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_five_step_neighbourBoneAngle_a'].update(loss_a_five_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_five_step_neighbourBoneAngle_b'].update(loss_b_five_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.sixStepBoneAngle_penalty:
                loss_all_sum['loss_six_step_neighbourBoneAngle_cosa'].update(loss_six_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_six_step_neighbourBoneAngle'].update(loss_six_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_six_step_neighbourBoneAngle_a'].update(loss_a_six_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_six_step_neighbourBoneAngle_b'].update(loss_b_six_step_bone_pairs.detach().cpu().numpy() * N, N)
            if opt.sevenStepBoneAngle_penalty:
                loss_all_sum['loss_seven_step_neighbourBoneAngle_cosa'].update(loss_seven_step_neighbourBoneAngle_cosa.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_seven_step_neighbourBoneAngle'].update(loss_seven_step_neighbourBoneAngle.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_seven_step_neighbourBoneAngle_a'].update(loss_a_seven_step_bone_pairs.detach().cpu().numpy() * N, N)
                loss_all_sum['loss_seven_step_neighbourBoneAngle_b'].update(loss_b_seven_step_bone_pairs.detach().cpu().numpy() * N, N)
        # train backpropogation
        if split == 'train':  #for each batchsize
            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()
            pred_out[:, :, 0, :] = 0
            joint_error = eval_cal.mpjpe(pred_out, out_target).item()
            error_sum.update(joint_error*N, N)



        elif split == 'test':
            pred_out[:, :, 0, :] = 0  #(512,1,17,3)
            joint_error = eval_cal.mpjpe(pred_out, out_target).item()
            error_sum.update(joint_error*N, N)
            action_error_sum = eval_cal.test_calculation(pred_out, out_target, action, action_error_sum,
                                                         opt.dataset, show_protocol2=opt.show_protocol2)

            if opt.post_refine:
                post_out[:, :, 0, :] = 0
                action_error_sum_post_out = eval_cal.test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset,
                                                             show_protocol2=opt.show_protocol2)


    mean_error['xyz'] = error_sum.avg
    print('loss gt each frame of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg))
    print('loss diff of 1 sample: %f' % (loss_all_sum['loss_diff'].avg))
    print('loss of 1 sample: %f' % (loss_all_sum['loss_sum'].avg))
    print('mean joint error: %f' % (mean_error['xyz']*1000))
    if opt.boneLength_penalty:
        print('loss boneLength each frame of 1 sample: %f mm' % (loss_all_sum['loss_boneLength'].avg))
    if opt.boneAngle_penalty:
        print('loss boneAngle each frame of 1 sample: %f mm' % (loss_all_sum['loss_boneAngle'].avg))
    if opt.neighbourBoneAngle_penalty:
        print('loss neighbourBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_neighbourBoneAngle'].avg))
        print('loss neighbourBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_neighbourBoneAngle_a'].avg))
        print('loss neighbourBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_neighbourBoneAngle_b'].avg))
    if opt.twoStepBoneAngle_penalty:
        print('loss 2-step neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_two_step_neighbourBoneAngle'].avg))
        print('loss 2-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_two_step_neighbourBoneAngle_a'].avg))
        print('loss 2-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_two_step_neighbourBoneAngle_b'].avg))
    if opt.threeStepBoneAngle_penalty:
        print('loss 3-step-neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_three_step_neighbourBoneAngle'].avg))
        print('loss 3-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_three_step_neighbourBoneAngle_a'].avg))
        print('loss 3-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_three_step_neighbourBoneAngle_b'].avg))
    if opt.fourStepBoneAngle_penalty:
        print('loss 4-step-neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_four_step_neighbourBoneAngle'].avg))
        print('loss 4-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_four_step_neighbourBoneAngle_a'].avg))
        print('loss 4-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_four_step_neighbourBoneAngle_b'].avg))
    if opt.fiveStepBoneAngle_penalty:
        print('loss 5-step-neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_five_step_neighbourBoneAngle'].avg))
        print('loss 5-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_five_step_neighbourBoneAngle_a'].avg))
        print('loss 5-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_five_step_neighbourBoneAngle_b'].avg))
    if opt.sixStepBoneAngle_penalty:
        print('loss 6-step-neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_six_step_neighbourBoneAngle'].avg))
        print('loss 6-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_six_step_neighbourBoneAngle_a'].avg))
        print('loss 6-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_six_step_neighbourBoneAngle_b'].avg))
    if opt.sevenStepBoneAngle_penalty:
        print('loss 7-step-neighBoneAngle each frame of 1 sample: %f ' % (loss_all_sum['loss_seven_step_neighbourBoneAngle'].avg))
        print('loss 7-step neighBoneAngle_a each frame of 1 sample: %f ' % (loss_all_sum['loss_seven_step_neighbourBoneAngle_a'].avg))
        print('loss 7-step neighBoneAngle_b each frame of 1 sample: %f ' % (loss_all_sum['loss_seven_step_neighbourBoneAngle_b'].avg))
    if opt.boneAngle_define == 'vector_product_cosa':
        if opt.boneAngle_penalty:
            print('loss boneAngle_cosa each frame of 1 sample: %f mm' % (loss_all_sum['loss_boneAngle_cosa'].avg))
        if opt.neighbourBoneAngle_penalty:
            print('loss neighbourBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_neighbourBoneAngle_cosa'].avg))
        if opt.twoStepBoneAngle_penalty:
            print('loss 2-step neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_two_step_neighbourBoneAngle_cosa'].avg))
        if opt.threeStepBoneAngle_penalty:
            print('loss 3-step-neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_three_step_neighbourBoneAngle_cosa'].avg))
        if opt.fourStepBoneAngle_penalty:
            print('loss 4-step-neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_four_step_neighbourBoneAngle_cosa'].avg))
        if opt.fiveStepBoneAngle_penalty:
            print('loss 5-step-neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_five_step_neighbourBoneAngle_cosa'].avg))
        if opt.sixStepBoneAngle_penalty:
            print('loss 6-step-neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_six_step_neighbourBoneAngle_cosa'].avg))
        if opt.sevenStepBoneAngle_penalty:
            print('loss 7-step-neighBoneAngle_cosa each frame of 1 sample: %f ' % (loss_all_sum['loss_seven_step_neighbourBoneAngle_cosa'].avg))
    if split == 'test':
        if not opt.post_refine:
            mean_error_all = print_error(opt.dataset, action_error_sum, opt.show_protocol2) #54.56790791648849
            mean_error['xyz'] = mean_error_all

        elif opt.post_refine:
            print('-----post out')
            mean_error_all = print_error(opt.dataset, action_error_sum_post_out,  opt.show_protocol2)
            mean_error['post'] = mean_error_all

        if opt.show_boneError:
            mean_bone_error_all = print_boneError(action_error_sum)

    return mean_error


def train(opt, actions, train_loader, model, criterion, optimizer):
     return step('train',  opt, actions, train_loader, model, criterion, optimizer)


def val(opt, actions, val_loader, model, criterion):
    return step('test',  opt, actions, val_loader, model, criterion)

def input_augmentation(input_2D, model_st_gcn, joints_left, joints_right):
    """
    for calculating augmentation results
    :param input_2D:
    :param model_st_gcn:
    :param joints_left: joint index of left part
    :param joints_right: joint index of right part
    :return:
    """
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M

    # flip and reverse to original xyz
    output_3D_flip = model_st_gcn(input_2D_flip, out_all_frame=False)
    output_3D_flip[:, 0] *= -1
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]
    output_3D_non_flip = model_st_gcn(input_2D_non_flip, out_all_frame=False)

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D