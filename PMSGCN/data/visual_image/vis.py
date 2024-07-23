import numpy as np
import scipy.ndimage
import skimage.transform
import cv2
import os
import torch
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data.visual_image.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image

CONNECTIVITY_DICT = {
    "human36m": [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 14), (14, 15), (15, 16), (8, 11), (11, 12), (12, 13)],
}

COLOR_DICT = {
    'human36m': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ]}

JOINT_NAMES_DICT = {
    'coco': {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow",
        8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}
}

def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)  #nd(2500,2000,4)

    return fig_image


def visualize_3d_gt_pred( keypoints_3d_gt_batch,
                    keypoints_3d_pred_batch,
                    kind="human36m",
                    batch_index=0,
                    ):

    connectivity = CONNECTIVITY_DICT[kind]
    keypoints_3dgt = to_numpy(keypoints_3d_gt_batch)[batch_index] #(512,1,17,3)[]=(1,17,2)
    keypoints_3dpred = to_numpy(keypoints_3d_pred_batch)[batch_index]

    keypoints_gt = keypoints_3dgt[0]  #(17,3)
    keypoints_pred = keypoints_3dpred[0]

    max_x_gt = max(keypoints_gt[:, 0]); min_x_gt = min(keypoints_gt[:, 0])
    max_y_gt = max(keypoints_gt[:, 1]); min_y_gt = min(keypoints_gt[:, 1])
    max_z_gt = max(keypoints_gt[:, 2]); min_z_gt = min(keypoints_gt[:, 2])
    max_gt = max(max_x_gt, max_y_gt, max_z_gt); min_gt =min(min_x_gt, min_y_gt, min_z_gt)

    max_x_pred = max(keypoints_pred[:, 0]); min_x_pred = min(keypoints_pred[:, 0])
    max_y_pred = max(keypoints_pred[:, 1]); min_y_pred = min(keypoints_pred[:, 1])
    max_z_pred = max(keypoints_pred[:, 2]); min_z_pred = min(keypoints_pred[:, 2])
    max_pred = max(max_x_pred, max_y_pred, max_z_pred); min_pred =min(min_x_pred, min_y_pred, min_z_pred)

    fig = plt.figure(figsize=(32, 16))
    ax = fig.add_subplot(121, projection='3d')
    # fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(1 * 12, 2 * 12))
    #  axes = axes.reshape(2, 1)

    #draw 3d gt skeleton graph
    ax.scatter(keypoints_gt[:, 0], keypoints_gt[:, 2], keypoints_gt[:, 1], c=np.array([230, 145, 56])/255, s=40, edgecolors='black')
   # ax.set_title('3d_gt_keypoints')
   # ax.set_xlabel('X')
   # ax.set_ylabel('Z') #垂直于平面的表示姿态的深度Z
   # ax.set_zlabel('Y') #竖直的设为Y

    ax.set_xlim([min_gt, max_gt]) #range max,min
    ax.set_ylim([min_gt, max_gt])  #min max
    ax.set_zlim([max_gt, min_gt])  #max min

    #  background_color = np.array([252, 252, 252]) / 255
    #  ax.w_xaxis.set_pane_color(background_color) #三个背景面颜色设置
    #  ax.w_yaxis.set_pane_color(background_color)
    #  ax.w_zaxis.set_pane_color(background_color)

    ax.set_xticklabels([])  #坐标轴标签设置
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    for i, joint in enumerate(connectivity):
        xs = np.array([keypoints_gt[joint[0], 0], keypoints_gt[joint[1], 0]])  #st-gcn论文纸后面画图，现实中成像面在焦点另一侧，X应为负的，YZ为正的。
        ys = np.array([keypoints_gt[joint[0], 2], keypoints_gt[joint[1], 2]])
        zs = np.array([keypoints_gt[joint[0], 1], keypoints_gt[joint[1], 1]])
        if kind in COLOR_DICT:
            color = COLOR_DICT[kind][i]
        else:
            color = (0, 0, 255)
        color = np.array(color) / 255
        ax.plot(xs, ys, zs, lw=4, c=color) #line width =

    #draw 3d pred skeleton graph
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(keypoints_pred[:, 0], keypoints_pred[:, 2], keypoints_pred[:, 1], c=np.array([230, 145, 56])/255, s=40, edgecolors='black')
 #   ax2.set_title('3d_pred_keypoints')

    ax2.set_xlim([min_pred, max_pred]) #range
    ax2.set_ylim([min_pred, max_pred])
    ax2.set_zlim([max_pred, min_pred])

    ax2.set_xticklabels([])  #坐标轴标签设置
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    for i, joint in enumerate(connectivity):
        xs = np.array([keypoints_pred[joint[0], 0], keypoints_pred[joint[1], 0]])  #st-gcn论文纸后面画图，现实中成像面在焦点另一侧，X应为负的，YZ为正的。
        ys, zs = [np.array([keypoints_pred[joint[0], j], keypoints_pred[joint[1], j]]) for j in (2, 1)]
        if kind in COLOR_DICT:
            color = COLOR_DICT[kind][i]
        else:
            color = (0, 0, 255)
        color = np.array(color) / 255
        ax2.plot(xs, ys, zs, lw=4, c=color)
    plt.show()
    fig.savefig(os.path.join('/home/chuanjiang/Projects/1view_3frames_originCode2_elu_noBN_SSE/', str('7_3390_' + str(batch_index) + '.png')), dpi=100)

    plt.close()

def visualize_volumes(images_batch, volumes_batch, proj_matricies_batch,
                      kind="cmu",
                      cuboids_batch=None,
                      batch_index=0, size=5,
                      max_n_rows=10, max_n_cols=10):
    n_views, n_joints = volumes_batch.shape[1], volumes_batch.shape[2]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig = plt.figure(figsize=(n_cols * size, n_rows * size))

    # images
    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    volumes = to_numpy(volumes_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                ax.set_ylabel(str(row), size='large')

                cuboid = cuboids_batch[batch_index]
                ax.imshow(cuboid.render(proj_matricies_batch[batch_index, row].detach().cpu().numpy(), images[row].copy()))
            else:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')

                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    ax.set_title(joint_name)

                draw_voxels(volumes[col - 1], ax, norm=True)

    fig.tight_layout()

    fig_image = fig_to_array(fig)

    plt.close('all')

    return fig_image


def draw_2d_pose(keypoints, ax, kind='cmu', keypoints_mask=None, point_size=2, line_width=1, radius=None, color=None):
    """
    Visualizes a 2d skeleton

    Args
        keypoints numpy array of shape (19, 2): pose to draw in CMU format.
        ax: matplotlib axis to draw on
    """
    connectivity = CONNECTIVITY_DICT[kind]  #[(0,1),(1,2),(2,6),(5,4),(4,3),(3,6),(6,7),(7,8),(8,16),(9,16),(8,12),(11,12),(10,11),(8,13),(13,14),(14,15)]

    color = 'blue' if color is None else color #'blue'

    if keypoints_mask is None: #true
        keypoints_mask = [True] * len(keypoints) # *17 -->[True,....,True]

    # points
    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], c='red', s=point_size) #(17,2)[]->(17,2)[]->(17,1)

    # connections
    for (index_from, index_to) in connectivity:
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            xs, ys = [np.array([keypoints[index_from, j], keypoints[index_to, j]]) for j in range(2)]
            ax.plot(xs, ys, c=color, lw=line_width)

    if radius is not None: #false
        root_keypoint_index = 0
        xroot, yroot = keypoints[root_keypoint_index, 0], keypoints[root_keypoint_index, 1]

        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])

    ax.set_aspect('equal')


def draw_2d_pose_cv2(keypoints, canvas, kind='cmu', keypoints_mask=None, point_size=2, point_color=(255, 255, 255), line_width=1, radius=None, color=None, anti_aliasing_scale=1):
    canvas = canvas.copy()

    shape = np.array(canvas.shape[:2])
    new_shape = shape * anti_aliasing_scale
    canvas = resize_image(canvas, tuple(new_shape))

    keypoints = keypoints * anti_aliasing_scale
    point_size = point_size * anti_aliasing_scale
    line_width = line_width * anti_aliasing_scale

    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # connections
    for i, (index_from, index_to) in enumerate(connectivity):
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            pt_from = tuple(np.array(keypoints[index_from, :]).astype(int))
            pt_to = tuple(np.array(keypoints[index_to, :]).astype(int))

            if kind in COLOR_DICT:
                color = COLOR_DICT[kind][i]
            else:
                color = (0, 0, 255)

            cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    if kind == 'coco':
        mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
        nose = keypoints[0, :]

        pt_from = tuple(np.array(nose).astype(int))
        pt_to = tuple(np.array(mid_collarbone).astype(int))

        if kind in COLOR_DICT:
            color = (153, 0, 51)
        else:
            color = (0, 0, 255)

        cv2.line(canvas, pt_from, pt_to, color=color, thickness=line_width)

    # points
    for pt in keypoints[keypoints_mask]:
        cv2.circle(canvas, tuple(pt.astype(int)), point_size, color=point_color, thickness=-1)

    canvas = resize_image(canvas, tuple(shape))

    return canvas


def draw_3d_pose(keypoints, ax, keypoints_mask=None, kind='cmu', radius=None, root=None, point_size=2, line_width=2, draw_connections=True):
    connectivity = CONNECTIVITY_DICT[kind]
    keypoints = keypoints[0]
    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    if draw_connections:
        # Make connection matrix
        for i, joint in enumerate(connectivity):
            if keypoints_mask[joint[0]] and  keypoints_mask[joint[1]]:
                xs, ys, zs = [np.array([keypoints[joint[0], j], keypoints[joint[1], j]]) for j in range(3)]

                if kind in COLOR_DICT:
                    color = COLOR_DICT[kind][i]
                else:
                    color = (0, 0, 255)

                color = np.array(color) / 255

                ax.plot(xs, ys, zs, lw=line_width, c=color)

    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], keypoints[keypoints_mask][:, 2],  c=np.array([230, 145, 56])/255, s=point_size, edgecolors='black')
    if radius is not None:
        if root is None:
            root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    ax.set_aspect('equal')


    # Get rid of the panes
    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def draw_voxels(voxels, ax, shape=(8, 8, 8), norm=True, alpha=0.1):
    # resize for visualization
    zoom = np.array(shape) / np.array(voxels.shape)
    voxels = skimage.transform.resize(voxels, shape, mode='constant', anti_aliasing=True)
    voxels = voxels.transpose(2, 0, 1)

    if norm and voxels.max() - voxels.min() > 0:
        voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())

    filled = np.ones(voxels.shape)

    # facecolors
    cmap = plt.get_cmap("Blues")

    facecolors_a = cmap(voxels, alpha=alpha)
    facecolors_a = facecolors_a.reshape(-1, 4)

    facecolors_hex = np.array(list(map(lambda x: matplotlib.colors.to_hex(x, keep_alpha=True), facecolors_a)))
    facecolors_hex = facecolors_hex.reshape(*voxels.shape)

    # explode voxels to perform 3d alpha rendering (https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html)
    def explode(data):
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    filled_2 = explode(filled)
    facecolors_2 = explode(facecolors_hex)

    # shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # draw voxels
    ax.voxels(x, y, z, filled_2, facecolors=facecolors_2)

    ax.set_xlabel("z"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.invert_xaxis(); ax.invert_zaxis()
