This is the code for the paper: PMSGCN: parallel multi-scale graph convolution network for estimating perceptually similar 3D human poses from monocular images in Pytorch.

Dependencies:
	cuda 9.0
	Python 3.6
	Pytorch 0.4.1.
	matplotlib==3.1.1 
	opencv-python==4.1.1.26
	tqdm==4.46.0

Argument adjustment in opt1.py:
	--root_path: change to the path where this project is stored on the server
	--input_inverse_intrinsic：If decoupling the camera intrinsic parameters from PMSGCN, it equals to True and corresponding --in_channels equals to 3; If not decoupling the parameters, it is false and corresponding --in_channels equals to 2.
	--use_projected_2dgt：default value is False. If it is True, then PMSGCN uses the 2D poses projected from the 3D labels as the network input.

Dataset setup:
2D pose detections and corresponding 3D labels are put in data/dataset which can be downloaded from:
https://drive.google.com/drive/folders/1r8cz9abdru6YRZVOGWjQ1vwsW10D3v62?usp=sharing

To train the PMSGCN, run:
python main_graph.py  --show_protocol2

To test the PMSGCN, run:
python main_graph.py --pro_train 0 --show_protocol2 --stgcn_reload 1 --previous_dir ‘/PMSGCN_SSE/PMSGCN/results/pms_gcn/no_pose_refine/ --stgcn_model 'model_pms_gcn_xx_eva_post_xxxx.pth’
