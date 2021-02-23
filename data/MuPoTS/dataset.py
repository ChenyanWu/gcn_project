import os.path as osp
import numpy as np
import math
import torch
import scipy.io as sio
import json
import copy
import transforms3d
import scipy.sparse
import cv2
from pycocotools.coco import COCO

from core.config import cfg
from graph_utils import build_coarse_graphs
from noise_utils import synthesize_pose

from smpl import SMPL
from coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox, pixel2cam
from aug_utils import affine_transform, j2d_processing, augm_params, j3d_processing

from funcs_utils import save_obj, stop
from vis import vis_3d_pose, vis_2d_pose

class MuPoTS(torch.utils.data.Dataset):
    def __init__(self, data_split='test', args=None):
        dataset_name = 'MuPoTS'
        self.data_split = 'test'
        self.num_person = cfg.num_person
        self.input_joint_name = 'human36'
        self.pred_pose_mupo_path = osp.join(cfg.data_dir, dataset_name, 'preds_2d_3d_kpt_mupots.json')

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # h36m skeleton
        self.human36_joint_num = 17
        self.human36_joints_name = (
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.joint_regressor_human36 = self.mesh_model.joint_regressor_h36m

        # change between mupo and hm36
        self.h36m2mupo = [10, 8, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6, 0, 7, 9]
        # self.h36m2mupo = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
        self.mupo2h36m = [self.h36m2mupo.index(i) for i in range(17)]

        self.datalist = self.load_data()

        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        # build graph
        self.graph_Adj, self.graph_L, self.graph_perm, self.graph_perm_reverse = \
            build_coarse_graphs(self.num_person, self.mesh_model.face, self.joint_num, self.skeleton, self.flip_pairs,
                                levels=9)

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_data(self):
        print('Load annotations of MuPoTS ')
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        print("Get annotation from " + self.pred_pose_mupo_path)
        with open(self.pred_pose_mupo_path) as f:
            annot = json.load(f)
        data = []
        for i in range(len(annot)):
            img_path = annot[i]['img_path']
            f = np.array(annot[i]['f'], dtype=np.float32);
            c = np.array(annot[i]['c'], dtype=np.float32);
            root_cam_raw = np.array(annot[i]['root_cam'], dtype=np.float32).reshape(3)
            bbox_raw = np.array(annot[i]['bbox'], dtype=np.float32).reshape(4)

            joint_cam = np.array(annot[i]['joint_img'], dtype=np.float32)

            joint_img = np.array(annot[i]['joint_cam'], dtype=np.float32)
            joint_img = np.concatenate([joint_img, joint_cam[:, 2:]], 1)
            root_cam = joint_img[14]

            bbox = get_bbox(joint_img)
            bbox = process_bbox(bbox.copy())
            joint_vis = np.ones((joint_img.shape[0], 1), dtype=np.float32)

            data.append({
                'img_path': img_path,
                'raw_bbox': bbox_raw,
                'bbox': bbox,
                'joint_img': joint_img, # [org_img_x, org_img_y, depth]
                'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c,
            })

        return data

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        rot, flip = 0, 0
        data = copy.deepcopy(self.datalist[idx])

        img_path, bbox, raw_bbox = data['img_path'], data['bbox'], data['raw_bbox']
        joint_img_mupo, joint_cam_mupo, joint_vis = data['joint_img'], data['joint_cam'], data['joint_vis']

        # get h36mJ
        joint_img, joint_cam = joint_img_mupo[self.mupo2h36m], joint_cam_mupo[self.mupo2h36m]

        # root relative camera coordinate
        joint_cam = joint_cam - joint_cam[:1]

        # aug
        try:
            joint_img, trans = j2d_processing(joint_img.copy(),
                                               (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                               bbox, rot, flip, None)
        except:
            joint_img, trans = j2d_processing(joint_img.copy(),
                                              (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                              raw_bbox, rot, flip, None)

        #  -> 0~1
        joint_img = joint_img[:, :2]
        joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
        joint_img = (joint_img.copy() - mean) / std

        inputs = {'pose2d': joint_img, 'lift_pose3d_pred': joint_cam}
        targets = {'mesh': np.zeros((6890,3), dtype=np.float32), 'reg_pose3d': joint_cam}
        meta = {'dummy': np.ones(1, dtype=np.float32)}

        return inputs, targets, meta

    def evaluate(self, preds, result_dir=None):

        print('Evaluation start...')
        gts = self.datalist
        sample_num = len(preds)

        pred_2d_save = {}
        pred_3d_save = {}
        for n in range(sample_num):

            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            img_name = gt['img_path'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0]  # e.g., TS1_img_0001

            # restore coordinates to original space
            pred_3d_kpt = preds[n].copy()
            # pred_3d_kpt = np.dot(self.joint_regressor_h36m, mesh_coord_out)
            # relative joints
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[:1, :] + gt_3d_root[None, :]
            # get mupoJ
            pred_3d_kpt = pred_3d_kpt[self.h36m2mupo]
            pred_2d_kpt = cam2pixel(pred_3d_kpt, f, c)

            # vis = False
            # if vis:
            #     cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #     filename = str(random.randrange(1, 500))
            #     tmpimg = cvimg.copy().astype(np.uint8)
            #     tmpkps = np.zeros((3, joint_num))
            #     tmpkps[0, :], tmpkps[1, :] = pred_2d_kpt[:, 0], pred_2d_kpt[:, 1]
            #     tmpkps[2, :] = 1
            #     tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
            #     cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            # pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

            # 2d kpt save
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:, :2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:, :2]]

            # 3d kpt save
            if img_name in pred_3d_save:
                pred_3d_save[img_name].append(pred_3d_kpt)
            else:
                pred_3d_save[img_name] = [pred_3d_kpt]

        output_path = osp.join(result_dir, 'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join(result_dir, 'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)

if __name__ == '__main__':
    import argparse
    from core.config import cfg, update_config

    import torchvision.transforms as transforms
    import torch

    # update_config('asset/yaml/pose3d2mesh_human36J_train_human36.yml')

    train_dataset = MuPoTS('test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    for i, b in enumerate(train_loader):
        if i == 1:
            break
        else:
            print(b[0]['pose2d'].shape, b[1]['reg_pose3d'].shape, b[0]['lift_pose3d_pred'][0])
