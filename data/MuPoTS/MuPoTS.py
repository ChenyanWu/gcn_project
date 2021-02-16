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
from coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox
from aug_utils import affine_transform, j2d_processing, augm_params, j3d_processing

from funcs_utils import save_obj, stop
from vis import vis_3d_pose, vis_2d_pose

class MuPoTS(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        self.data_split = 'test'
        self.data = self.load_data()

        # Use ground truth
        self.use_gt_info = cfg.use_gt_info

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

        self.datalist = self.load_data()

    def load_data(self):
        print('Load annotations of MuPoTS ')
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        data = []
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))

        # the groundtruth bbox
        # print("Get bounding box and root from groundtruth")
        # for aid in db.anns.keys():
        #     ann = db.anns[aid]
        #     if ann['is_valid'] == 0:
        #         continue
        #
        #     image_id = ann['image_id']
        #     img = db.loadImgs(image_id)[0]
        #     img_path = osp.join(self.img_dir, img['file_name'])
        #     fx, fy, cx, cy = img['intrinsic']
        #     f = np.array([fx, fy]);
        #     c = np.array([cx, cy]);
        #
        #     joint_cam = np.array(ann['keypoints_cam'])
        #     root_cam = joint_cam[self.root_idx]
        #
        #     joint_img = np.array(ann['keypoints_img'])
        #     joint_img = np.concatenate([joint_img, joint_cam[:, 2:]], 1)
        #     joint_img[:, 2] = joint_img[:, 2] - root_cam[2]
        #     joint_vis = np.ones((self.original_joint_num, 1))
        #
        #     bbox = np.array(ann['bbox'])
        #     img_width, img_height = img['width'], img['height']
        #     bbox = process_bbox(bbox, img_width, img_height)
        #     if bbox is None: continue
        #
        #     data.append({
        #         'img_path': img_path,
        #         'bbox': bbox,
        #         'joint_img': joint_img,  # [org_img_x, org_img_y, depth - root_depth]
        #         'joint_cam': joint_cam,  # [X, Y, Z] in camera coordinate
        #         'joint_vis': joint_vis,
        #         'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
        #         'f': f,
        #         'c': c,
        #     })

        print("Get bounding box and root from " + self.human_bbox_root_dir)
        with open(self.human_bbox_root_dir) as f:
            annot = json.load(f)

        for i in range(len(annot)):
            image_id = annot[i]['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_dir, img['file_name'])
            fx, fy, cx, cy = img['intrinsic']
            f = np.array([fx, fy]);
            c = np.array([cx, cy]);
            root_cam = np.array(annot[i]['root_cam']).reshape(3)
            bbox = np.array(annot[i]['bbox']).reshape(4)

            data.append({
                'img_path': img_path,
                'bbox': bbox,
                'joint_img': np.zeros((self.original_joint_num, 3)),  # dummy
                'joint_cam': np.zeros((self.original_joint_num, 3)),  # dummy
                'joint_vis': np.zeros((self.original_joint_num, 1)),  # dummy
                'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c,
            })

        return data


    def evaluate(self, preds, result_dir):

        print('Evaluation start...')
        gts = self.data
        sample_num = len(preds)
        joint_num = self.original_joint_num

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
            pred_2d_kpt = preds[n].copy()
            # only consider eval_joint
            pred_2d_kpt = np.take(pred_2d_kpt, self.eval_joint, axis=0)
            pred_2d_kpt[:, 0] = pred_2d_kpt[:, 0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_2d_kpt[:, 1] = pred_2d_kpt[:, 1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:, 2] = (pred_2d_kpt[:, 2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0] / 2) + gt_3d_root[2]

            # 2d kpt save
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:, :2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:, :2]]

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1, 500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3, joint_num))
                tmpkps[0, :], tmpkps[1, :] = pred_2d_kpt[:, 0], pred_2d_kpt[:, 1]
                tmpkps[2, :] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)

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