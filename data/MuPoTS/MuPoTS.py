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
        dataset_name = 'MuPoTS'
        self.data_split = 'test'
        self.debug = args.debug
        self.img_dir = osp.join(cfg.data_dir, dataset_name, 'data', 'MultiPersonTestSet')
        self.det_path = osp.join(cfg.data_dir, dataset_name, 'bbox_root', 'bbox_mupots_output.json')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'data', 'MuPoTS-3D.json')
        # TODO no smpl parameters currently
        # self.smpl_param_path = osp.join(cfg.data_dir, dataset_name, 'data', 'smpl_param.json')
        self.data = self.load_data()

        # Use ground truth
        self.use_gt_info = cfg.use_gt_info

        # h36m skeleton
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_joint_num = 17
        self.human36_joints_name = (
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.joint_regressor_h36m = self.mesh_model.joint_regressor_h36m

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

        # change MuPo joints to hm36
        self. =

        self.input_joint_name = cfg.DATASET.input_joint_set
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        self.datalist = self.load_data()

    def load_data(self):
        print('Load annotations of MuPoTS ')
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        data = []
        db = COCO(self.annot_path)

        # get ground truth 3D pose and predicted 2D bbox and root
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

            aid = self.find_id_bbox(bbox)
            ann = db.anns[aid]
            gt_joint_img =
            gt_joint_cam =
            gt_joint_vis =

            pred_joint_cam =
            data.append({
                'img_path': img_path,
                'bbox': bbox,
                'gt_joint_img': np.zeros((self.original_joint_num, 3)),  # dummy
                'gt_joint_cam': np.zeros((self.original_joint_num, 3)),  # dummy
                'gt_joint_vis': np.zeros((self.original_joint_num, 1)),  # dummy
                'pred_joint_cam': pred_joint_cam
                'root_cam': root_cam,  # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c,
            })

        return data

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smpl_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data[
            'smpl_param'], data['cam_param']

        flip, rot = augm_params(is_train=(self.data_split == 'train'))

        # regress h36m, coco joints
        mesh_cam, joint_cam_smpl = self.get_smpl_coord(smpl_param)
        joint_cam_h36m, joint_img_h36m = self.get_joints_from_mesh(mesh_cam, 'human36', cam_param)
        joint_cam_coco, joint_img_coco = self.get_joints_from_mesh(mesh_cam, 'coco', cam_param)
        # vis_3d_save(joint_cam_h36m, self.human36_skeleton, prefix='3d_gt', gt=True)
        # vis_2d_joints(joint_img_h36m, img_path, self.human36_skeleton, prefix='h36m joint')
        # vis_2d_joints(joint_img_coco, img_path, self.coco_skeleton, prefix='coco joint')

        # root relative camera coordinate
        mesh_cam = mesh_cam - joint_cam_h36m[:1]
        joint_cam_coco = joint_cam_coco - joint_cam_coco[-2:-1]
        joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

        if self.input_joint_name == 'coco':
            joint_img, joint_cam = joint_img_coco, joint_cam_coco
        elif self.input_joint_name == 'human36':
            joint_img, joint_cam = joint_img_h36m, joint_cam_h36m

        joint_img =
        joint_cam =

        # make new bbox
        init_bbox = get_bbox(joint_img)
        bbox = process_bbox(init_bbox.copy())

        # aug
        joint_img, trans = j2d_processing(joint_img.copy(), (cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]),
                                          bbox, rot, flip, self.flip_pairs)
        joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)

        # TODO compute the predictive joints
        joint_cam_pred_relative = joint_cam

        if not cfg.DATASET.use_gt_input:
            joint_img = self.replace_joint_img(joint_img, bbox, trans)

        #  -> 0~1
        joint_img = joint_img[:, :2]
        joint_img /= np.array([[cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]]])

        # normalize loc&scale
        mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
        joint_img = (joint_img.copy() - mean) / std

        # TODO change the input into multi-person
        # default valid
        mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
        reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
        lift_joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
        # if fitted mesh is too far from h36m gt, discard it
        error = self.get_fitting_error(joint_cam_h36m, mesh_cam)
        if error > self.fitting_thr:
            mesh_valid[:], reg_joint_valid[:], lift_joint_valid[:] = 0, 0, 0

        inputs = {'pose2d': joint_img, 'lift_pose3d_pred': joint_cam_pred_relative}
        targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
        meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

        return inputs, targets, meta

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