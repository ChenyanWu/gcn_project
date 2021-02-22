import os.path as osp
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import Counter

import Human36M.dataset, SURREAL.dataset, COCO.dataset, PW3D.dataset, AMASS.dataset, MuCo.dataset, FreiHAND.dataset
import models
from multiple_datasets import MultipleDatasets
from core.loss import get_loss
from core.config import cfg
from display_utils import display_model
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, stop, lr_check, save_obj
from vis import vis_2d_pose, vis_3d_pose

logger = logging.getLogger(__name__)
def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    dataset_list, dataloader_list = [], []

    logger.info(f"==> Preparing {dataset_split} Dataloader...")
    for name in dataset_names:
        dataset = eval(f'{name}.dataset')(dataset_split.lower(), args=args)
        logger.info("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset * len(dataset_names), shuffle=cfg[dataset_split].shuffle,
                                     num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    if is_train or load_dir:
        logger.info(f"==> Preparing {cfg.MODEL.name} MODEL...")
        model = models.pose3d2mesh_net.get_model(num_joint=main_dataset.joint_num, graph_L=main_dataset.graph_L)
        logger.info('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.mesh_model.face)
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        logger.info('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'posenet'))
        model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            logger.info('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)

        self.normal_weight = cfg.MODEL.normal_loss_weight
        self.edge_weight = cfg.MODEL.edge_loss_weight
        self.joint_weight = cfg.MODEL.joint_loss_weight
        self.edge_add_epoch = cfg.TRAIN.edge_loss_start

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        # batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(self.batch_generator):
            # convert to cuda
            input_2dpose = inputs['pose2d'].cuda()
            # input_3dpose = inputs['lift_pose3d_pred'].cuda()
            input_3dpose = targets['lift_pose3d'].cuda()
            gt_lift3dpose, gt_reg3dpose, gt_mesh = targets['lift_pose3d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
            val_lift3dpose, val_reg3dpose, val_mesh = meta['lift_pose3d_valid'].cuda(), meta['reg_pose3d_valid'].cuda(), meta['mesh_valid'].cuda()

            # model
            pred_mesh, lift_pose = self.model(input_2dpose, input_3dpose)  # B x 12288 x 3
            pred_mesh = pred_mesh[:, self.main_dataset.graph_perm_reverse[:self.main_dataset.mesh_model.face.max() + 1], :]  # B x 6890 x 3
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh * 1000)

            # loss
            loss1, loss2, loss4 = self.loss[0](pred_mesh, gt_mesh, val_mesh),  \
                                                self.normal_weight * self.loss[1](pred_mesh, gt_mesh), \
                                                self.joint_weight * self.loss[3](pred_pose,  gt_reg3dpose, val_reg3dpose)
            loss3 = 0
            loss = loss1 + loss2 + loss3 + loss4

            if epoch > self.edge_add_epoch:
                loss3 = self.edge_weight * self.loss[2](pred_mesh, gt_mesh)
                loss += loss3

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(loss.detach().item())
            if i % self.print_freq == 0:
                loss1, loss2, loss4 = loss1.detach(), loss2.detach(), loss4.detach()
                loss3 = loss3.detach() if epoch > self.edge_add_epoch else 0
                logger.info(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                               f'vertice loss: {loss1:.4f} '
                                               f'normal loss: {loss2:.4f} '
                                               f'edge loss: {loss3:.4f} '
                                               f'mesh->3d joint loss: {loss4:.4f} ')

        self.loss_history.append(running_loss / len(self.batch_generator))

        logger.info(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        # loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(self.val_loader):
                input_pose, gt_pose3d, gt_mesh = inputs['pose2d'].cuda(), targets['reg_pose3d'].cuda(), targets['mesh'].cuda()
                # input_3dpose = targets['lift_pose3d'].cuda()
                input_3dpose = inputs['lift_pose3d_pred'].cuda()
                pred_mesh, pred_pose_from2d = self.model(input_pose, input_3dpose)
                pred_mesh = pred_mesh[:, self.val_dataset.graph_perm_reverse[:self.val_dataset.mesh_model.face.max() + 1], :]
                pred_mesh, gt_mesh = pred_mesh * 1000, gt_mesh * 1000

                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)

                j_error, s_error = self.val_dataset.compute_both_err(pred_mesh, gt_mesh, pred_pose, gt_pose3d)

                # vis_3d_pose(pred_pose[0].detach().cpu().numpy(), self.val_dataset.skeleton, joint_set_name='smpl')
                # vis_3d_pose(gt_pose3d[0].detach().cpu().numpy(), self.val_dataset.skeleton, joint_set_name='smpl')
                if i % self.print_freq == 0:
                    logger.info(f'{eval_prefix}({i}/{len(self.val_loader)}) => surface error: {s_error:.4f}, joint error: {j_error:.4f}')

                joint_error += j_error
                surface_error += s_error

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy()
                    for j in range(len(input_pose)):
                        out = {}
                        out['mesh_coord'], out['mesh_coord_target'] = pred_mesh[j], target_mesh[j]
                        result.append(out)

            self.surface_error = surface_error / len(self.val_loader)
            self.joint_error = joint_error / len(self.val_loader)
            logger.info(f'{eval_prefix} MPVPE: {self.surface_error:.2f}, MPJPE: {self.joint_error:.2f}')

            # Final Evaluation
            if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                self.val_dataset.evaluate(result)


class Tester_mupo:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)

        self.val_loader, self.val_dataset = self.val_loader[0], self.val_dataset[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.val_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        if self.model:
            self.model = self.model.cuda()
            self.model = nn.DataParallel(self.model)

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        surface_error = 0.0
        joint_error = 0.0

        result = []
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (inputs, targets, meta) in enumerate(loader):
                input_pose = inputs['pose2d'].cuda()
                input_3dpose = inputs['lift_pose3d_pred'].cuda()
                pred_mesh, pred_pose_from2d = self.model(input_pose, input_3dpose)
                pred_mesh = pred_mesh[:, self.val_dataset.graph_perm_reverse[:self.val_dataset.mesh_model.face.max() + 1], :]
                pred_mesh = pred_mesh * 1000
                pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)

                pred_pose = pred_pose.detach().cpu().numpy()
                for j in range(len(input_pose)):
                    result.append(pred_pose[j])

            # Final Evaluation
            self.val_dataset.evaluate(result, './')


