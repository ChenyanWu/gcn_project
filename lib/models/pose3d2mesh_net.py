import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import meshnet


class FlatPose2Mesh(nn.Module):
    def __init__(self, num_joint, graph_L):
        super(FlatPose2Mesh, self).__init__()

        self.num_joint = num_joint
        self.pose2mesh = meshnet.get_model(num_joint_input_chan=2 + 3, num_mesh_output_chan=3, graph_L=graph_L)

    def forward(self, pose2d, pose3d):
        print(pose2d.shape, pose3d.shape)
        pose_combine = torch.cat((pose2d, pose3d / 1000), dim=2)
        cam_mesh = self.pose2mesh(pose_combine)

        return cam_mesh, pose3d


def get_model(num_joint, graph_L):
    model = FlatPose2Mesh(num_joint, graph_L)

    return model


