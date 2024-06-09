# codes from https://github.com/dx118/dynaip.git


import roma
import torch
import random
import numpy as np

V_MASK = torch.tensor([3021, 1176, 4662, 411, 1961, 5424])
J_MASK = torch.tensor([0, 4, 5, 15, 18, 19])
AMASS_ROT = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])


def _syn_acc(trans):
    r"""
    Synthesize accelration from poseitions
    
    :param trans: translation in [T, N, 3]
    :return: accelration in [T, N, 3]
    """
    acc = torch.zeros_like(trans)
    acc[1:-1] = (trans[0:-2] + trans[2:] - 2 * trans[1:-1]) * 3600
    acc[4:-4] = (trans[0:-8] + trans[8:] - 2 * trans[4:-4]) * 3600 / 16
    return acc


def _syn_ang_vel(pose):
    r"""
    Synthesize angular velocity from pose
    
    :param pose: pose in [T, N, 3, 3]
    :return: accelration in [T, N, 3]
    """
    ang_v = torch.zeros(pose.shape[:-1])
    quat = roma.rotmat_to_unitquat(pose)
    sub = quat[2:] - quat[:-2] if torch.linalg.norm(
        quat[2:] - quat[:-2]) < torch.linalg.norm(
        quat[2:] + quat[:-2]) else quat[2:] + quat[:-2]
    dori = 2 * roma.quat_product(sub, roma.quat_conjugation(quat[2:]))
    ang_v[1:-1] = (dori * 30)[..., :3]
    return ang_v


def _syn_vel(trans, root):
    r"""
    Synthesize velocity from poseitions
    
    :param trans: poseitions in [T, N, 3]
    :return: velocity in [T, N, 3]
    """
    vel = trans.clone()
    vel[1:] = trans[1:] - trans[:-1]
    vel = root.transpose(-1, -2) @ vel.unsqueeze(-1)
    return vel[..., 0] * 30


def _syn_fix_joint(trans):
    r"""
    Synthesize fix poseitions from poseitions
    
    :param trans: poseitions in [T, N, 3]
    :return: fix or not in [T, N]
    """
    fix_joint = torch.zeros(*trans.shape[:2])
    fix_joint[1:] = torch.where((trans[1:] - trans[:-1]).norm(dim=-1) < 0.008, 1.0, 0.0)
    return fix_joint


def imu_calibration(imu_data, joints_data, n_frame, joint_mask):
    r""" 
    Sensor Calibration for Xesns datasets
    
    :param imu_data: imu orientation data in [T, N, 3, 3]
    :param joint_data: joint orientation data in [T, N, 3, 3]
    :param n_frame: number of frame for calibraion 
    :param joint_mask: imu mask
    :return: calibrated imu orientation data in [T, N, 3, 3]
    """
    quat1 = imu_data[:n_frame]
    quat2 = joints_data[:n_frame, joint_mask]
    quat_off = roma.quat_product(roma.quat_inverse(quat1), quat2)
    ds = quat_off.abs().mean(dim=0).max(dim=-1)[1]
    for i, d in enumerate(ds):
        quat_off[:, i] = quat_off[:, i] * quat_off[:, i, d:d + 1].sign()

    quat_off = roma.quat_normalize(roma.quat_normalize(quat_off).mean(dim=0))
    return roma.quat_product(imu_data,
                             quat_off.repeat(imu_data.shape[0], 1, 1))


def convert_quaternion_xsens(quat):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    old_quat = quat.reshape(-1, 4).clone()
    quat.view(-1, 4)[:, 1] = old_quat[:, 2]
    quat.view(-1, 4)[:, 2] = old_quat[:, 3]
    quat.view(-1, 4)[:, 3] = old_quat[:, 1]


def convert_point_xsens(point):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    old_point = point.reshape(-1, 3).clone()
    point.view(-1, 3)[:, 0] = old_point[:, 1]
    point.view(-1, 3)[:, 1] = old_point[:, 2]
    point.view(-1, 3)[:, 2] = old_point[:, 0]


def convert_quaternion_excel(quat):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_pose = R mvnx_pose R^T
    """
    quat[:, :, [1, 2, 3]] = quat[:, :, [2, 3, 1]]
    return quat


def convert_point_excel(point):
    r""" inplace convert
        R = [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
        smpl_point = R mvnx_point
    """
    point[:, :, [0, 1, 2]] = point[:, :, [1, 2, 0]]
    return point




if __name__ == '__main__':
    a = np.random.random((128, 6, 3))
    b = np.single(a)
