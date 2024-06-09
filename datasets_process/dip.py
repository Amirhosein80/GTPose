# codes from https://github.com/dx118/dynaip.git

import glob
import os
import pickle

import numpy as np
import roma
import torch
from tqdm import tqdm

from .utils import V_MASK, J_MASK, _syn_acc, _syn_ang_vel, _syn_vel, _syn_fix_joint
from ..smpl import ParametricModel, joint_position_to_bone_vector, normalize_tensor

IMU_MASK = torch.tensor([2, 11, 12, 0, 7, 8])
TRAIN_SPLIT = ["s_01", "s_02", "s_03", "s_04", "s_05", "s_06", "s_07", "s_08"]
VALID_SPLIT = ["s_09", "s_10"]
TEST_SPLIT = []

def gen_trans_dip(pose, body_model):
    r"""
    Synthesize velocity for y-axis
    
    :param pose: body pose in [T, 24, 3]
    :param body_model: SMPL body model
    :return: translation vector in [T, 3]
    """
    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
    body_joints, _ = body_model.get_zero_pose_joint_and_vertex()
    b = joint_position_to_bone_vector(body_joints[lower_body].unsqueeze(0),
                                        lower_body_parent).squeeze(0)
    bone_orientation, bone_length = normalize_tensor(b, return_norm=True)
    b = bone_orientation * bone_length
    b[:3] = 0
    j = body_model.forward_kinematics(pose=pose, calc_mesh=False)[1]
    floor_y = j[10:12, 1].min().item()
    trans = torch.zeros(j.shape[0], 3)

    for i in range(j.shape[0]):
        current_foot_y = j[i, [10, 11], 1].min().item()
        if current_foot_y > floor_y:
            trans[i, 1] = floor_y - current_foot_y

    return trans

def process_dip(smpl_path, dip_path):
    r"""
    Load Pose and Place Virtual IMU on AMASS datasets +
    Synthesize Joint Velocity, Joint Accelration, Joint Angular Velocity, Jonit Position, 

    :param smpl_path: path to smpl.pkl model.
    :param amass_path: folder containing AMASS datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path)
    dip_files = glob.glob(os.path.join(dip_path, '*/*.pkl'))
    loop = tqdm(dip_files)
    for file in loop:

        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        ori = torch.from_numpy(data['imu_ori']).float()[:, IMU_MASK]
        acc = torch.from_numpy(data['imu_acc']).float()[:, IMU_MASK]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)
        for _ in range(4):
            acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
            ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
            acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
            ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

        if torch.isnan(acc).sum() != 0 or torch.isnan(ori).sum() != 0 or torch.isnan(pose).sum() != 0:
            print(f'DIP-IMU: {file} has too much nan! Discard!')
            continue

        acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
        length = pose.shape[0]

        rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)
        trans = gen_trans_dip(rot_mat, body_model)

        grot, joint, vert, lj = body_model.forward_kinematics(pose=rot_mat,
                                                              tran=trans, calc_mesh=True)
        javel = _syn_ang_vel(grot)
        jlacc = _syn_acc(joint)
        bone_length, adj_matrix = body_model.get_bone_features_and_adj_matrix()

        targets = {
            'imu_acc': acc,
            'imu_ori': ori,
            'pose': roma.rotvec_to_rotmat(pose),
            'trans': trans,
            'velocity': _syn_vel(trans, roma.rotvec_to_rotmat(pose)[:, 0]),
            'joint': joint,
            'joint_wo_tr': lj,
            'javel': javel,
            'jlacc': jlacc,
            'jvel': _syn_vel(joint, roma.rotvec_to_rotmat(pose)),
            'vacc': _syn_acc(vert[:, V_MASK]),
            'vrot': grot[:, J_MASK],
            'fix_joint': _syn_fix_joint(joint),
            'bone_length': bone_length,
            'adj_matrix': adj_matrix,
            'grot': grot,
            'gender': "male",
            'dataset': "DIP"
        }
        folder = file.split("\\")[1]
        if folder in TRAIN_SPLIT:
            folder = "Train"
        elif folder in VALID_SPLIT:
            folder = "Valid"
        elif folder in TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {folder} not recognized")
        target_folder = file.split(os.path.basename(file))[0].replace("DIP_IMU",
                                                                      "DIP_process_data").replace("raw data",
                                                                                                  folder)
        os.makedirs(target_folder,
                    exist_ok=True)
        torch.save(targets, os.path.join(target_folder, os.path.basename(file).replace(".pkl", ".pt")))
        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, rot_mat, grot, joint, vert, javel, jlacc, targets


if __name__ == '__main__':
    data_dir = "../data/raw data/DIP_IMU/"
    smpl_dir = "../data/SMPL Model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    process_dip(smpl_path=smpl_dir, dip_path=data_dir)
