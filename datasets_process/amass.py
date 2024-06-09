# codes from https://github.com/Xinyu-Yi/TransPose.git

import glob
import os

import numpy as np
import roma
import torch
from tqdm import tqdm

from ..smpl import ParametricModel
from .utils import J_MASK, V_MASK, AMASS_ROT, _syn_acc, _syn_ang_vel, _syn_vel, _syn_fix_joint

TRAIN_SPLIT = ['ACCAD', 'BMLhandball', 'BMLmovi', 'CMU',
               'GRAB', 'HUMAN4D', 'HumanEva', 'MPI_HDM05',
               'MPI_Limits', 'MPI_mosh', 'SFU', 'SOMA',
               'SSM_synced', 'TCD_handMocap', 'Transitions_mocap', 'Eyes_Japan_Dataset']
VALID_SPLIT = ['DanceDB', 'DFaust_67', 'EKUT']
TEST_SPLIT = []

def process_amass(smpl_path, amass_path):
    r"""
    Load Pose & Translation and Place Virtual IMU on AMASS datasets +
    Synthesize Joint Velocity, Joint Accelration, Joint Angular Velocity, Jonit Position

    :param smpl_path: path to smpl.pkl model.
    :param amass_path: folder containing AMASS datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path)
    amass_files = glob.glob(os.path.join(amass_path, '*/*/*_poses.npz'))
    loop = tqdm(amass_files)
    for file in loop:

        try:
            cdata = np.load(file)
            loop.set_description(f'processing {file}')
        except:
            continue
        if "mocap_framerate" not in cdata.keys():
            print(f'no mocap_framerate in {file}')
            discread_files.append(file)
            continue
        framerate = int(cdata['mocap_framerate'])
        if framerate == 120 or framerate == 100:
            step = 2
        elif framerate == 60 or framerate == 59 or framerate == 50:
            step = 1
        elif framerate == 250:
            step = 5
        elif framerate == 150:
            step = 3
        else:
            print(f'framerate {framerate} not supported in {os.path.basename(file)}')
            discread_files.append(file)
            continue
        poses = cdata['poses'][::step].astype(np.float32).reshape(-1, 52, 3)
        trans = cdata['trans'][::step].astype(np.float32)
        shape = torch.tensor(cdata['betas'][:10]).float()
        gender = cdata['gender']
        del cdata
        length = poses.shape[0]
        if length <= 10:
            discread_files.append(file)
            loop.set_description(
                f'discreaded file {os.path.basename(file)}, discrete num {len(discread_files)}, length {length}')
            continue
        poses[:, 23] = poses[:, 37]
        poses = poses[:, :24]

        poses = torch.tensor(poses)
        trans = torch.tensor(trans)

        poses[:, 0] = roma.rotmat_to_rotvec(AMASS_ROT.matmul(roma.rotvec_to_rotmat(poses[:, 0])))
        trans = AMASS_ROT.matmul(trans.unsqueeze(-1)).view_as(trans)
        rot_mat = roma.rotvec_to_rotmat(poses).view(-1, 24, 3, 3)

        grot, joint, vert, lj = body_model.forward_kinematics(pose=rot_mat, tran=trans, calc_mesh=True, shape=shape)
        javel = _syn_ang_vel(grot)
        jlacc = _syn_acc(joint)
        bone_length, adj_matrix = body_model.get_bone_features_and_adj_matrix(shape=shape)
        folder = file.split("\\")[1]

        targets = {
            'pose': roma.rotvec_to_rotmat(poses),
            'trans': trans,
            'joint': joint,
            'joint_wo_tr': lj,
            'javel': javel,
            'jlacc': jlacc,
            'jvel': _syn_vel(joint, roma.rotvec_to_rotmat(poses)),
            'velocity': _syn_vel(trans, roma.rotvec_to_rotmat(poses)[:, 0]),
            'vacc': _syn_acc(vert[:, V_MASK]),
            'vrot': grot[:, J_MASK],
            'grot': grot,
            'fix_joint': _syn_fix_joint(joint),
            'gender': gender,
            'bone_length': bone_length,
            'adj_matrix': adj_matrix,
            'framerate': framerate,
            "dataset": folder,
        }

        if folder in TRAIN_SPLIT:
            split = "Train"
        elif folder in VALID_SPLIT:
            split = "Valid"
        elif folder in TEST_SPLIT:
            split = "Test"
        else:
            raise ValueError(f"Folder {folder} not recognized")

        targets_folder = file.split("AMASS")[0].replace("raw data", split)
        targets_folder = os.path.join(targets_folder, f"{folder}_process_data", file.split("\\")[2])
        os.makedirs(targets_folder, exist_ok=True)
        torch.save(targets, os.path.join(targets_folder, os.path.basename(file).replace(".npz", ".pt")))
        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del poses, trans, rot_mat, grot, joint, vert, javel, jlacc, targets, framerate

    print("discread files: ", discread_files)
    print("AMASS process done!")


if __name__ == '__main__':
    data_dir = "../data/raw data/AMASS/"
    smpl_dir = "../data/SMPL Model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    process_amass(smpl_path=smpl_dir, amass_path=data_dir)
