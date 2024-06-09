import glob
import os
import pickle

import numpy as np
import roma
import torch
from tqdm import tqdm

from .utils import (J_MASK, V_MASK, _syn_acc,
                                 _syn_ang_vel, _syn_vel, _syn_fix_joint, AMASS_ROT)
from ..smpl import ParametricModel

IMU_MASK = torch.tensor([5, 0, 1, 4, 2, 3])
TRAIN_SPLIT = []
VALID_SPLIT = []
TEST_SPLIT = ["S1", "S2", "S3", "S4", "S5"]

def process_total_capture(smpl_path, tc_path):
    r"""
    Load Pose & Translation and Place Virtual IMU on Total Capture dataset +
    Synthesize Joint Velocity, Joint Accelration, Joint Angular Velocity, Jonit Position

    :param smpl_path: path to smpl.pkl model.
    :param amass_path: folder containing AMASS datasets.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path)
    tc_files = glob.glob(os.path.join(tc_path, '*/*.pkl'))
    loop = tqdm(tc_files)
    for file in loop:

            with open(file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            ori = torch.from_numpy(data['ori']).float()[:, IMU_MASK]
            acc = torch.from_numpy(data['acc']).float()[:, IMU_MASK]
            pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

            if acc.shape[0] < pose.shape[0]:
                pose = pose[:acc.shape[0]]
            elif acc.shape[0] > pose.shape[0]:
                acc = acc[:pose.shape[0]]
                ori = ori[:pose.shape[0]]
            length = pose.shape[0]

            pose_path = file.split("TotalCapture_Real_60FPS")[0]
            pose_file = os.path.basename(file).split("_")
            subject = pose_file[0].upper()
            pose_file = os.path.join(pose_path, "TotalCapture",
                                     pose_file[0].lower(), f"{pose_file[1].split('.')[0]}_poses.npz")
            if not os.path.exists(pose_file):
                print(f"No trans file found for {file}")
                continue
            cdata = np.load(pose_file)
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

            trans = cdata['trans'][::step].astype(np.float32)
            shape = torch.tensor(cdata['betas'][:10]).float()
            trans = torch.tensor(trans)

            trans = AMASS_ROT.matmul(trans.unsqueeze(-1)).view_as(trans)


            if acc.shape[0] < trans.shape[0]:
                trans = trans[:acc.shape[0]]
            assert trans.shape[0] == acc.shape[0]

            rot_mat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)

            grot, joint, vert, lj = body_model.forward_kinematics(pose=rot_mat, tran=trans, shape=shape, calc_mesh=True)
            javel = _syn_ang_vel(grot)
            jlacc = _syn_acc(joint)
            vacc = _syn_acc(vert[:, V_MASK])
            d = vacc.mean(dim=0, keepdim=True) - acc.mean(dim=0, keepdim=True)
            bone_length, adj_matrix = body_model.get_bone_features_and_adj_matrix()


            targets = {
                'imu_acc': acc + d,
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
                'grot': grot,
                'bone_length': bone_length,
                'adj_matrix': adj_matrix,
                'gender': "male",
                'dataset': "TotalCapture",
            }
            if subject in TRAIN_SPLIT:
                folder = "Train"
            elif subject in VALID_SPLIT:
                folder = "Valid"
            elif subject in TEST_SPLIT:
                folder = "Test"
            else:
                raise ValueError(f"Folder {subject} not recognized")

            target_folder = file.split("TotalCapture_Real_60FPS")[0].replace("TotalCapture", "TC_process_data").replace("raw data", folder)
            os.makedirs(os.path.join(target_folder, subject), exist_ok=True)
            torch.save(targets, os.path.join(target_folder, subject, os.path.basename(file).replace(".pkl", ".pt")))
            loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
            del pose, trans, rot_mat, grot, joint, vert, javel, jlacc, targets



if __name__ == '__main__':
    data_dir = "../data/TotalCapture/"
    smpl_dir = "../data/SMPL Model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    process_total_capture(smpl_path=smpl_dir, tc_path=data_dir)
