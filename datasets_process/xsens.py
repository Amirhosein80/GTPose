# codes from https://github.com/dx118/dynaip.git


import glob
import os

import torch
from tqdm import tqdm

from .utils import _syn_vel, _syn_acc, _syn_ang_vel, J_MASK, V_MASK, _syn_fix_joint
from .xsens_utils import read_mvnx, read_xlsx
from ..smpl import ParametricModel

IMU_MASK = torch.tensor([0, 15, 12, 2, 9, 5])
EM_TRAIN_SPLIT = ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7"]
EM_VALID_SPLIT = ["seq8", "seq9"]
EM_TEST_SPLIT = []

MTW_TRAIN_SPLIT = ["trials"]
MTW_VALID_SPLIT = ["trials_extra_indoors", "trials_extra_outdoors"]
MTW_TEST_SPLIT = []

def process_emonike(smpl_path, emonike_path):
    r"""
    Load Pose & Translation and Place Virtual IMU on EMONIKE dataset +
    Synthesize Joint Velocity, Joint Accelration, Joint Angular Velocity, Jonit Position

    :param smpl_path: path to smpl.pkl model.
    :param emonike_path: folder containing EMONIKE dataset.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path)
    em_files = glob.glob(os.path.join(emonike_path, '*.mvnx'))
    loop = tqdm(em_files)
    for file in loop:
        data = read_mvnx(file=file, smpl_file=smpl_path)
        pose = data['joint']['orientation']
        trans = data['joint']['translation']
        length = pose.shape[0]

        ori = data['imu']['calibrated orientation'][:, IMU_MASK]
        acc = data['imu']['free acceleration'][:, IMU_MASK]

        grot, joint, vert, lj = body_model.forward_kinematics(pose=pose, tran=trans, calc_mesh=True)
        javel = _syn_ang_vel(grot)
        jlacc = _syn_acc(joint)
        bone_length, adj_matrix = body_model.get_bone_features_and_adj_matrix()

        targets = {
            'imu_acc': acc,
            'imu_ori': ori,
            'pose': pose,
            'trans': trans,
            'velocity': _syn_vel(trans, pose[:, 0]),
            'joint': joint,
            'joint_wo_tr': lj,
            'javel': javel,
            'jlacc': jlacc,
            'jvel': _syn_vel(joint, pose),
            'vacc': _syn_acc(vert[:, V_MASK]),
            'vrot': grot[:, J_MASK],
            'fix_joint': _syn_fix_joint(joint),
            'grot': grot,
            'bone_length': bone_length,
            'adj_matrix': adj_matrix,
            'gender': "male",
            'dataset': "Emonike"
        }

        seq = os.path.basename(file).split("_")[1]
        if seq in EM_TRAIN_SPLIT:
            folder = "Train"
        elif seq in EM_VALID_SPLIT:
            folder = "Valid"
        elif seq in EM_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {seq} not recognized")
        target_folder = file.split('EmokineDataset_v1.0')[0].replace("raw data", folder)
        target_folder = os.path.join(target_folder, "EM_process_data", seq)
        os.makedirs(target_folder,
                    exist_ok=True)

        torch.save(targets, os.path.join(target_folder, os.path.basename(file).replace('.mvnx', '.pt')))
        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, grot, joint, vert, javel, jlacc, targets, length

def process_xsens(smpl_path, mvnx_path):
    r"""
    Load Pose & Translation and Place Virtual IMU on MVNX dataset +
    Synthesize Joint Velocity, Joint Accelration, Joint Angular Velocity, Jonit Position

    :param smpl_path: path to smpl.pkl model.
    :param mvnx_path: folder containing MVNX dataset.
    """
    discread_files = []
    body_model = ParametricModel(smpl_path)
    em_files = glob.glob(os.path.join(mvnx_path, "*/*/*/*/*.xlsx"))
    loop = tqdm(em_files)
    for file in loop:
        if "calibration" in file:
            loop.set_description(f"Calibration file {os.path.basename(file)}")
            continue
        if file.split("\\")[-2][0] == "_":
            loop.set_description(f"Bug file {os.path.basename(file)}")
            discread_files.append(os.path.basename(file))
            continue
        data = read_xlsx(xsens_file_path=file, smpl_file=smpl_path)
        pose = data['joint']['orientation']
        trans = data['joint']['translation']
        length = pose.shape[0]

        ori = data['imu']['calibrated orientation'][:, IMU_MASK]
        acc = data['imu']['free acceleration'][:, IMU_MASK]

        grot, joint, vert, lj = body_model.forward_kinematics(pose=pose, tran=trans, calc_mesh=True)
        javel = _syn_ang_vel(grot)
        jlacc = _syn_acc(joint)
        bone_length, adj_matrix = body_model.get_bone_features_and_adj_matrix()

        targets = {
            'imu_acc': acc,
            'imu_ori': ori,
            'pose': pose,
            'trans': trans,
            'velocity': _syn_vel(trans, pose[:, 0]),
            'joint': joint,
            'joint_wo_tr': lj,
            'javel': javel,
            'jlacc': jlacc,
            'jvel': _syn_vel(joint, pose),
            'vacc': _syn_acc(vert[:, V_MASK]),
            'vrot': grot[:, J_MASK],
            'fix_joint': _syn_fix_joint(joint),
            'bone_length': bone_length,
            'adj_matrix': adj_matrix,
            'grot': grot,
            'gender': "male",
            'dataset': "MTwAwinda",
        }
        target_folder = file.split('MTwAwinda')[0]
        split_foder = file.split('\\')[1]

        if split_foder in MTW_TRAIN_SPLIT:
            folder = "Train"
        elif split_foder in MTW_VALID_SPLIT:
            folder = "Valid"
        elif split_foder in MTW_TEST_SPLIT:
            folder = "Test"
        else:
            raise ValueError(f"Folder {split_foder} not recognized")

        target_folder = os.path.join(target_folder, f"MTW_process_data/{split_foder}").replace("raw data", folder)
        os.makedirs(target_folder,
                    exist_ok=True)
        torch.save(targets, os.path.join(target_folder, os.path.basename(file).replace('.xlsx', '.pt')))
        loop.set_description(f'Processed {file}, discrete num {len(discread_files)}, length {length}')
        del pose, trans, grot, joint, vert, javel, jlacc, targets, length


if __name__ == '__main__':
    data_dir = "../data/raw data/MTwAwinda"
    smpl_dir = "../data/SMPL Model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    process_xsens(smpl_path=smpl_dir, emonike_path=data_dir)
    data_dir = "../data/raw data/EmokineDataset_v1.0/Data/MVNX"
    process_emonike(smpl_path=smpl_dir, emonike_path=data_dir)


