import xml.etree.ElementTree as ET

import roma
import torch
import pandas as pd
import numpy as np

from ..smpl import ParametricModel
from .utils import imu_calibration, convert_point_xsens, convert_quaternion_xsens, convert_point_excel, convert_quaternion_excel


def read_mvnx(file: str, smpl_file: str):
    r"""
    Load IMU & Joint data from MVNX file.

    :param file: MVNX file path.
    :param smpl_file: path to smpl.pkl model.
    :return: data dictionary containing IMU and Joint data.
    """
    model = ParametricModel(smpl_file)
    tree = ET.parse(file)

    # read framerate
    frameRate = int(tree.getroot()[2].attrib['frameRate'])

    segments = tree.getroot()[2][1]
    n_joints = len(segments)
    joints = []
    for i in range(n_joints):
        assert int(segments[i].attrib['id']) == i + 1
        joints.append(segments[i].attrib['label'])

    sensors = tree.getroot()[2][2]
    n_imus = len(sensors)
    imus = []
    for i in range(n_imus):
        imus.append(sensors[i].attrib['label'])

    # read frames
    frames = tree.getroot()[2][-1]
    data = {'framerate': frameRate,
            'timestamp ms': [],
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            }

    if frameRate != 60:
        step = int(frameRate // 60)
    else:
        step = 1

    for i in range(len(frames)):
        if frames[i].attrib['type'] in ['identity', 'tpose', 'tpose-isb']:  # virginia
            continue

        elif ('index' in frames[i].attrib) and (frames[i].attrib['index'] == ''):  # unipd
            continue

        orientation = torch.tensor([float(_) for _ in frames[i][0].text.split(' ')]).view(n_joints, 4)
        position = torch.tensor([float(_) for _ in frames[i][1].text.split(' ')]).view(n_joints, 3)
        sensorFreeAcceleration = torch.tensor([float(_) for _ in frames[i][7].text.split(' ')]).view(n_imus, 3)
        try:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][9].text.split(' ')]).view(n_imus, 4)
        except:
            sensorOrientation = torch.tensor([float(_) for _ in frames[i][8].text.split(' ')]).view(n_imus, 4)

        data['timestamp ms'].append(int(frames[i].attrib['time']))
        data['joint']['orientation'].append(orientation)
        data['joint']['position'].append(position)
        data['imu']['free acceleration'].append(sensorFreeAcceleration)
        data['imu']['orientation'].append(sensorOrientation)

    data['timestamp ms'] = torch.tensor(data['timestamp ms'])
    for k, v in data['joint'].items():
        data['joint'][k] = torch.stack(v)
    for k, v in data['imu'].items():
        data['imu'][k] = torch.stack(v)

    data['joint']['name'] = joints
    data['imu']['name'] = imus

    # to smpl coordinate frame

    convert_quaternion_xsens(data['joint']['orientation'])
    convert_point_xsens(data['joint']['position'])
    convert_quaternion_xsens(data['imu']['orientation'])
    convert_point_xsens(data['imu']['free acceleration'])

    if step != 1:
        data['joint']['orientation'] = data['joint']['orientation'][::step].clone()
        data['joint']['position'] = data['joint']['position'][::step].clone()
        data['imu']['free acceleration'] = data['imu']['free acceleration'][::step].clone()
        data['imu']['orientation'] = data['imu']['orientation'][::step].clone()

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]

    imu_ori = imu_calibration(roma.quat_wxyz_to_xyzw(data['imu']['orientation']),
                              roma.quat_wxyz_to_xyzw(data['joint']['orientation']),
                              n_frames_for_calibration,
                              imu_idx)
    data['imu']['calibrated orientation'] = imu_ori

    data['joint']['orientation'] = roma.quat_normalize(roma.quat_wxyz_to_xyzw(data['joint']['orientation']))
    data['joint']['orientation'] = roma.unitquat_to_rotmat(data['joint']['orientation'])

    data['imu']['calibrated orientation'] = roma.quat_normalize(data['imu']['calibrated orientation'])
    data['imu']['calibrated orientation'] = roma.unitquat_to_rotmat(data['imu']['calibrated orientation'])

    data['imu']['orientation'] = roma.quat_normalize(data['imu']['orientation'])
    data['imu']['orientation'] = roma.unitquat_to_rotmat(data['imu']['orientation'])

    indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
    data['joint']['orientation'] = data['joint']['orientation'][:, indices]
    data['joint']['orientation'] = model.inverse_kinematics_R(data['joint']['orientation'])
    data['joint']['position'] = data['joint']['position'][:, indices]
    data['joint']['translation'] = data['joint']['position'][:, 0]


    return data


def read_xlsx(xsens_file_path: str, smpl_file: str):
    r"""
    Load IMU & Joint data from xlsx file.

    :param file: xlsx file path.
    :param smpl_file: path to smpl.pkl model.
    :return: data dictionary containing IMU and Joint data.
    """
    model = ParametricModel(smpl_file)
    pos3s_com, segments_pos3d, segments_quat, \
        imus_ori, imus_free_acc = pd.read_excel(
        xsens_file_path,
        sheet_name=["Center of Mass",
                    "Segment Position",  # positions of joints in 3d space
                    "Segment Orientation - Quat",  # segment global orientation
                    "Sensor Orientation - Quat",  # sensor orientation
                    "Sensor Free Acceleration",  # sensor free acceleration (accelerometer data without gravity vector)
                    ],
        index_col=0
    ).values()

    data = {'framerate': 60.,
            'joint': {'orientation': [], 'position': []},
            'imu': {'free acceleration': [], 'orientation': []},
            }

    # add dim (S, [1], 3)  +  ignore com_vel / com_accel
    pos3s_com = np.expand_dims(pos3s_com.values, axis=1)[..., [0, 1, 2]]
    n_samples = len(pos3s_com)

    # assumes a perfect sampling freq of 60hz
    timestamps = np.arange(1, n_samples + 1) * (1 / 60.)

    segments_pos3d = segments_pos3d.values.reshape(n_samples, -1, 3)
    segments_quat = segments_quat.values.reshape(n_samples, -1, 4)
    imus_free_acc = imus_free_acc.values.reshape(n_samples, -1, 3)
    imus_ori = imus_ori.values.reshape(n_samples, -1, 4)
    mask = torch.tensor([0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21])
    imus_ori = imus_ori[:, mask, :]
    imus_free_acc = imus_free_acc[:, mask, :]

    data['joint']['orientation'] = torch.tensor(segments_quat.astype(np.float32)).clone()
    data['joint']['position'] = torch.tensor(segments_pos3d.astype(np.float32)).clone()
    data['imu']['orientation'] = torch.tensor(imus_ori.astype(np.float32)).clone()
    data['imu']['free acceleration'] = torch.tensor(imus_free_acc.astype(np.float32)).clone()

    data['joint']['orientation'] = convert_quaternion_excel(data['joint']['orientation'])
    data['joint']['position'] = convert_point_excel(data['joint']['position'])
    data['imu']['orientation'] = convert_quaternion_excel(data['imu']['orientation'])
    data['imu']['free acceleration'] = convert_point_excel(data['imu']['free acceleration'])

    data['joint']['name'] = ['Pelvis', 'L5', 'L3', 'T12', 'T8', 'Neck', 'Head', 'RightShoulder', 'RightUpperArm',
                             'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand',
                             'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'RightToe', 'LeftUpperLeg', 'LeftLowerLeg',
                             'LeftFoot', 'LeftToe']
    data['imu']['name'] = ['Pelvis', 'T8', 'Head', 'RightShoulder', 'RightUpperArm', 'RightForeArm', 'RightHand',
                           'LeftShoulder', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg',
                           'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']

    # use first 150 frames for calibration
    n_frames_for_calibration = 150
    imu_idx = [data['joint']['name'].index(_) for _ in data['imu']['name']]

    imu_ori = imu_calibration(roma.quat_wxyz_to_xyzw(data['imu']['orientation']),
                              roma.quat_wxyz_to_xyzw(data['joint']['orientation']),
                              n_frames_for_calibration,
                              imu_idx)
    data['imu']['calibrated orientation'] = imu_ori

    data['joint']['orientation'] = roma.quat_normalize(roma.quat_wxyz_to_xyzw(data['joint']['orientation']))
    data['joint']['orientation'] = roma.unitquat_to_rotmat(data['joint']['orientation'])

    data['imu']['calibrated orientation'] = roma.quat_normalize(data['imu']['calibrated orientation'])
    data['imu']['calibrated orientation'] = roma.unitquat_to_rotmat(data['imu']['calibrated orientation'])

    data['imu']['orientation'] = roma.quat_normalize(data['imu']['orientation'])
    data['imu']['orientation'] = roma.unitquat_to_rotmat(data['imu']['orientation'])

    indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
    data['joint']['orientation'] = data['joint']['orientation'][:, indices]
    data['joint']['orientation'] = model.inverse_kinematics_R(data['joint']['orientation'])
    data['joint']['position'] = data['joint']['position'][:, indices]
    data['joint']['translation'] = data['joint']['position'][:, 0]

    return data

if __name__ == '__main__':
    data_dir = "../data/MTwAwinda/trials/subject02/subject02_task/subject02_task_rep1/subject02_task_rep1.xlsx"
    smpl_dir = "../data/SMPL Model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    data = read_xlsx(data_dir, smpl_file=smpl_dir)
    model = ParametricModel(smpl_dir)
    # po = model.inverse_kinematics_R(data['joint']['orientation'])
    model.view_motion(pose_list=[data['joint']['orientation']],
                      tran_list=[torch.zeros_like(data['joint']['translation'])])
