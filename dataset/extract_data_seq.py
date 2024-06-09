import glob
import math
import os

import numpy as np
import tqdm
import torch

def extract_seq(phase, data_path, seq_length, overlap):
    data_dir = os.path.join(data_path, phase, "*/*/*.pt")
    data_dirs = glob.glob(data_dir)
    loop = tqdm.tqdm(data_dirs)
    for data_dir in loop:

        data = torch.load(data_dir)
        length = data['pose'].shape[0]
        if length < seq_length:
            continue

        loop.set_description(f"Extracting sequences from {data_dir} with seq length {seq_length}")

        target_folder = os.path.join(data_path, f"{phase}_seq", f"seq_{seq_length}")
        os.makedirs(target_folder, exist_ok=True)
        num_sequences = int(math.ceil((length - overlap) / (seq_length - overlap)))
        for idx in range(num_sequences):
            start_index = idx * (seq_length - overlap)
            end_index = start_index + seq_length
            len_data = data['pose'][start_index:end_index].shape[0]
            if len_data == seq_length:

                targets = {
                    'pose': data['pose'][start_index:end_index].detach().numpy(),
                    'javel': data['javel'][start_index:end_index].detach().numpy(),
                    'jlacc': data['jlacc'][start_index:end_index].detach().numpy(),
                    'jvel': data['jvel'][start_index:end_index].detach().numpy(),
                    'grot': data['grot'][start_index:end_index].detach().numpy(),
                    'fix_joint': data['fix_joint'][start_index:end_index].detach().numpy(),
                    'bone_length': data['bone_length'].detach().numpy(),
                    'dataset': data['dataset'],
                }
                if "imu_acc" and "imu_ori" in data.keys():
                    targets['imu_acc'] = data['imu_acc'][start_index:end_index].detach().numpy()
                    targets['imu_ori'] = data['imu_ori'][start_index:end_index].detach().numpy()
                else:
                    targets['imu_acc'] = data['vacc'][start_index:end_index].detach().numpy()
                    targets['imu_ori'] = data['vrot'][start_index:end_index].detach().numpy()
                name = [f"Seq_{idx+1}"] + data_dir.split("\\")[-3:-1] + [os.path.basename(data_dir).replace(".pt", "")]
                name = "_".join(name)
                np.savez(os.path.join(target_folder, name), **targets)



if __name__ == '__main__':
    data_dir = '../data/'
    extract_seq(data_path=data_dir, phase="Train", seq_length=256, overlap=32)
    extract_seq(data_path=data_dir, phase="Valid", seq_length=256, overlap=32)
    # data = np.load("../data/Train_seq/seq_256/Seq_1_ACCAD_process_data_Female1General_c3d_A2 - Sway t2_poses.npz")