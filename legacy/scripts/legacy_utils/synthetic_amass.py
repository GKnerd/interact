import logging

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from interact.utils.read_json_data import read_json
# from read_json_data import read_json, get_pose_history

log = logging.getLogger(__name__)

class Synthetic_AMASS(Dataset):

    def __init__(
            self,
            data_dir,
            mapping_json,
            input_n = 15,
            output_n = 15,
            sample_rate = 15,
            output_rate = 15,
            split='train'
            ):
        self.data_dir = data_dir
        self.input_frames = input_n
        self.output_frames = output_n
        self.sample_rate = sample_rate
        self.output_rate = output_rate
        self.split = split
        self.mapping_json = mapping_json

        self.alice_input, self.alice_output = [], []
        self.bob_input, self.bob_output = [], []
        self.sequence_len = input_n + output_n
        self.input_n = input_n
        self.output_n = output_n

        joint_names = ['Neck', 
            'L_Shoulder', 'R_Shoulder',
            'L_Elbow', 'R_Elbow',
            'L_Wrist', 'R_Wrist', 
            'L_Hand', 'R_Hand']

        mapping = read_json(self.mapping_json)
        self.joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        self.add_amass_dataset()

    def add_amass_dataset(self):
        split_dir = f'{self.data_dir}/{self.split}'
        # sorted, so the window order does not depend on the file system
        episodes = sorted(f for f in os.listdir(split_dir) if f.endswith('.npz'))
        if not episodes:
            raise FileNotFoundError(f'no .npz files in {split_dir}')
        # which file each window comes from: self.episodes[self.episode_index[i]] for window i
        self.episodes = episodes
        window_counts = []
        for episode in episodes:
            with np.load(f'{split_dir}/{episode}') as data:
                alice_all = torch.from_numpy(data['alice'])
                bob_all = torch.from_numpy(data['bob'])

            downsample_rate = self.sample_rate // self.output_rate
            alice_tensor = alice_all[::downsample_rate, self.joint_used]
            bob_tensor = bob_all[::downsample_rate, self.joint_used]
            # chop the tensor into a bunch of slices of size sequence_len
            for start_frame in range(alice_tensor.shape[0]-self.sequence_len):
                end_frame = start_frame + self.sequence_len
                self.alice_input.append(alice_tensor[start_frame:start_frame+self.input_n])
                self.alice_output.append(alice_tensor[start_frame+self.input_n:end_frame])

                self.bob_input.append(bob_tensor[start_frame:start_frame+self.input_n])
                self.bob_output.append(bob_tensor[start_frame+self.input_n:end_frame])
            window_counts.append(max(alice_tensor.shape[0] - self.sequence_len, 0))
        self.episode_index = np.repeat(np.arange(len(episodes)), window_counts)
        log.info(f'{len(self.alice_input)} windows in {self.data_dir}/{self.split}')

    def __len__(self):
        return len(self.alice_input)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.alice_input[idx], self.alice_output[idx], self.bob_input[idx], self.bob_output[idx]

if __name__ == "__main__":
    dataset = Synthetic_AMASS()
