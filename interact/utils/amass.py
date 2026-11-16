"""
Windows of AMASS motion, read from the files of scripts/convert_amass.py.

A window holds history_len + horizon_len frames of the chosen joints. In the window frame (the
default) they are relative to the person's pelvis and heading at the last history frame
(interact/utils/body_frame.py); in the world frame they are as converted (z up, metres).
"""
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from interact.constants import AMASS_FRAMES, MIN_HIP_LINE_LENGTH, SMPL_HIP_AXIS
from interact.utils.body_frame import heading_frame, to_frame
from interact.utils.read_json_data import read_json

log = logging.getLogger(__name__)


class AMASS(Dataset):

    def __init__(
            self,
            data_dir,
            split,
            mapping_json,
            joint_names,
            history_len=15,
            horizon_len=15,
            frame='window',
            min_hip_line_length=MIN_HIP_LINE_LENGTH,
            ):
        """
        :param data_dir: save_dir of scripts/convert_amass.py, holding one folder per split.
        :param joint_names: names in mapping_json, in the order of the output's joint axis.
        :param frame: 'window' or 'world'.
        :param min_hip_line_length: window frame only; windows whose heading at the last history
            frame is less defined than this are left out.
        """
        if frame not in AMASS_FRAMES:
            raise ValueError(f'frame must be one of {AMASS_FRAMES}, not {frame!r}')
        self.history_len = history_len
        self.horizon_len = horizon_len
        self.frame = frame
        mapping = read_json(mapping_json)
        joints = np.array([mapping[name] for name in joint_names])
        pelvis = mapping['Pelvis']

        split_dir = Path(data_dir) / split
        files = sorted(split_dir.rglob('*.npz'))
        if not files:
            raise FileNotFoundError(f'no .npz files in {split_dir}')
        # which file each window comes from: self.episodes[self.episode_index[i]] for window i
        self.episodes = [str(file.relative_to(split_dir)) for file in files]
        self.positions, self.pelvis, self.hip_line = [], [], []
        starts, episode_index = [], []
        window_len = history_len + horizon_len
        too_short = undefined_heading = 0
        for episode, file in enumerate(files):
            with np.load(file) as data:
                positions = data['positions']
                hip_line = data['pelvis_rotation'][:, :, SMPL_HIP_AXIS]
            self.positions.append(np.ascontiguousarray(positions[:, joints]))
            self.pelvis.append(np.ascontiguousarray(positions[:, pelvis]))
            self.hip_line.append(np.ascontiguousarray(hip_line))

            window_count = positions.shape[0] - window_len + 1
            if window_count <= 0:
                too_short += 1
                continue
            start = np.arange(window_count)
            if frame == 'window':
                _, length = heading_frame(hip_line[start + history_len - 1])
                defined = length >= min_hip_line_length
                undefined_heading += int(np.count_nonzero(~defined))
                start = start[defined]
            starts.append(start)
            episode_index.append(np.full(len(start), episode))
        if not starts:
            raise ValueError(f'no sequence in {split_dir} fills one window of {window_len} frames')
        self.starts = np.concatenate(starts)
        self.episode_index = np.concatenate(episode_index)
        log.info(f'{len(self.starts)} windows from {len(files)} sequences in {split_dir}; '
                 f'{too_short} sequences shorter than one window, '
                 f'{undefined_heading} windows left out for an undefined heading')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        """History (history_len, joints, 3) and future (horizon_len, joints, 3), float32."""
        episode = self.episode_index[idx]
        start = self.starts[idx]
        window = self.positions[episode][start:start + self.history_len + self.horizon_len]
        if self.frame == 'window':
            now = start + self.history_len - 1
            rotation, _ = heading_frame(self.hip_line[episode][now])
            window = to_frame(window, self.pelvis[episode][now], rotation)
        # a copy, so the tensor does not share memory with the stored sequence
        window = torch.from_numpy(np.array(window, dtype=np.float32))
        return window[:self.history_len], window[self.history_len:]
