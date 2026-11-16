"""
Converts AMASS to joint positions in the world frame at an exact rate, one file per sequence.

AMASS stores joint rotations (axis-angle), the root position `trans` and the capture rate; the
models need joint positions. Unlike the original synthetic AMASS
(legacy/scripts/create_synthetic_amass_data.py), the pelvis rotation is kept:
forward kinematics with the real pelvis rotation plus `trans` gives AMASS's own world frame (z up),
with heading, lean and the path across the floor. The body is the template skeleton; betas are
not used.

Resampling: forward kinematics at the capture rate, then positions linearly and the pelvis
rotation by slerp onto the time grid k / rate.

Each file, save_dir/<split>/<path in amass_dir>, holds
    positions        (frames, 52, 3) float32, metres, world frame, z up
    pelvis_rotation  (frames, 3, 3) float32, pelvis orientation in the world frame
    source           path of the AMASS file, relative to amass_dir
    source_rate      capture rate of the AMASS file / Hz
    rate             rate of this file / Hz
"""
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from interact.utils.ang2joint import ang2joint

log = logging.getLogger(__name__)

# frames per forward-kinematics call, bounds the memory for long sequences
FK_CHUNK = 10000


def forward_kinematics(poses, trans, p3d0, parent):
    """Joint positions (frames, joints, 3) in the world from poses (frames, joints * 3), trans (frames, 3)."""
    poses = torch.from_numpy(poses).float().reshape(poses.shape[0], -1, 3)
    chunks = []
    for begin in range(0, poses.shape[0], FK_CHUNK):
        chunk = poses[begin:begin + FK_CHUNK]
        chunks.append(ang2joint(p3d0.expand(chunk.shape[0], -1, -1), chunk, parent))
    return torch.cat(chunks).numpy() + trans[:, None, :]


def axis_angle_to_quaternion(r):
    """(..., 3) axis-angle -> (..., 4) unit quaternion (w, x, y, z)."""
    angle = np.linalg.norm(r, axis=-1, keepdims=True)
    # sin(angle / 2) / angle, finite at angle 0; np.sinc(x) = sin(pi x) / (pi x)
    scale = 0.5 * np.sinc(angle / (2 * np.pi))
    return np.concatenate([np.cos(angle / 2), scale * r], axis=-1)


def quaternion_to_matrix(q):
    """(..., 4) unit quaternion (w, x, y, z) -> (..., 3, 3) rotation matrix."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def slerp(q0, q1, weight):
    """Spherical linear interpolation between unit quaternions (n, 4), weight (n,) in [0, 1]."""
    dot = np.sum(q0 * q1, axis=-1)
    # q and -q are the same rotation: take the shorter way
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sin_angle = np.sin(angle)
    # nearly equal rotations: linear interpolation, normalised below
    close = sin_angle < 1e-6
    safe = np.where(close, 1.0, sin_angle)
    w0 = np.where(close, 1 - weight, np.sin((1 - weight) * angle) / safe)
    w1 = np.where(close, weight, np.sin(weight * angle) / safe)
    q = w0[:, None] * q0 + w1[:, None] * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(positions, quaternions, source_rate, rate):
    """Onto the time grid k / rate, k = 0, 1, ..., up to the last captured frame."""
    last = positions.shape[0] - 1
    count = int(np.floor(last * rate / source_rate)) + 1
    # the grid in capture frames; whole numbers where the two grids coincide
    source_frame = np.arange(count) * (source_rate / rate)
    before = np.minimum(np.floor(source_frame).astype(int), last)
    after = np.minimum(before + 1, last)
    weight = source_frame - before
    positions = ((1 - weight)[:, None, None] * positions[before]
                 + weight[:, None, None] * positions[after])
    return positions, slerp(quaternions[before], quaternions[after], weight)


def convert_sequence(poses, trans, source_rate, rate, p3d0, parent):
    """Positions (frames, joints, 3) and pelvis rotations (frames, 3, 3) at rate, float32."""
    positions = forward_kinematics(poses, trans, p3d0, parent)
    quaternions = axis_angle_to_quaternion(poses[:, :3].astype(np.float64))
    positions, quaternions = resample(positions, quaternions, source_rate, rate)
    return positions.astype(np.float32), quaternion_to_matrix(quaternions).astype(np.float32)


@hydra.main(version_base=None, config_path='../config', config_name='amass_conversion')
def convert(cfg: DictConfig):
    amass_dir = Path(cfg.amass_dir)
    save_dir = Path(cfg.save_dir)
    skeleton = np.load(cfg.body_models_dir)
    p3d0 = torch.from_numpy(skeleton['p3d0']).float().reshape(1, -1, 3)
    parent = {i: int(p) for i, p in enumerate(skeleton['parents'])}

    splits = {'train': cfg.train_datasets, 'val': cfg.val_datasets, 'test': cfg.test_datasets}
    for split, datasets in splits.items():
        for dataset in datasets:
            dataset_dir = amass_dir / dataset
            if not dataset_dir.is_dir():
                log.warning(f'{dataset_dir} not found, skipped')
                continue
            converted = frames = 0
            for source in sorted(dataset_dir.rglob('*.npz')):
                with np.load(source) as data:
                    if 'poses' not in data:
                        # e.g. CMU's shape.npz: gender and betas only, no motion
                        log.warning(f'no poses in {source}, skipped')
                        continue
                    poses, trans = data['poses'], data['trans']
                    source_rate = float(data['mocap_framerate'])
                positions, pelvis_rotation = convert_sequence(
                    poses, trans, source_rate, cfg.rate, p3d0, parent)

                relative = source.relative_to(amass_dir)
                target = save_dir / split / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, positions=positions, pelvis_rotation=pelvis_rotation,
                         source=str(relative), source_rate=source_rate, rate=float(cfg.rate))
                converted += 1
                frames += positions.shape[0]
            log.info(f'{split} {dataset}: {converted} sequences, {frames} frames '
                     f'({frames / cfg.rate / 3600:.1f} h)')


if __name__ == '__main__':
    convert()
