"""
Centring of the pretraining windows before they go into the model, chosen by `centring` in
config/pretraining.yaml and shared by pretrain_intent_forecaster.py and its evaluation.

- joint: each joint minus its own last history position (script_utils.pretraining.centre); the
  last history frame is zero for every joint, so the pose is hidden. Runs 0 and 1.
- pelvis: no centring. In the window frame the origin is already the pelvis at the last history
  frame, so the pose stays visible. Ablation run 2.
"""
import torch

from script_utils.pretraining import centre

CENTRINGS = ('joint', 'pelvis')


def centre_on_pelvis(batch):
    """
    Flattened to (B, T, J * 3) without centring: in the window frame the origin is already the
    pelvis at the last history frame.

    :param batch: (history, future), each (B, T, J, 3), in the window frame.
    :return: alice_hist, alice_fut, bob_hist, bob_fut; Bob is zeros.
    """
    hist, fut = [b.reshape(b.shape[0], b.shape[1], -1) for b in batch]
    return hist, fut, torch.zeros_like(hist), torch.zeros_like(fut)


def get_centring(centring, frame):
    """
    The centring function of a run.

    :param centring: one of CENTRINGS.
    :param frame: the run's AMASS frame; pelvis centring needs 'window'.
    :return: function batch -> (alice_hist, alice_fut, bob_hist, bob_fut).
    """
    if centring not in CENTRINGS:
        raise ValueError(f'unknown centring {centring!r}, expected one of {CENTRINGS}')
    if centring == 'pelvis' and frame != 'window':
        raise ValueError(f"centring 'pelvis' needs frame 'window' (the pelvis is its origin), "
                         f'got {frame!r}')
    return centre if centring == 'joint' else centre_on_pelvis
