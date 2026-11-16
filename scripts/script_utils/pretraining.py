"""
Data of the pretraining runs, shared by pretrain_intent_forecaster.py and its evaluation, so both
build and centre the windows the same way.

The data is AMASS from convert_amass.py (z up). It holds one person; the partner inputs are zeros,
so only a model that ignores the partner (Marginal) can train on it. The original synthetic
two-person AMASS (y up) is in legacy/; the state of the comparison is the git tag
synthetic-amass-baseline.
"""
import torch

from interact.utils.amass import AMASS


def build_dataset(cfg, split, data_dir=None):
    """
    The dataset of a run's settings.

    :param cfg: the training config (DictConfig, or the dict of a run's .hydra/config.yaml).
    :param data_dir: overrides cfg's data_dir.
    """
    return AMASS(data_dir or cfg['data_dir'], split, cfg['mapping_json'],
                 list(cfg['amass']['joint_names']), frame=cfg['amass']['frame'])


def check_model(model_config):
    """The data has no partner: only a model that ignores the partner (Marginal) can train on it."""
    model = model_config['model']
    if not model.get('one_hist') or model.get('conditional_forecaster'):
        raise ValueError('AMASS has no partner: the model needs one_hist true and '
                         'conditional_forecaster false (Marginal)')


def centre(batch):
    """
    Each joint minus its own last history position, flattened to (B, T, J * 3).

    :param batch: (history, future), each (B, T, J, 3).
    :return: alice_hist, alice_fut, bob_hist, bob_fut; Bob is zeros.
    """
    hist, fut = [b.reshape(b.shape[0], b.shape[1], -1) for b in batch]
    offset = hist[:, -1:]
    hist, fut = hist - offset, fut - offset
    return hist, fut, torch.zeros_like(hist), torch.zeros_like(fut)
