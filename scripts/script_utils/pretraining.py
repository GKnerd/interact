"""
Data of the pretraining runs, shared by pretrain_intent_forecaster.py and its evaluation, so both
build and centre the windows the same way.

Datasets (the training config's `dataset`):
- synthetic_amass: Alice and Bob from create_synthetic_amass_data.py, y up.
- amass: Alice only, from convert_amass.py, z up. Bob is zeros, so only Marginal can train on it.
"""
import torch

from interact.utils.amass import AMASS
from interact.utils.synthetic_amass import Synthetic_AMASS

DATASETS = ('synthetic_amass', 'amass')
UP_AXIS = {'synthetic_amass': 'y', 'amass': 'z'}

# (data, dataset the model was trained on): axis order that turns the data into the model's axes.
# Window frame of amass: x forward, y left, z up. Synthetic AMASS has the SMPL template's axes,
# with every body facing forward: x left, y up, z forward.
AXIS_ORDER = {('amass', 'synthetic_amass'): (1, 2, 0)}


def check_dataset(name):
    if name not in DATASETS:
        raise ValueError(f'dataset must be one of {DATASETS}, not {name!r}')
    return name


def dataset_name(cfg):
    """The dataset of a run's settings; runs from before the option read synthetic AMASS."""
    return check_dataset(cfg.get('dataset', 'synthetic_amass'))


def build_dataset(cfg, split, data_dir=None, dataset=None):
    """
    The dataset of a run's settings.

    :param cfg: the training config (DictConfig, or the dict of a run's .hydra/config.yaml).
    :param data_dir: overrides cfg's data_dir.
    :param dataset: overrides cfg's dataset.
    """
    name = check_dataset(dataset) if dataset else dataset_name(cfg)
    data_dir = data_dir or cfg['data_dir']
    if name == 'synthetic_amass':
        return Synthetic_AMASS(data_dir, cfg['mapping_json'], split=split)
    if 'amass' not in cfg:
        raise ValueError('the config has no amass section (joint_names, frame)')
    return AMASS(data_dir, split, cfg['mapping_json'], list(cfg['amass']['joint_names']),
                 frame=cfg['amass']['frame'])


def check_model(cfg, model_config):
    """amass has no partner: only a model that ignores the partner (Marginal) can train on it."""
    model = model_config['model']
    if dataset_name(cfg) == 'amass' and (not model.get('one_hist')
                                         or model.get('conditional_forecaster')):
        raise ValueError('dataset amass has no partner: the model needs one_hist true and '
                         'conditional_forecaster false (Marginal)')


def centre(batch):
    """
    Each joint minus its own last history position, flattened to (B, T, J * 3).

    :param batch: (alice_hist, alice_fut, bob_hist, bob_fut) or, from amass, (alice_hist, alice_fut),
        each (B, T, J, 3).
    :return: alice_hist, alice_fut, bob_hist, bob_fut; without Bob, Bob is zeros.
    """
    flat = [b.reshape(b.shape[0], b.shape[1], -1) for b in batch]
    offset = flat[0][:, -1:]
    centred = [b - offset for b in flat]
    if len(centred) == 2:
        centred += [torch.zeros_like(b) for b in centred]
    return centred
