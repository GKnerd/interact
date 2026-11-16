"""Evaluate a run of scripts/pretrain_intent_forecaster.py on synthetic AMASS or AMASS.

Reads the run folder: the run's settings (.hydra/config.yaml), model.yaml and checkpoints/.
Every checkpoint is evaluated on the val split. With test=true, the best-val and the last
checkpoint are also evaluated on the test split. Windows, centring and forward pass are the ones
of training (script_utils/pretraining.py: each joint minus its own last history position,
model(alice_hist, bob_hist, bob_future)).

Data: the run's own dataset, or with dataset=amass the AMASS data for a run trained on synthetic
AMASS (cross-evaluation). Its axes are then reordered to the run's (AXIS_ORDER), so both models
can be compared on the same windows.

Metrics, computed per window and averaged over windows:
- MPJPE / mm: joint position error averaged over the 15 forecast steps (the training loss)
- FDE / mm: joint position error at the last forecast step, t = 1 s (the paper's metric)
- both over all 9 joints and over wrists and hands (joints 5-8, as in eval_hh.py)
- error per forecast step, per joint, FDE quantiles, FDE per bin of motion speed
- heading error / degree: angle between the true and the forecast torso heading (perpendicular
  to the shoulder line on the floor), per step and at t = 1 s; from the uncentred positions
- 95 % confidence intervals by bootstrap over sequences: windows of one sequence overlap
  in 29 of 30 frames, so sequences, not windows, are resampled
- references: zero velocity (the person stays) and constant velocity (the last step continues)

The best checkpoint is the one with the lowest val MPJPE (all joints).

Writes CSV tables and PDF figures to <run_dir>/evaluation/<date_time>/.

Run in the container:
    python scripts/pretrain_intent_forecaster_evaluation.py run_dir=/home/interact/outputs/runs/<run>
    python scripts/pretrain_intent_forecaster_evaluation.py run_dir=... test=true
"""
import csv
import logging
import pathlib

import hydra
import numpy as np
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from matplotlib.ticker import MaxNLocator
from omegaconf import DictConfig, OmegaConf

from interact.api import build_model
from script_utils import plot_style as ps
from script_utils.pretraining import (AXIS_ORDER, UP_AXIS, build_dataset, centre, check_dataset,
                                      dataset_name)

log = logging.getLogger(__name__)

METHODS = ['model', 'zero_velocity', 'constant_velocity']
REFERENCES = ['zero_velocity', 'constant_velocity']
GROUPS = {'all': slice(None), 'wrists_hands': slice(5, 9)}
QUANTILES = (50, 90, 95, 99)
EXAMPLE_PERCENTILES = (50, 90, 99)


# ---------------------------------------------------------------- run and data

def load_run(run_dir):
    run_cfg = OmegaConf.to_container(OmegaConf.load(run_dir / '.hydra' / 'config.yaml'),
                                     resolve=True)
    model_config = yaml.safe_load((run_dir / 'model.yaml').read_text())
    checkpoints = {int(p.stem): p for p in (run_dir / 'checkpoints').glob('*.model')}
    if not checkpoints:
        raise FileNotFoundError(f'no checkpoints in {run_dir / "checkpoints"}')
    return run_cfg, model_config, dict(sorted(checkpoints.items()))


def train_curve(run_dir):
    """Training MPJPE per epoch in mm, from the run's TensorBoard log."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        acc = EventAccumulator(str(run_dir / 'tensorboard'))
        acc.Reload()
        return {int(e.step): e.value * 1000 for e in acc.Scalars('train/mpjpe')}
    except Exception as e:
        log.warning(f'no training curve from {run_dir / "tensorboard"}: {e}')
        return {}


class ReorderedAxes:
    """A dataset with its x, y, z reordered to the axes a model was trained on."""

    def __init__(self, ds, order):
        self.ds = ds
        self.order = list(order)
        self.episode_index = ds.episode_index

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return tuple(x[..., self.order] for x in self.ds[i])


def stack(ds, start, stop):
    """Windows start to stop as the DataLoader of training batches them, each (B, T, J, 3)."""
    return [torch.stack(x) for x in zip(*(ds[i] for i in range(start, stop)))]


# ---------------------------------------------------------------- forecasters

def model_forecaster(model):
    # the model zeroes Bob's future in place when it is not conditional; keep the batch intact
    return lambda alice_hist, bob_hist, bob_future: model(alice_hist, bob_hist, bob_future.clone())


def zero_velocity(horizon):
    return lambda alice_hist, *_: alice_hist[:, -1:].expand(-1, horizon, -1)


def constant_velocity(horizon):
    def forecast(alice_hist, *_):
        step = alice_hist[:, -1:] - alice_hist[:, -2:-1]
        k = torch.arange(1, horizon + 1, device=alice_hist.device,
                         dtype=alice_hist.dtype).view(1, -1, 1)
        return alice_hist[:, -1:] + k * step
    return forecast


def heading_error(pred, truth, shoulders, up):
    """
    Angle between forecast and true torso heading in degrees, (B, steps).

    The heading is perpendicular to the shoulder line on the floor, so its error is the angle
    between the two shoulder lines projected onto the floor.

    :param pred, truth: uncentred positions (B, steps, J, 3).
    :param shoulders: indices of the left and the right shoulder.
    :param up: index of the up axis.
    """
    lines = []
    for x in (pred, truth):
        line = x[..., shoulders[0], :] - x[..., shoulders[1], :]
        line[..., up] = 0
        lines.append(line)
    a, b = lines
    return torch.rad2deg(torch.atan2(torch.linalg.cross(a, b).norm(dim=-1), (a * b).sum(dim=-1)))


@torch.no_grad()
def errors(ds, forecast, device, batch_size, shoulders, up):
    """Joint position error in mm (windows, forecast steps, joints) and heading error in degrees
    (windows, forecast steps)."""
    n = len(ds)
    err = heading = None
    for start in range(0, n, batch_size):
        raw = [b.to(device) for b in stack(ds, start, min(start + batch_size, n))]
        alice_hist, alice_fut, bob_hist, bob_fut = centre(raw)
        batch, steps = alice_fut.shape[:2]
        pred = forecast(alice_hist, bob_hist, bob_fut).reshape(batch, steps, -1, 3)
        e = (pred - alice_fut.reshape(batch, steps, -1, 3)).norm(dim=-1) * 1000
        # headings need the real shoulder line: undo the per-joint centring
        h = heading_error(pred + raw[0][:, -1:], raw[1], shoulders, up)
        if err is None:
            err = np.empty((n, steps, e.shape[-1]), np.float32)
            heading = np.empty((n, steps), np.float32)
        err[start:start + batch] = e.cpu().numpy()
        heading[start:start + batch] = h.cpu().numpy()
    return err, heading


def history_speed(ds, rate_hz, batch_size):
    """Mean joint speed over the history of each window, in m/s."""
    n = len(ds)
    out = np.empty(n, np.float32)
    for start in range(0, n, batch_size):
        hist = stack(ds, start, min(start + batch_size, n))[0]           # (B, T, J, 3)
        out[start:start + len(hist)] = (
            (hist[:, 1:] - hist[:, :-1]).norm(dim=-1).mean(dim=(1, 2)) * rate_hz).numpy()
    return out


# ---------------------------------------------------------------- statistics

class Bootstrap:
    """95 % intervals of window means, resampling whole sequences.

    The same resamples are used for every method and metric, so differences are paired.
    """

    def __init__(self, episode_index, samples, seed):
        if np.any(np.diff(episode_index) < 0):
            raise ValueError('windows are not grouped by sequence')
        self.starts = np.flatnonzero(np.r_[True, np.diff(episode_index) != 0])
        self.n_sequences = len(self.starts)
        self.counts = np.diff(np.r_[self.starts, len(episode_index)]).astype(np.float64)
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(
            self.n_sequences, np.full(self.n_sequences, 1 / self.n_sequences),
            size=samples).astype(np.float64)

    def mean_ci(self, x):
        """Mean over windows and its 95 % interval; x has shape (windows, ...)."""
        x = np.asarray(x, np.float64)
        sums = np.add.reduceat(x, self.starts, axis=0).reshape(self.n_sequences, -1)
        boot = (self.weights @ sums) / (self.weights @ self.counts)[:, None]
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        shape = x.shape[1:]
        return x.mean(axis=0), low.reshape(shape), high.reshape(shape)


def summarise(err, heading, boot):
    s = {'heading': dict(final=boot.mean_ci(heading[:, -1]), step=boot.mean_ci(heading))}
    for group, joints in GROUPS.items():
        e = err[:, :, joints].mean(axis=2)                     # (windows, steps)
        fde = e[:, -1]
        s[group] = dict(mpjpe=boot.mean_ci(e.mean(axis=1)), fde=boot.mean_ci(fde),
                        step=boot.mean_ci(e), fde_window=fde,
                        quantiles=np.percentile(fde, QUANTILES))
    s['joint_mpjpe'] = boot.mean_ci(err.mean(axis=1))
    s['joint_fde'] = boot.mean_ci(err[:, -1])
    return s


def speed_bins(speed, n_bins):
    edges = np.quantile(speed, np.linspace(0, 1, n_bins + 1))
    index = np.clip(np.searchsorted(edges, speed, side='right') - 1, 0, n_bins - 1)
    return edges, index


# ---------------------------------------------------------------- tables

def r(x):
    return round(float(x), 3)


def metric_rows(split, epoch, method, s, n_windows, n_sequences):
    for group in GROUPS:
        m, f = s[group]['mpjpe'], s[group]['fde']
        # the heading belongs to the whole body: in the rows of all joints only
        h = [r(v) for v in s['heading']['final']] if group == 'all' else ['', '', '']
        yield dict(split=split, epoch=epoch, method=method, group=group,
                   n_windows=n_windows, n_sequences=n_sequences,
                   mpjpe_mm=r(m[0]), mpjpe_ci_low=r(m[1]), mpjpe_ci_high=r(m[2]),
                   fde_mm=r(f[0]), fde_ci_low=r(f[1]), fde_ci_high=r(f[2]),
                   **{f'fde_p{q}_mm': r(v) for q, v in zip(QUANTILES, s[group]['quantiles'])},
                   heading_deg=h[0], heading_ci_low=h[1], heading_ci_high=h[2])


def horizon_rows(split, epoch, method, s, rate_hz):
    h_mean, h_low, h_high = s['heading']['step']
    for group in GROUPS:
        mean, low, high = s[group]['step']
        for k in range(len(mean)):
            h = [r(h_mean[k]), r(h_low[k]), r(h_high[k])] if group == 'all' else ['', '', '']
            yield dict(split=split, epoch=epoch, method=method, group=group, step=k + 1,
                       t_s=r((k + 1) / rate_hz), error_mm=r(mean[k]),
                       ci_low=r(low[k]), ci_high=r(high[k]),
                       heading_deg=h[0], heading_ci_low=h[1], heading_ci_high=h[2])


def joint_rows(split, epoch, method, s, joint_names):
    m, f = s['joint_mpjpe'], s['joint_fde']
    for j, name in enumerate(joint_names):
        yield dict(split=split, epoch=epoch, method=method, joint=name,
                   mpjpe_mm=r(m[0][j]), mpjpe_ci_low=r(m[1][j]), mpjpe_ci_high=r(m[2][j]),
                   fde_mm=r(f[0][j]), fde_ci_low=r(f[1][j]), fde_ci_high=r(f[2][j]))


def speed_rows(split, epoch, method, s, speed, edges, index):
    fde = s['all']['fde_window']
    for b in range(len(edges) - 1):
        sel = index == b
        yield dict(split=split, epoch=epoch, method=method, bin=b + 1,
                   speed_low_m_s=r(edges[b]), speed_high_m_s=r(edges[b + 1]),
                   speed_median_m_s=r(np.median(speed[sel])), n_windows=int(sel.sum()),
                   fde_mm=r(fde[sel].mean()))


def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------- figures

def line_style(method, markers=True):
    st = ps.METHOD_STYLE[method]
    return dict(color=st['color'], linestyle=st['linestyle'], label=st['label'],
                marker=st['marker'] if markers else None)


def legend_on_top(fig, axes, ncols=3):
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=ncols)


def plot_epochs(path, epochs, model_s, ref_s, train, best):
    fig, axes = ps.figure(100, nrows=2, sharex=True)
    for ax, metric, label in zip(axes, ('mpjpe', 'fde'), (ps.LABEL_MPJPE, ps.LABEL_FDE)):
        mean = np.array([model_s[e]['all'][metric][0] for e in epochs])
        low = np.array([model_s[e]['all'][metric][1] for e in epochs])
        high = np.array([model_s[e]['all'][metric][2] for e in epochs])
        st = line_style('model')
        ax.fill_between(epochs, low, high, color=st['color'], alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, **{**st, 'label': 'Model, val'})
        if metric == 'mpjpe' and train:
            e = sorted(train)
            ax.plot(e, [train[k] for k in e], color=ps.MUTED, marker='o',
                    label='Model, train')
        for method in REFERENCES:
            st = line_style(method, markers=False)
            ax.axhline(ref_s[method]['all'][metric][0], **{**st, 'label': f'{st["label"]}, val'})
        i = epochs.index(best)
        ax.annotate('best', (best, mean[i]), xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, color=ps.INK_SECONDARY)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel(ps.LABEL_EPOCH)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_on_top(fig, axes[0], ncols=2)
    ps.save(fig, path, 'Validation error per epoch')


def plot_horizon(path, summaries, rate_hz, title):
    fig, axes = ps.figure(100, nrows=2, sharex=True)
    for ax, group in zip(axes, GROUPS):
        for method in METHODS:
            mean, low, high = summaries[method][group]['step']
            t = np.arange(1, len(mean) + 1) / rate_hz
            st = line_style(method)
            ax.fill_between(t, low, high, color=st['color'], alpha=0.15, linewidth=0)
            ax.plot(t, mean, **st)
        ax.set_title(ps.GROUP_LABEL[group], loc='left')
        ax.set_ylabel(ps.LABEL_ERROR)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel(ps.LABEL_TIME)
    legend_on_top(fig, axes[0])
    ps.save(fig, path, title)


def plot_heading(path, summaries, rate_hz, title):
    fig, ax = ps.figure(65)
    for method in METHODS:
        mean, low, high = summaries[method]['heading']['step']
        t = np.arange(1, len(mean) + 1) / rate_hz
        st = line_style(method)
        ax.fill_between(t, low, high, color=st['color'], alpha=0.15, linewidth=0)
        ax.plot(t, mean, **st)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(ps.LABEL_TIME)
    ax.set_ylabel(ps.LABEL_HEADING)
    legend_on_top(fig, ax)
    ps.save(fig, path, title)


def plot_joints(path, summaries, joint_names, title):
    fig, ax = ps.figure(85)
    y = np.arange(len(joint_names))
    height = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        st = ps.METHOD_STYLE[method]
        ax.barh(y + (i - (len(METHODS) - 1) / 2) * height, summaries[method]['joint_fde'][0],
                height=height * 0.85, color=st['color'], label=st['label'], linewidth=0)
    ax.set_yticks(y, [n[0].upper() + n[1:] for n in joint_names])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(ps.LABEL_FDE)
    legend_on_top(fig, ax)
    ps.save(fig, path, title)


def plot_fde_distribution(path, summaries, title):
    fig, ax = ps.figure(65)
    x_max = max(np.percentile(summaries[m]['all']['fde_window'], 99) for m in METHODS)
    grid = np.linspace(0, x_max, 600)
    for method in METHODS:
        fde = np.sort(summaries[method]['all']['fde_window'])
        share = np.searchsorted(fde, grid, side='right') / len(fde) * 100
        ax.plot(grid, share, **line_style(method, markers=False))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 100)
    ax.set_xlabel(ps.LABEL_FDE)
    ax.set_ylabel(ps.LABEL_SHARE)
    legend_on_top(fig, ax)
    ps.save(fig, path, title)


def plot_speed(path, summaries, speed, edges, index, title):
    fig, ax = ps.figure(65)
    x = [np.median(speed[index == b]) for b in range(len(edges) - 1)]
    for method in METHODS:
        fde = summaries[method]['all']['fde_window']
        ax.plot(x, [fde[index == b].mean() for b in range(len(edges) - 1)],
                **line_style(method))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(ps.LABEL_SPEED)
    ax.set_ylabel(ps.LABEL_FDE)
    legend_on_top(fig, ax)
    ps.save(fig, path, title)


@torch.no_grad()
def plot_examples(path, ds, model, model_err, joint, joint_name, up_axis, device, title):
    """Columns: windows at the median, P90 and P99 of the model's FDE.
    Rows: top view and a view from the side, for data with y or z up."""
    fde = model_err[:, -1].mean(axis=1)
    order = np.argsort(fde)
    if up_axis == 'y':
        views = [(0, 2, 'x', 'z'), (0, 1, 'x', 'y')]
        view_names = 'top row: top view, bottom row: front view'
    else:
        views = [(0, 1, 'x', 'y'), (0, 2, 'x', 'z')]
        view_names = 'top row: top view, bottom row: side view'
    fig, axes = ps.figure(115, width=ps.TEXT_WIDTH, nrows=2, ncols=len(EXAMPLE_PERCENTILES))
    for col, q in enumerate(EXAMPLE_PERCENTILES):
        i = order[int(round(q / 100 * (len(order) - 1)))]
        alice_hist, alice_fut, bob_hist, bob_fut = [b.to(device) for b in centre(stack(ds, i, i + 1))]
        pred = model(alice_hist, bob_hist, bob_fut.clone())
        hist, truth, pred = [x.reshape(x.shape[1], -1, 3)[:, joint].cpu().numpy() * 1000
                             for x in (alice_hist, alice_fut, pred)]
        # forecasts start at the current position, the last history frame
        truth, pred = [np.vstack([hist[-1:], x]) for x in (truth, pred)]
        for row, (a, b, name_a, name_b) in enumerate(views):
            ax = axes[row, col]
            ax.plot(hist[:, a], hist[:, b], color=ps.MUTED, marker='o', markersize=2.5,
                    label='History')
            ax.plot(truth[:, a], truth[:, b], color=ps.INK, marker='o', markersize=2.5,
                    label='Ground truth')
            ax.plot(pred[:, a], pred[:, b], **line_style('model'))
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel(f'${name_a}$ / mm')
            ax.set_ylabel(f'${name_b}$ / mm')
        label = 'Median window' if q == 50 else f'P{q} window'
        axes[0, col].set_title(f'{label}, $e_\\mathrm{{FDE}}$ = {fde[i]:.0f} mm', loc='left')
    fig.suptitle(f'{joint_name[0].upper() + joint_name[1:]}, relative to its current position; '
                 f'{view_names}')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3)
    ps.save(fig, path, title)


# ---------------------------------------------------------------- main

@hydra.main(version_base=None, config_path='../config', config_name='pretrain_evaluation')
def main(cfg: DictConfig) -> None:
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir()
    run_dir = pathlib.Path(cfg.run_dir)
    run_cfg, model_config, checkpoints = load_run(run_dir)
    trained_on = dataset_name(run_cfg)
    dataset = check_dataset(cfg.dataset or trained_on)
    axis_order = AXIS_ORDER.get((dataset, trained_on))
    if dataset != trained_on and axis_order is None:
        raise ValueError(f'a run trained on {trained_on} cannot be evaluated on {dataset}')
    if cfg.data_dir:
        data_dir = cfg.data_dir
    elif dataset == trained_on:
        data_dir = run_cfg['data_dir']
    else:
        data_dir = run_cfg.get('data_dirs', {}).get(dataset)
        if data_dir is None:
            raise ValueError(f'the run config has no data_dirs.{dataset}: set data_dir=')
    rate_hz = model_config.get('rate_hz', 15)
    joint_names = [n.replace('_', ' ') for n in model_config['joint_names']]
    shoulders = tuple(model_config['joint_names'].index(n)
                      for n in ('left_shoulder', 'right_shoulder'))
    # after reordering, the data has the axes of the run's training data
    up_axis = UP_AXIS[trained_on]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'run {run_dir} (trained on {trained_on}), model {model_config["model_class"]}, '
             f'data {data_dir} ({dataset}), checkpoints {list(checkpoints)}, device {device}')
    if axis_order is not None:
        log.info(f'data axes reordered to the run\'s: {axis_order}')
    log.info(f'wrists and hands: {joint_names[GROUPS["wrists_hands"]]}')

    def tagged(rows):
        """Rows with the run and the data, so tables of several evaluations can be joined."""
        return [dict(run=run_dir.name, dataset=dataset, **row) for row in rows]

    ps.apply_style()
    model = build_model(model_config, device=device)
    tables = {name: [] for name in ('metrics', 'horizon', 'joints', 'speed_bins')}

    def load(epoch):
        model.load_state_dict(torch.load(checkpoints[epoch], map_location=device,
                                         weights_only=True))
        model.eval()

    def evaluate(split, epochs):
        ds = build_dataset(run_cfg, split, data_dir, dataset)
        if axis_order is not None:
            ds = ReorderedAxes(ds, axis_order)
        boot = Bootstrap(ds.episode_index, cfg.bootstrap_samples, cfg.seed)
        horizon = ds[0][1].shape[0]
        speed = history_speed(ds, rate_hz, cfg.batch_size)
        refs = {'zero_velocity': zero_velocity(horizon),
                'constant_velocity': constant_velocity(horizon)}
        up = 'xyz'.index(up_axis)
        ref_s = {m: summarise(*errors(ds, f, device, cfg.batch_size, shoulders, up), boot)
                 for m, f in refs.items()}
        model_err, model_s = {}, {}
        for epoch in epochs:
            load(epoch)
            model_err[epoch], heading = errors(ds, model_forecaster(model), device,
                                               cfg.batch_size, shoulders, up)
            model_s[epoch] = summarise(model_err[epoch], heading, boot)
        log.info(f'{split}: {len(ds)} windows from {boot.n_sequences} sequences')
        for m in REFERENCES:
            s = ref_s[m]['all']
            log.info(f'{split} {m}: MPJPE {s["mpjpe"][0]:.1f} mm, FDE {s["fde"][0]:.1f} mm, '
                     f'heading at 1 s {ref_s[m]["heading"]["final"][0]:.1f} deg')
            tables['metrics'] += tagged(metric_rows(split, '', m, ref_s[m], len(ds),
                                                    boot.n_sequences))
        for epoch in epochs:
            s = model_s[epoch]['all']
            h = model_s[epoch]['heading']['final']
            log.info(f'{split} epoch {epoch}: MPJPE {s["mpjpe"][0]:.1f} mm '
                     f'[{s["mpjpe"][1]:.1f}, {s["mpjpe"][2]:.1f}], FDE {s["fde"][0]:.1f} mm '
                     f'[{s["fde"][1]:.1f}, {s["fde"][2]:.1f}], heading at 1 s {h[0]:.1f} deg '
                     f'[{h[1]:.1f}, {h[2]:.1f}]')
            tables['metrics'] += tagged(metric_rows(split, epoch, 'model', model_s[epoch],
                                                    len(ds), boot.n_sequences))
        return ds, speed, ref_s, model_err, model_s

    def details(split, ds, speed, ref_s, model_err, model_s, epochs, best):
        edges, index = speed_bins(speed, cfg.speed_bins)
        for epoch in epochs:
            for m, s in [('model', model_s[epoch])] + [(m, ref_s[m]) for m in REFERENCES]:
                e = epoch if m == 'model' else ''
                if m != 'model' and epoch != epochs[0]:
                    continue
                tables['horizon'] += tagged(horizon_rows(split, e, m, s, rate_hz))
                tables['joints'] += tagged(joint_rows(split, e, m, s, joint_names))
                tables['speed_bins'] += tagged(speed_rows(split, e, m, s, speed, edges, index))
        summaries = {'model': model_s[best], **ref_s}
        name = f'{split}, checkpoint of epoch {best}'
        plot_horizon(fig_dir / f'{split}_horizon.pdf', summaries, rate_hz,
                     f'Error over the forecast horizon ({name})')
        plot_heading(fig_dir / f'{split}_heading.pdf', summaries, rate_hz,
                     f'Heading error over the forecast horizon ({name})')
        plot_joints(fig_dir / f'{split}_joints.pdf', summaries, joint_names,
                    f'FDE per joint ({name})')
        plot_fde_distribution(fig_dir / f'{split}_fde_distribution.pdf', summaries,
                              f'Distribution of the FDE ({name})')
        plot_speed(fig_dir / f'{split}_speed.pdf', summaries, speed, edges, index,
                   f'FDE per motion speed ({name})')
        load(best)
        plot_examples(fig_dir / f'{split}_examples.pdf', ds, model, model_err[best],
                      cfg.example_joint, joint_names[cfg.example_joint], up_axis, device,
                      f'Example trajectories ({name})')
        for m in REFERENCES:
            for metric in ('mpjpe', 'fde'):
                gain = 1 - model_s[best]['all'][metric][0] / ref_s[m]['all'][metric][0]
                log.info(f'{split} epoch {best}: {metric.upper()} {gain:+.1%} lower than {m}')
            gain = 1 - model_s[best]['heading']['final'][0] / ref_s[m]['heading']['final'][0]
            log.info(f'{split} epoch {best}: heading error at 1 s {gain:+.1%} lower than {m}')

    # val: every checkpoint
    epochs = list(checkpoints)
    ds, speed, ref_s, model_err, model_s = evaluate('val', epochs)
    best = min(epochs, key=lambda e: model_s[e]['all']['mpjpe'][0])
    log.info(f'best checkpoint: epoch {best} (lowest val MPJPE), last: epoch {epochs[-1]}')
    plot_epochs(fig_dir / 'val_epochs.pdf', epochs, model_s, ref_s, train_curve(run_dir), best)
    details('val', ds, speed, ref_s, model_err, model_s, [best], best)
    del ds, model_err

    # test: on request, best-val and last checkpoint
    if cfg.test:
        test_epochs = sorted({best, epochs[-1]})
        ds, speed, ref_s, model_err, model_s = evaluate('test', test_epochs)
        details('test', ds, speed, ref_s, model_err, model_s, test_epochs, best)

    for name, rows in tables.items():
        write_csv(out_dir / f'{name}.csv', rows)
    log.info(f'results in {out_dir}')


if __name__ == '__main__':
    main()
