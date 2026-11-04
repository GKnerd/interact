"""
Build InterACT models from the package configs and run their forecasts.

All forecast functions take and return joint positions of shape (B, T, J, 3) in the frame of
the training data. 

B := Batch: How many independent windows are forecast in one call.
T := Frames over time.
J := Joints in the config's 'joint_names' order
3 := x,y,z position of the joints

Centring and the offset back are done here, so training and deployment use
the same steps.
"""

from importlib.resources import files

import torch
import yaml

from interact.constants import MODEL_CLASS_NAMES, ROBOT_CENTRE
from interact.model import Models
from interact.model.Models import IntentInformedForecaster, IntentInformedHRForecaster


def load_config(name: str) -> dict:
    """
    Read the config of one checkpoint.

    :param name: File name in ``interact/config`` without extension, e.g. ``'marginal'``.
    :return: The config.
    """
    return yaml.safe_load((files('interact') / 'config' / f'{name}.yaml').read_text())


def build_model(
    config: dict, weights_path: str | None = None, device: str | torch.device = 'cpu'
) -> torch.nn.Module:
    """
    Build the model of a config and load its weights.

    :param config: Config from :func:`load_config`.
    :param weights_path: Checkpoint (``state_dict``); ``None`` leaves the weights untrained.
    :param device: Device the model is moved to.
    :return: The model in eval mode.
    """
    name = config['model_class']
    if name not in MODEL_CLASS_NAMES:
        raise ValueError(f'unknown model_class {name!r}')
    model = getattr(Models, name)(**config['model'], device=device).to(device)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def marginal_forecast(model: IntentInformedForecaster, history: torch.Tensor) -> torch.Tensor:
    """
    Forecast one person from their own history.

    :param model: Human-human model with ``one_hist`` and without ``conditional_forecaster``.
    :param history: (B, T, J, 3) history of the person.
    :return: (B, H, J, 3) forecast of the person.
    """
    _check_human_human(model, one_hist=True, conditional=False)
    alice, offset = _centre(history)
    empty = torch.zeros_like(alice)
    return _restore(model(alice, empty, empty), offset)


@torch.no_grad()
def human_human_marginal_forecast(
    model: IntentInformedForecaster, history: torch.Tensor, partner_history: torch.Tensor
) -> torch.Tensor:
    """
    Forecast one person from their own history and a second person's history.

    :param model: Human-human model without ``one_hist`` and ``conditional_forecaster``.
    :param history: (B, T, J, 3) history of the person.
    :param partner_history: (B, T, J, 3) history of the second person.
    :return: (B, H, J, 3) forecast of the person.
    """
    _check_human_human(model, one_hist=False, conditional=False)
    alice, offset = _centre(history)
    bob_hist = _flatten(partner_history, offset)
    return _restore(model(alice, bob_hist, torch.zeros_like(bob_hist)), offset)


@torch.no_grad()
def human_human_conditional_forecast(
    model: IntentInformedForecaster,
    history: torch.Tensor,
    partner_history: torch.Tensor,
    partner_future: torch.Tensor,
) -> torch.Tensor:
    """
    Forecast one person, conditioned on a second person's future.

    :param model: Human-human model with ``conditional_forecaster`` and without ``one_hist``.
    :param history: (B, T, J, 3) history of the person.
    :param partner_history: (B, T, J, 3) history of the second person.
    :param partner_future: (B, F, J, 3) future of the second person; only the last frame is used.
    :return: (B, H, J, 3) forecast of the person.
    """
    _check_human_human(model, one_hist=False, conditional=True)
    alice, offset = _centre(history)
    bob_hist = _flatten(partner_history, offset)
    bob_fut = _flatten(partner_future, offset)
    return _restore(model(alice, bob_hist, bob_fut), offset)


@torch.no_grad()
def human_robot_conditional_forecast(
    model: IntentInformedHRForecaster,
    history: torch.Tensor,
    robot_history: torch.Tensor,
    robot_future: torch.Tensor,
) -> torch.Tensor:
    """
    Forecast one person, conditioned on the robot's planned future.

    The robot points are centred on the person's last two joints (L_Hand, R_Hand), as in
    training.

    :param model: Human-robot model.
    :param history: (B, T, J, 3) history of the person.
    :param robot_history: (B, T, 2, 3) history of the two end-effector points.
    :param robot_future: (B, F, 2, 3) planned future of the end-effector points; only the last
        frame is used.
    :return: (B, H, J, 3) forecast of the person.
    """
    if not isinstance(model, IntentInformedHRForecaster):
        raise ValueError(f'expected IntentInformedHRForecaster, got {type(model).__name__}')
    alice, offset = _centre(history)
    robot_offset = offset[:, :, ROBOT_CENTRE]
    robot_hist = _flatten(robot_history, robot_offset)
    robot_fut = _flatten(robot_future, robot_offset)
    forecast, _ = model.forward_inference(alice, None, None, robot_hist, robot_fut)
    return _restore(forecast, offset)


def _check_human_human(model: torch.nn.Module, one_hist: bool, conditional: bool) -> None:
    if not (
        isinstance(model, IntentInformedForecaster)
        and model.one_hist == one_hist
        and model.conditional_forecaster == conditional
    ):
        raise ValueError(
            f'expected IntentInformedForecaster with one_hist={one_hist} and '
            f'conditional_forecaster={conditional}'
        )


def _centre(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten the joints and subtract each joint's last position.

    :return: Centred history (B, T, J*3) and offset (B, 1, J*3).
    """
    flat = history.reshape(history.shape[0], history.shape[1], -1)
    offset = flat[:, -1:]
    return flat - offset, offset


def _flatten(poses: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return poses.reshape(poses.shape[0], poses.shape[1], -1) - offset


def _restore(forecast: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return (forecast + offset).reshape(forecast.shape[0], forecast.shape[1], -1, 3)
