import logging
import pathlib

import hydra
import torch
import torch.optim as optim
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from interact.api import build_model, load_config
from interact.utils.loss_funcs import mpjpe_loss
from script_utils.pretraining import build_dataset, centre, check_model

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../config', config_name='pretraining')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'training on {device}, seed {cfg.seed}')

    run_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir()

    model_config = load_config(cfg.model)
    check_model(model_config)
    # the package config describes its own training data; this run's data is z up
    model_config = {**model_config, 'up_axis': 'z', 'frame': cfg.amass.frame}
    (run_dir / 'model.yaml').write_text(
        f'# run: {run_dir}\n' + yaml.safe_dump(model_config, sort_keys=False)
    )
    model = build_model(model_config, device=device)

    writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))

    dataset = build_dataset(cfg, 'train')
    log.info(f'{len(dataset)} training windows from {cfg.data_dir}')
    train_dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.gamma
    )

    for epoch in range(cfg.epochs):
        total_loss, n = 0, 0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{cfg.epochs}')
        for batch in progress_bar:
            alice_hist, alice_fut, bob_hist, bob_fut = [b.to(device) for b in centre(batch)]
            alice_forecasts = model(alice_hist, bob_hist, bob_fut)
            loss = mpjpe_loss(alice_forecasts, alice_fut)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_dim = alice_hist.shape[0]
            total_loss += loss.item() * batch_dim
            n += batch_dim
            progress_bar.set_postfix(loss=loss.item())
        log.info(f'epoch {epoch + 1}: train mpjpe {total_loss / n:.4f}')
        writer.add_scalar('train/mpjpe', total_loss / n, epoch + 1)

        torch.save(model.state_dict(), checkpoint_dir / f'{epoch + 1}.model')
        scheduler.step()


if __name__ == '__main__':
    main()
