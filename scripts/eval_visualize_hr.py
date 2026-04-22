import torch 
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Import the model and dataloader
from interact.model.Models import IntentInformedHRForecaster
from interact.utils.comad_hr import CoMaD_HR

# Import the Facebook Renderer
from interact.utils.visualization.generic_moded import AnimationRenderer
from interact.utils.skeleton import SkeletonCoMaDHR


# ==========================================
# 2. THE GENERATOR WRAPPER
# ==========================================
def test_poses_generator(test_dataloader, model, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            
            # Offset normalization (exactly like their code)
            offset = batch[0].reshape(batch[0].shape[0], batch[0].shape[1], -1)[:, -1].unsqueeze(1)
            
            alice_hist, alice_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset).to(device) for i in range(2)]
            bob_hist, bob_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, -6:]).to(device) for i in range(2,4)]
            robot_hist, robot_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, -6:]).to(device) for i in range(4,6)]
            
            # Forward pass
            alice_forecasts, _ = model(alice_hist, bob_hist, bob_fut, robot_hist, robot_fut)
            
            # Reshape [Batch, Sequence_Length, Flat_Dims] -> [Batch, Sequence_Length, Joints, 3]
            alice_hist_3d = alice_hist.view(alice_hist.shape[0], alice_hist.shape[1], 9, 3)
            alice_fut_3d = alice_fut.view(alice_fut.shape[0], alice_fut.shape[1], 9, 3)
            alice_pred_3d = alice_forecasts.view(alice_forecasts.shape[0], alice_forecasts.shape[1], 9, 3)
            
            robot_hist_3d = robot_hist.view(robot_hist.shape[0], robot_hist.shape[1], 2, 3)
            robot_fut_3d = robot_fut.view(robot_fut.shape[0], robot_fut.shape[1], 2, 3)

            # Stitch History + Future (15 + 15 = 30 frames)
            alice_gt_time = torch.cat([alice_hist_3d, alice_fut_3d], dim=1)
            alice_pred_time = torch.cat([alice_hist_3d, alice_pred_3d], dim=1)
            robot_time = torch.cat([robot_hist_3d, robot_fut_3d], dim=1)

            # Combine Alice (9) + Robot (2) = 11 Joints
            scene_gt = torch.cat([alice_gt_time, robot_time], dim=2)
            scene_pred = torch.cat([alice_pred_time, robot_time], dim=2)

            # Add Dummy Participant Dimension: [Batch, Time, 1, 11, 3]
            scene_gt = scene_gt.unsqueeze(2)
            scene_pred = scene_pred.unsqueeze(2)

            # Return the first sequence in the batch for the renderer
            poses_dict = {
                'gt': scene_gt[0].cpu().numpy(),
                'Prediction': scene_pred[0].cpu().numpy()
            }
            
            yield poses_dict, batch_idx

# ==========================================
# 3. THE MAIN SCRIPT
# ==========================================
@hydra.main(config_path="../config", config_name="training")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Dataset
    print("Loading CoMaD_HR Test Set...")
    test_dataset = CoMaD_HR(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 2. Load Model
    print("Loading Model...")
    selected_model = cfg.selected_hr_model
    model_config = cfg.hr_models[selected_model] 
    
    # Reconstruct the model ID exactly like the training script does
    model_id = f'{"1hist" if model_config.one_hist else "2hist"}_{"marginal" if not model_config.conditional_forecaster else "conditional"}'
    model_id += '_ft_hr'
    model_id += '_noalign' if not model_config.align_rep else ''
    
    model = hydra.utils.instantiate(model_config, device=device).to(device)
    
    weights_path = f'{cfg.Training.hr_training.output_dir}/{model_id}/{cfg.Training.epochs}.model'
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: Could not find weights at {weights_path}. Check your path!")
        return

    # 3. Setup Visualization
    skeleton = SkeletonCoMaDHR()
    generator = test_poses_generator(test_dataloader, model, device)

    renderer = AnimationRenderer(
        skeleton=skeleton, 
        poses_generator=generator, 
        algos=['Prediction'], 
        t_hist=15, 
        t_pred=15, 
        baselines=['gt'], 
        output_dir=f"{cfg.Training.hr_training.output_dir}/visualizations",
        type="3d"
    )

    # 4. Render
    print("Generating GIFs... Check the ./visualizations folder.")
    # idx=0 means it will render the 'Prediction' alongside 'Ground_Truth'
    renderer.store_all(type="gif", idx=0) 

if __name__ == "__main__":
    main()
