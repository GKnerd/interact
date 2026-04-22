import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import csv
import os
from interact.model.Models import IntentInformedHRForecaster
from torch.utils.data import DataLoader, ConcatDataset
from interact.utils.comad_hr import CoMaD_HR
from interact.utils.loss_funcs import mpjpe_loss, fde_error, perjoint_error, perjoint_fde

@hydra.main(config_path="../config", config_name="eval")
def main(cfg: DictConfig):
    dataset_map = {
        'cabinet': lambda: CoMaD_HR(split='test'),
        'take': lambda: CoMaD_HR(split='test'),
        'cart': lambda: CoMaD_HR(split='test'),
    }

    models = cfg.hr_eval.models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device.upper()}")
    print(f"Evaluating Models: {models}")

    # Initialize Dataset
    Dataset = ConcatDataset([dataset_map[cfg.hr_eval.eval_data]()])
    loader_test = DataLoader(
        Dataset,
        batch_size=cfg.hr_eval.batch_size,
        shuffle=False,
        num_workers=0
    )

    model_info_cfg = hydra.compose(config_name="eval_checkpoints", overrides=[])
    
    # Store results for the final CSV
    benchmark_results = []

    for model_path in models:
        print(f"\n{'='*40}")
        print(f"Benchmarking Model: {model_path}")
        print(f"{'='*40}")

        model_config = model_info_cfg.hr_models[model_path]
        model = hydra.utils.instantiate(model_config, device=device).to(device)
        
        weights_path = f'{cfg.hr_eval.checkpoint_dir}/{model_path}/30.model'
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print(f"WARNING: Weights not found at {weights_path}. Skipping...")
            continue
            
        model.eval()

        # Tracking variables
        running_per_joint_errors = []
        running_per_joint_fdes = []
        inference_times = []
        n = 0

        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for cnt, batch in enumerate(loader_test): 
                # 1. Prepare Data
                offset = batch[0].reshape(batch[0].shape[0], batch[0].shape[1], -1)[:, -1].unsqueeze(1)
                alice_hist, alice_fut, bob_hist, bob_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, :batch[i].shape[2]*3]).to(device) for i in range(4)]
                robot_hist, robot_fut = [(batch[i].reshape(batch[i].shape[0], batch[i].shape[1], -1) - offset[:, :, -6:]).to(device) for i in range(4,6)]
                
                batch_dim = alice_hist.shape[0]
                n += batch_dim

                # 2. Measure Inference Latency
                if device == "cuda":
                    start_event.record()
                    alice_forecasts, alignment_loss = model.forward_inference(alice_hist, bob_hist, bob_fut, robot_hist, robot_fut)
                    end_event.record()
                    torch.cuda.synchronize()
                    # Only record timing after the first few warmup batches
                    if cnt > 5: 
                        inference_times.append(start_event.elapsed_time(end_event))
                else:
                    # CPU fallback (less precise but functional)
                    import time
                    t0 = time.time()
                    alice_forecasts, alignment_loss = model.forward_inference(alice_hist, bob_hist, bob_fut, robot_hist, robot_fut)
                    if cnt > 5:
                        inference_times.append((time.time() - t0) * 1000) # Convert to ms

                # 3. Compute Accuracy Metrics
                per_joint_error, per_joint_error_list = perjoint_error(alice_forecasts, alice_fut)
                per_joint_fde, per_joint_fde_list = perjoint_fde(alice_forecasts, alice_fut)

                running_per_joint_errors += list(per_joint_error_list.cpu().numpy())
                running_per_joint_fdes += list(per_joint_fde_list.cpu().numpy())
        
        # Calculate Accuracy Stats (converting to millimeters)
        all_joints_ade_mean = np.array(running_per_joint_errors).mean(axis=1).mean() * 1000
        all_joints_fde_mean = np.array(running_per_joint_fdes).mean(axis=1).mean() * 1000
        
        # NOTE: Indices 5:9 target the wrists/hands assuming the 9-joint baseline. 
        # If you are using 15 joints, you will need to update these indices!
        wrist_ade_mean = np.array(running_per_joint_errors)[:, 5:9].mean(axis=1).mean() * 1000
        wrist_fde_mean = np.array(running_per_joint_fdes)[:, 5:9].mean(axis=1).mean() * 1000

        # Calculate Speed Stats
        avg_latency_ms = np.mean(inference_times)
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

        # Print Terminal Report
        print(f"--- Accuracy (Lower is better) ---")
        print(f"Overall ADE:   {all_joints_ade_mean:.2f} mm")
        print(f"Overall FDE:   {all_joints_fde_mean:.2f} mm")
        print(f"Wrist ADE:     {wrist_ade_mean:.2f} mm")
        print(f"Wrist FDE:     {wrist_fde_mean:.2f} mm")
        print(f"--- Computational Speed ---")
        print(f"Avg Latency:   {avg_latency_ms:.2f} ms per batch")
        print(f"Throughput:    {fps:.2f} FPS")

        # Save to dict for CSV export
        benchmark_results.append({
            "Model": model_path,
            "Dataset": cfg.hr_eval.eval_data,
            "Overall_ADE_mm": round(all_joints_ade_mean, 2),
            "Overall_FDE_mm": round(all_joints_fde_mean, 2),
            "Wrist_ADE_mm": round(wrist_ade_mean, 2),
            "Wrist_FDE_mm": round(wrist_fde_mean, 2),
            "Latency_ms": round(avg_latency_ms, 2),
            "FPS": round(fps, 2)
        })

    # Export to CSV
    csv_file = f"{cfg.hr_eval.checkpoint_dir}/benchmark_report.csv"
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=benchmark_results[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_results)
    
    print(f"\n{'-'*40}")
    print(f"Benchmark complete! Report saved to: {csv_file}")
    print(f"{'-'*40}")

if __name__ == "__main__":
    main()
