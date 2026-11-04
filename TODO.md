## Done

- **Packaging:** `pyproject.toml` replaces `setup.py`; `__init__.py` in `interact/`, `interact/model/`, `interact/utils/`, `interact/utils/visualization/`. Dependencies `torch>=2.4`, `numpy>=1.24`, `torch-dct>=0.1.6`, `pyyaml>=6.0`, `requires-python >=3.10`; extras `train` (`hydra-core`, `matplotlib`, `tqdm`, `tensorboard`) and `test` (`pytest`). No torch pin in the package, so the image decides torch and CUDA.
- **Public interface** in `interact/api.py`: `load_config(name)`, `build_model(config, weights_path, device)` (loads the weights strictly with `weights_only=True`), `marginal_forecast`, `human_human_marginal_forecast`, `human_human_conditional_forecast`, `human_robot_conditional_forecast`. Described in the README section "Use as a package".
- **Checkpoint configs** in `interact/config/`: `marginal`, `marginal_2hist`, `interact_hh`, `interact_align_hr` (Cornell checkpoints). Each holds `model_class`, the constructor arguments under `model`, the SMPL-H joint names, parents, `up_axis: y`, 15 Hz rate, 15/15 history/horizon and `num_persons`.
- **Constants** in `interact/constants.py`: plain, immutable values, no imports from `interact`. `tests/test_constants.py` checks both rules.
- **Checked in the deployment versions** (`hpp_docker:jazzy-cuda_x86`: Python 3.12.3, torch 2.11.0+cu130, numpy 1.26.4 and PyYAML 6.0.1 from apt): the install keeps apt's numpy and PyYAML and torch, adds `torch-dct`; all four checkpoints load strictly on CPU and CUDA; the forecast functions match the original scripts' calls exactly.
- **Training container** in `docker/`: `python:3.12-slim`, `torch==2.11.0` (cu130), `numpy==1.26.4`, `torch-dct==0.1.6`, the package installed editable with `[train,test]`. Container user with the host's UID/GID. Mounts under `/home/interact`: repo → `interact`, `data/` → `data`, `outputs/` → `outputs`. Built and tested on the x86 PC.
- **Configs** resolve all paths through `${oc.env:HOME}` (data, mappings, body model, Cornell weights, checkpoints, TensorBoard logs, Hydra run folders). `config/eval.yaml` points to the Cornell weights. All scripts pass `version_base=None` to Hydra.
- **`hr_transfer.py`** passed the weight decay as learning rate (from the upstream Hydra port, 2024-03-03); fixed. The own HR run `2hist_conditional_ft_hr` was trained without weight decay.

## Facts to keep in mind

- Checkpoints are plain `state_dict`s. Unused modules (`position_embeddings`, `pos_table2`, `layer_norm` of `Encoder`/`Decoder`) are among the saved keys; removing any module breaks strict loading.
- Deployment image (`ros:jazzy-ros-base`): Python 3.12.3, numpy 1.26.4 and PyYAML 6.0.1 from apt. The package must keep working with numpy 1.26, or pip installs a second numpy over the one ROS uses.
- Own runs from before the new container are in `interact/checkpoints/`; new runs write to `outputs/checkpoints/`.

## Next

**Step 1: package**

1. Test: one pytest file loads Marginal with `strict=True`, runs a fixed input through it and compares the result with a stored reference output. The weights path comes from an environment variable; without it the test is skipped. **Open.**
2. Repeat the deployment check in the new meta image and tag the fork (e.g. `v0.1.0`). Deferred; the meta repo is out of scope for now.

**Step 2: training pipeline**

1. Training container on the x86 PC. **Done.**
2. Training container on the Jetson Orin (aarch64), with the torch NVIDIA ships for its JetPack.
3. Pipeline cleanup, tested on the Jetson where the datasets are (see Open → Code Related). Decisions still to make: Hydra only in the entry scripts; model configs taken from `interact/config/`; rename the Hydra folder `config/` to tell it apart from `interact/config/`; merge the scripts into one `train.py` and one `evaluate.py`.
4. Logs (TensorBoard, Hydra run folders) into a `log/` folder.

**Step 3: weights**

- Deferred. The Cornell checkpoints stay tracked on GitHub for now; no decision yet on release assets.

## Open

### Hardware Related
- Jetson Orin: which torch NVIDIA ships for it via docker. That sets the torch lower bound in `pyproject.toml`.

### Github Related
- `trained_models_cornell/` is still tracked: an install from a tag downloads ~200 MB until the weights move out of git (step 3).

### Licence Related
- `body_models/smpl_skeleton.npz` is derived from SMPL-H and still tracked. MPI's licence forbids sharing the Software/Data or making it available to third parties; it is in `.gitignore` but needs `git rm --cached` to leave the repo. Old commits keep it.
- SMPL-H and AMASS are licensed for non-commercial research only, including training networks for commercial use. To check for the project as a whole.

### Environment Related
- Eval scripts load a fixed `30.model`; finetuning and HR transfer also expect exactly `30.model` of the previous stage.
- Data loaders read their own config: every dataset class calls `hydra.compose(...)` in its constructor to find its data folder. So they only work inside a running Hydra app, command-line overrides do not reach them, and they can't be used or tested on their own.
- 'log' folder needs to be filled with logs from the training runs.

### Code Related
- Two model descriptions (This is also by and large a problem of how hydra is being utilized right now and also the existence of two 'config' folders which carry different meaning):
    - Training builds models from hh_models/hr_models in training.yaml (with _target_); `eval_checkpoints.yaml` describes the Cornell models a third time.
    - The package builds them from interact/config/*.yaml.
- A trained run produces weights, but no config in the package format, so you have to write interact/config/<name>.yaml for it by hand.
- Centring is copied into every script, separately from the package's version.
- Training runs without validation (commented out) and without a fixed random seed; a checkpoint is saved every epoch.
- Smaller items: pretrain and finetune are about 90% the same file; `hr_transfer` is the same loop plus the alignment loss; `benchmark.py` is `eval_hr.py` plus timing; eval_hr.py hard-codes .to('cuda').
- 'bone_lengths' are missing from the API right now. Interact does not really see different bone lenghts due to the way it is implemented, but still might be usable. Need to decide on that.
