# Additional Improvements

Last Updated: 02-11-2027

### What is this about? 

This section adds improvements to the InterACT model that are worth exploring after the initial migration and code improvements.

## Findings 1

#### 1. What the network sees

Every input is centred per joint: `offset = alice_hist[:, -1]` holds all 27 values, and each joint has its own last position subtracted. This holds for the local encoder, the global encoder and the distance weighting (SPE). The forecast is a sum of displacements (`cumsum`) with the offset added back.

- The network sees how each joint moved over the last second relative to where it is now. It never sees where the joints are relative to each other, so it sees neither the pose nor the bone lengths.
- Bone lengths act only indirectly: the same rotation moves the wrist further on a longer forearm.
- The network cannot tell which way the person faces. It only sees the direction of each movement in the axes of the training frame.
- What does matter: which body point each joint is, consistently over time and across persons. A bone that stretches through tracker noise shows up as joint motion. A fixed SMPL-H skeleton per person avoids this.
- Retargeting to training bone lengths (STATUS, generic node) has little effect for InterACT.

#### 2. Point definitions differ between the training datasets

The 9 slots mean different body points per dataset:

| Slot | Synthetic AMASS | CMU-Mocap | CoMaD |
|---|---|---|---|
| 0 | `Neck` (SMPL joint centre) | `lowerneck` (ASF joint) | `BackTop` (marker on the upper back) |
| 1–2 | `L/R_Shoulder` | `l/rclavicle` | `L/RShoulderBack` (marker) |
| 3–4 | `L/R_Elbow` | `l/rradius` | `L/RElbowOut` (marker) |
| 5–8 | joint centres | ASF joints | surface markers |

- Synthetic AMASS uses one fixed skeleton (`p3d0` from `body_models/smpl_skeleton.npz`); the AMASS `betas` are never read.
- CoMaD consists of OptiTrack surface markers on 8 persons. The deployed Cornell Marginal was fine-tuned on it.
- A constant offset between marker and joint cancels through the per-joint centring. A marker rigidly on a bone at distance d from the joint differs from the joint's path by at most `2·d·sin(θ/2)` over the window (d = 5 cm, θ = 30°: about 2.6 cm).
- With retraining on own SMPL-H data, training and deployment use the same points and this mismatch disappears.

#### 3. Up axis of the training data

All training data, as the network sees it, is y-up. The deployment world (ROS) is z-up.

- Synthetic AMASS: the pelvis orientation is set to identity (`poses[:, 0] = 0`), so the joints stay in the SMPL body frame (y up, facing +z, +x = body's left). The pairing code places and rotates Bob in the x–z plane. Read from the code.
- CoMaD H-H: the loader changes no axes. Cornell's RViz playback script (`run_comad_episode.py`, removed in b669951) shows the data in ROS's z-up frame with `(-x, z, y)`, the conversion from OptiTrack's y-up. Inferred, not confirmed on data.
- CoMaD H-R: `comad_hr.py` applies `fix_orientation` (`(x, y, z) → (-x, z, y)`); the HR playback script applies the same mapping again for RViz, which undoes it. This fits z-up raw data converted to y-up for the model.
- The current deployment (`interact_inference_node.py`, `skeleton_mapper_node.py`) feeds z-up data without conversion.
- Check for CoMaD: in one H-H `data.json`, the `BackTop` marker should sit at about 1.3–1.5 m on the y axis.

#### 4. AMASS preprocessing in `scripts/create_synthetic_amass_data.py`

AMASS stores per frame the pelvis orientation in the AMASS world (z-up; 3 degrees of freedom: heading and tilt), the joint rotations, and the pelvis position `trans`. Lines 93–102:

```python
R = rodrigues(global_orient)                # pelvis orientation, per frame
trans_rot = trans @ R                        # = Rᵀ·trans, per frame
poses[:, 0] = 0                              # pelvis orientation set to identity
p3d = ang2joint(p3d0, poses, parent) + trans_rot
```

| Effect | Status |
|---|---|
| a) Heading removed: Alice's pelvis faces SMPL's +z in every frame of every sequence. Turning in place disappears from the data | Read from the code |
| b) Tilt removed: the pelvis is forced upright. Bending forward, recorded largely as pelvis tilt, appears as a more upright upper body with the bend moved into the legs. Reaching down is distorted | Read from the code |
| c) Fake translation when turning: `trans` is the absolute pelvis position and is rotated by the current orientation every frame. A person 3 m from the AMASS origin turning 90° in place appears to slide about 4.2 m | Derived from the formula; to confirm by plotting the pelvis path of one AMASS sequence with a turn in place |

History: the first version (a21b201) zeroed the orientation without translation, as in the older code it was copied from. bd79260 ("Add translations to AMASS data") added the `Rᵀ·trans` line.

The orientation is only an input to forward kinematics. Keeping it means applying the real rigid transform per frame (`p_world = R · p_body + t`, R about the pelvis joint) instead of identity and `Rᵀ·t`. The stored data stays Cartesian joint positions, now in the AMASS world frame (z-up).

#### 5. Heading

With up axis `u` (`e_z` in a z-up frame), the pelvis orientation splits into `R = R_u(ψ) · R_tilt`: a rotation about `u` by the heading ψ, and a tilt with no rotation about `u`. The split is unique as long as the body's forward axis is not vertical.

From the 9 joints (z-up, right-handed):

1. `l = p_L_Shoulder − p_R_Shoulder`
2. `l_h = l − (l·u) u` (projection onto the floor)
3. `f = l_h × u` (forward on the floor)
4. `ψ = atan2(f_y, f_x)`

ψ is undefined when the shoulder line is nearly vertical (`l_h ≈ 0`).

The network works on raw x/y/z through linear layers, so it is not rotation-invariant by construction. A forward reach that was almost always "+z" in training looks like a sideways movement when it points along another axis. Two consistent ways to handle this:

- A: random rotation about `u` per training window. The model learns that a movement continues in its own direction regardless of where it points. No heading estimate at runtime. Invariance is approximate and costs model capacity.
- B: rotate each window so the person faces a fixed direction, in training and in the adapter. Exact by construction, but needs a reliable ψ every window; an error in ψ rotates the prediction.

The Cornell Marginal saw mostly one heading in pretraining (effect a) and a few from CoMaD.

#### 6. CoMaD for Marginal

Marginal (`one_hist: true`) uses only one person. CoMaD contributes Cornell's lab tasks and the marker-instead-of-joint mismatch. It is the only paired human–robot data, needed only for the HR models; for conditional models it provides two-person data if none is recorded.

#### 7. Human-robot "marginal" checkpoints see the robot's future

In `IntentInformedHRForecaster`, `conditional_forecaster: false` only zeroes Bob's future, which feeds only the alignment loss. The decoder is always queried with the robot's future, in `forward` and in `forward_inference`. This holds from the first HR model (930eb58, Sept 2023) to today, and `hr_transfer.py` and `eval_hr.py` pass the real robot future to every HR model.

- The HR checkpoints `Marginal` and `OnlyFineTuned` are therefore conditioned on the robot's planned future. "Marginal" there only means "no robot history", contrary to the paper's definition (independent of the robot).
- What the code does is certain; whether the released checkpoints were trained with exactly this code is not provable.
- The HH models are not affected: there the partner's future is zeroed before the decoder.
- The package therefore has no human-robot marginal forecast function.

### Potential Improvements

- Training frame z-up everywhere (AMASS world and ROS world). No axis change in the adapter.
- Regenerate synthetic AMASS with the real pelvis orientation and the real `trans`, stored in the AMASS world frame. Removes effects a)–c). The pairing code then places and rotates Bob in the x–y plane instead of x–z.
- Heading: A (random rotation about the vertical axis per training window, in pretraining and fine-tuning), as one function in the package. Evaluate the error per heading bin; it should be flat. Move to B only if it is not. Tilt is never removed.
- Retrain Marginal on synthetic AMASS (pretraining) and own SMPL-H data (fine-tuning), without CoMaD. Compare against the Cornell Marginal on the own bags.
- Until then, for the Cornell Marginal: rotate z-up to y-up in the adapter with a proper rotation (e.g. `(x, y, z) → (x, z, -y)` or `(-x, z, y)`; never a plain swap of two axes, which mirrors left and right) and the inverse on the output. Heading normalisation (B) in the adapter is worth a test only if the interim model is used for evaluation.
- Bone lengths in the adapter contract: retargeting to training bone lengths has little effect for InterACT (finding 1).


## Potential Redesign of the training pipeline

### Initially Proposed Design

#### How Hydra is used

| Problem | Consequence |
|---|---|
| Dataset classes call `hydra.compose("datasets")` in their constructors | They only work inside a running Hydra app. Command-line overrides don't reach them: `data_dir=/x` is impossible because `datasets.yaml` is composed separately from the main config. |
| Eval scripts also `compose("eval_checkpoints")`; `eval_checkpoints.yaml` repeats the `Training` block and describes the Cornell models a second time | Three places describe models: `training.yaml`, `eval_checkpoints.yaml`, `interact/config/` |
| Models are built with `_target_` plus `hydra.utils.instantiate` | Library code depends on Hydra conventions |
| Two folders called `config` with different meanings | `config/` holds run settings (Hydra); `interact/config/` holds the checkpoint description (package) |

#### How models are produced

| Problem | Consequence |
|---|---|
| Stages are chained through a folder name built from two flags (`2hist_conditional_ft`) and a hard-coded `30.model` | Finetuning finds pretraining only if pretraining ran exactly 30 epochs with matching flags; a different `epochs` breaks the chain |
| A checkpoint is saved every epoch (30 × 22 MB), with no config next to it | You can't tell which settings produced a file, and deploying a run means writing its package config by hand |
| Validation is commented out in all three training scripts | Training runs blind on training loss; whether a run is good shows only in a separate eval |
| No random seed is set | Runs aren't reproducible |
| Centring is copied into every script | Already caused inconsistent Bob centring between `eval_hr` and `hr_transfer` |

#### Data loaders

| Problem | Consequence |
|---|---|
| Four classes repeat the same windowing loop and the Alice/Bob flip | Each fix has to be made four times |
| Joint selection is hard-coded in each class (`comad_hr.py` even uses bare indices) | Your own SMPL-H dataset would be a fifth copy |
| Bare `except: continue` when reading files | Broken or missing files are skipped silently |
| JSON is parsed fresh on every run, and every window is kept as a separate object (about two million for AMASS) | Slow start-up; the JSON parsing is the main cost |

#### Scripts

- Pretrain and finetune are about 90% the same file.
- `hr_transfer` is the same loop plus the alignment loss and loading weights into `model.hh`.
- `benchmark.py` is `eval_hr.py` plus timing.

#### Proposed target design (for discussion, nothing written)

1. **Hydra only in the entry scripts.** Everything under `interact/` takes plain arguments: datasets get their folder and joint list passed in, and models come from `build_model`. Hydra stays where it's useful: composing configs, command-line overrides, run folders.
2. **One model description:**
   - Training selects the model from `interact/config/<name>.yaml` as a Hydra config group. Hydra can search an installed package (`pkg://interact.config`).
   - A new model means a new YAML there. Train with `model=<name>`, deploy with `checkpoint=<name>`; nothing is copied or retyped.
   - `hh_models`/`hr_models` and `eval_checkpoints.yaml` go away.
3. **One `train.py` with stages.** A stage config (`pretrain`, `finetune`, `hr_transfer`) sets:
   - the datasets
   - learning rate and scheduler
   - the alignment loss weight
   - which run to initialise from: the whole model, or only `.hh` for HR

   Each run gets one folder containing:
   - the config used
   - TensorBoard logs (or under `log/`, per your open point)
   - the last checkpoint plus every N-th
   - validation each epoch

   Stages chain by run folder, not by flag-derived names and `30.model`.
4. **One `evaluate.py`** for H-H and H-R, with optional timing; this replaces `eval_hh`, `eval_hr` and `benchmark`. It builds models through `load_config` and `build_model` and uses the package's forecast functions, so evaluation runs exactly the deployed code path. `eval_visualize_hr.py` stays separate.
5. **One windowed dataset over sequences.** A small reader per source (AMASS, CMU, CoMaD H-H, CoMaD H-R, later your own SMPL-H data) returns whole sequences (T, J, 3) at 15 Hz. One class cuts the windows and optionally flips Alice and Bob.
   - Room for ADDITIONAL_IMPROVEMENTS: the random rotation about the vertical axis (heading option A) becomes one transform in that class, and your own dataset becomes one more reader.
   - Optional: cache the parsed JSON once as `.npz` to cut start-up time.

## Findings 2

Scan of the training code and of the data on the Jetson Orin (`~/datasets`), 12-11-2026. Nothing above was changed. Statements above that these findings contradict are listed under "Contradictions with the sections above".

#### 8. Frame rate of the synthetic AMASS data

`scripts/create_synthetic_amass_data.py:84` takes every `int(frame_rate // 15)`-th frame. This gives 15 Hz only when the source rate is a multiple of 15.

| Source rate | Step | Resulting rate | Train datasets |
|---|---|---|---|
| 120 Hz | 8 | 15 Hz | CMU (most), Eyes_Japan_Dataset, ACCAD, MPI_Limits |
| 60 Hz | 4 | 15 Hz | CMU (some) |
| 100 Hz | 6 | 16.7 Hz | KIT, EKUT |
| 250 Hz | 16 | 15.6 Hz | Eyes_Japan_Dataset (one file in the sample) |

- Train files per dataset: KIT 4232, CMU 2088, Eyes_Japan_Dataset 750, EKUT 349, ACCAD 246, MPI_Limits 35. KIT and EKUT are 4581 of 7700 files (59%).
- The loader reads all files as 15 Hz. A 16.7 Hz sequence read as 15 Hz moves 10% less per frame, and its 15-frame horizon covers 0.9 s instead of 1.0 s.
- The line is from the upstream commit 05ec26e, so Cornell's pretraining data had the same mix if it was created with this script and these datasets.
- The CoMaD and CMU-Mocap loaders are not affected (120 Hz, step 8).
- Status: the step is read from the code; the source rates are measured on about 26 files per dataset (`mocap_framerate`), not on all files.

#### 9. Validation

Validation cannot be switched on by uncommenting the existing calls.

- CoMaD has no `val` folder, only `train` and `test`. `CoMaD(split='val')` and `CoMaD_HR(split='val')` fail when they list the folder.
- CoMaD episodes: train 385 H-H and 249 H-R, test 93 H-H and 55 H-R.
- Synthetic AMASS val is 44 files (SFU), 13,725 frames, 12,405 windows. The test split (3061 files, BioMotionLab_NTroje) is read by no script.
- `log_metrics` in `scripts/hr_transfer.py:53`:
  - reads `cfg.align_rep` and `cfg.align_weight`, which do not exist (the values are `model_config.align_rep` and `cfg.Training.align_weight`);
  - centres Bob's 6 values with the 27-value offset, a third variant next to `hr_transfer` (training) and `eval_hr`.
- The commented calls in all three scripts pass four arguments; the functions take six or seven.
- Windows of one episode overlap in 29 of 30 frames, so a validation split has to be made by episode, not by window.

#### 10. Data loading cost

- The synthetic AMASS train split is 7700 JSON files, 21 GB. The files hold all 52 joints of both persons as indented text; the loaders use 9 joints.
- Measured on the Orin, val split (196 MB), read the way the loader does: `json.load` 3.3 s, `torch.tensor` 1.3 s, 4.8 s in total. Extrapolated by size to the train split: about 9 minutes at every start.
- Extrapolated the same way, the train split has about 1.5 million frames and about 1.3 million windows. Each window is four tensor objects, and pretraining copies these lists into 8 DataLoader workers.
- The 9 joints of both persons as float32 are about 330 MB for the whole train split. This fits into memory as one array; on the Orin, CPU and GPU share that memory.
- CoMaD is 43 GB on disk, but almost all of it is `video.MP4`; a `data.json` is about 3 MB.
- The file names of the synthetic data contain the absolute path of the machine that created them (`_home_torch_orin_docker_interact_data_amass_KIT_...`). Source dataset and subject can only be recovered from that name.

#### 11. Logging

- The Hydra log files of the three earlier pretraining runs (`outputs/2026-04-2*/`) are 0 bytes: the scripts use `print`, which Hydra does not capture.
- Finetuning writes no TensorBoard scalar; `scripts/finetune_intent_forecaster.py:134` is commented out.
- `scripts/hr_transfer.py:151` logs the alignment loss of the last batch divided by `n`, not the epoch mean (`total_alignment_loss` is summed but not used).

#### 12. Learning-rate schedule and epochs

- `config/training.yaml:15`: milestones `[15, 25, 35, 40]` with `epochs: 30`. The last two never fire.
- Pretraining and finetuning share the milestones and the one `epochs` value, although AMASS and CoMaD differ in size by orders of magnitude.
- `hr_transfer.py` creates no scheduler.

#### 13. Bob's joints: training configs and package configs differ

| Where | `bob_joints_list` | `bob_joints_num` |
|---|---|---|
| `config/training.yaml`, `hh_models` | `[7, 8]` | 2 |
| `config/training.yaml`, `hr_models` | `[0, 1]` | 2 |
| `interact/config/*.yaml` (Cornell checkpoints) | `[0, ..., 8]` | 9 |

- A model trained with the current defaults has a 6-value Bob encoder and fits none of the package configs.
- In the H-R data Bob has two points (finding 14). `forward` of the HR model indexes Bob with the list in every case, so a 9-joint Bob cannot run through `hr_transfer.py` on this data. How the Cornell HR checkpoints with 9 Bob joints were trained cannot be reproduced with this code and data.
- The same `hh` submodule needs `[7, 8]` in the H-H stages (hands out of 9 joints) and `[0, 1]` in the H-R stage (the two points that exist). The list describes the layout of the data, not the weights.

#### 14. CoMaD on disk

- Persons: the `metadata.json` files name 14 person IDs in H-H (1–14) and 9 in H-R (1–7, 10, 11).
- H-R layout (one episode checked, `train/cabinet/HR/1`): `Person_1` (T, 11, 3), `Person_2` (T, 2, 3), `Robot` (T, 12, 3). `comad_hr.py` assigns them by key order and picks hard-coded indices (`[0..6, 9, 10]`, `[0, 1]`, `[10, 11]`). No mapping file names the 11 points, so which body points the H-R models see cannot be read from the repo. The H-R `metadata.json` lists `Person_1` only.
- Up axis, H-H: the `BackTop` marker, missing frames excluded, has a mean y of 1.34–1.62 m in 5 episodes (10 tracks), while x and z vary with the position in the room. CoMaD H-H is y-up.
- Up axis, H-R (same episode, raw): point 0 of `Person_1` has a mean z of 1.43 m and a mean y of 2.39 m. This fits z-up raw data.
- Missing markers: in `train/cabinet/HH/0`, 17% (`Person_1`) and 13% (`Person_2`) of all marker samples are zero. The loader drops every window in which one of the 9 used markers is missing in any frame.

#### 15. Evaluation

- `eval_data` has no effect for CoMaD: `handover`, `react_stir` and `table_set` (and `cabinet`, `take`, `cart` for H-R) all load the whole test split. The `subtask` argument of `CoMaD_HR` is unused.
- Only the FDE is printed; nothing is written to a file.
- The print loop refers to a model `cvm`, but no baseline exists. Without a zero-velocity or constant-velocity number the errors have no reference.

#### 16. Synthetic pairing

- `random` and `np.random` are used without a seed, so the dataset itself is not reproducible.
- Bob is another sequence, cut from its first frame to Alice's length. If Alice is the longest sequence, Bob is Alice herself.
- The distance check subtracts joints of the same index (`cand_bob_pose - alice_pose`). It is not a distance between the two bodies.
- A model with `one_hist: true` (Marginal) never uses Bob, so its pretraining does not need the pairing at all.

#### 17. Smaller code items

- `total_loss += loss * batch_dim` in all three training loops adds tensors that still carry their graph; the graph nodes of a whole epoch stay alive. `loss.item()` or `loss.detach()` avoids this.
- Debug code is left in the pretraining loop (`scripts/pretrain_intent_forecaster.py:112-123`).
- Checkpoints hold no optimizer or scheduler state, so a stopped run cannot be resumed.
- The four dataset modules in `interact/utils/` import hydra and the two visualisation modules import matplotlib. `import interact` does not load them, but no test checks this.

#### 18. Container on the Jetson

- `docker/run_container.sh` mounts `<repo>/data`. On the Orin that folder is empty and owned by root; the datasets are in `~/datasets`.
- The container is started interactively only (`-it --rm`), so a training run ends with the terminal.

### Contradictions with the sections above

| Statement above | What the scan shows |
|---|---|
| Header: "Last Updated: 02-11-2027" | The date lies in the future; left unchanged. |
| Finding 2: "CoMaD consists of OptiTrack surface markers on 8 persons" | The metadata names 14 person IDs in H-H and 9 in H-R (finding 14). |
| Finding 2, table: one CoMaD column with marker names | Holds for H-H. H-R has 11 unnamed points per person and a different order (finding 14). |
| Finding 3, CoMaD H-H: "Inferred, not confirmed on data", check value 1.3–1.5 m | Confirmed on data: y-up, `BackTop` at 1.34–1.62 m (finding 14). |
| Finding 4: "Lines 93–102" | The quoted code is at lines 94–102. |
| Redesign, "How models are produced": "only if pretraining ran exactly 30 epochs", "a different `epochs` breaks the chain" | Fewer than 30 epochs break the chain. More than 30 do not: the next stage silently loads epoch 30 instead of the last one. |
| Redesign, "Data loaders": "Four classes repeat the same windowing loop and the Alice/Bob flip" | All four repeat the windowing loop. Only `CMU_Mocap` and `CoMaD` flip; `Synthetic_AMASS` and `CoMaD_HR` do not. |
| Redesign, "Data loaders": "about two million for AMASS" | About 1.3 million windows, each kept as four tensor objects (extrapolated, finding 10). |
| Target design 2: "Train with `model=<name>`, deploy with `checkpoint=<name>`; nothing is copied or retyped" | Bob's joint list differs between the H-H and the H-R stage of the same model, and the training defaults differ from all package configs (finding 13). One file per model works only if the joint selection moves out of the model config. |
| Target design 3: "validation each epoch" | CoMaD has no validation split (finding 9). The split has to be defined first. |
| Target design 5: readers return sequences "at 15 Hz"; "Optional: cache the parsed JSON once as `.npz`" | The stored synthetic data is not at 15 Hz for 59% of the files (finding 8), so the reader cannot fix it; the data has to be regenerated. The package configs state `rate_hz: 15`. |
| Potential Improvements: "Retrain Marginal on synthetic AMASS (pretraining)" | Marginal never uses Bob (finding 16). Its pretraining can read single-person AMASS sequences directly. |

### Additions to the proposed design

1. **Regenerate the synthetic data once**, together with the pelvis-orientation fix of Findings 1:
   - resample by time to exactly 15 Hz;
   - seed the pairing;
   - write one `.npz` per split with the 9 joints only, plus source dataset, subject and source rate per sequence.
2. **Hold each split as one tensor on the device and draw windows by index.** No DataLoader workers and no per-window objects. This replaces the "optional" cache of target design 5.
3. **Joint selection in the readers.** Every reader delivers Bob (and the robot) in the layout the model expects, so the model config needs one Bob description for all stages (finding 13).
4. **One training step for all stages**, built on the package's centring. A test checks that `forward` and `forward_inference` of the HR model agree.
5. **Validation:**
   - a CoMaD validation split by episode, taken from `train`;
   - a zero-velocity baseline logged next to the model;
   - the best and the last checkpoint, with optimizer and scheduler state;
   - the config in package format next to the checkpoint.
6. **Epochs and schedule per stage**, and a seed. `logging` instead of `print`, so Hydra's log file is filled.
7. **`evaluate.py`** filters by task and writes the metrics to a file in the run folder.
8. **A test that `import interact` loads none of hydra, matplotlib, tqdm, tensorboard.**
9. **Container:** the data folder of `run_container.sh` becomes configurable, and a detached mode lets a run outlive the terminal.

Order: 1 and 5 first, because every later comparison between runs depends on correct data and a validation number.
