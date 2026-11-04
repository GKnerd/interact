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
