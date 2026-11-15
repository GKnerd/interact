# CLAUDE.md

Fork of InteRACT (Kedia et al., ICRA 2024): transformer models that forecast a person's upper-body
motion, optionally conditioned on a second agent (human or robot). Remote `origin` is
`GKnerd/interact`, the working branch is `*_devel`, with * being dependent on the architecture. The package `interact` is used by deployment only through its public API. This repo also holds the training scripts and the training container.

## Where to find information
- Obsidian notes, read first: `~/bwSyncShare/Personal/R&D/Projects/Human Pose Prediction Baseline Model Training/`
  - `InterACT - Code Status.md`: current scope, open to-dos, completed steps.
  - `InterACT - Findings.md`: findings about the model and its training data.
  - `InterACT - Verified Improvements.md`: completed steps confirmed by a real training run.
- Being phased out into the Obsidian notes; they still hold what has not moved yet:
  - `TODO.md`: package status, container, weights, licences, open code items.
  - `ADDITIONAL_IMPROVEMENTS.md`: findings on CoMaD, H-R and evaluation; the pipeline redesign.
- `README.md`, section "Use as a package": the public interface and the checkpoint configs.
- `~/Projects/cognisafe3d/hpp_migration/hpp_deployment_docker/STATUS.md`: deployment side and
  adapter contract.
- Details about the model (joints, rates, classes) come from the code and the configs in
  `interact/config/`, not from this file.

## Rules
- The model code runs unchanged with the deployment versions. `docker/Dockerfile.x86_64` and
  `docker/Dockerfile.aarch64` (Jetson Orin) pin the same torch and numpy.
  The package declares lower bounds only, never pins torch, and keeps working with numpy 1.26.
- Importing `interact` never imports training-only packages (hydra, matplotlib, tqdm,
  tensorboard).
- Existing checkpoints must keep loading strictly: no removing or renaming modules in
  `interact/model/`.
- The public interface changes only together with the adapter in `hpp_deployment`. A new
  checkpoint gets a new config file.
- Module constants go into `interact/constants.py`: plain, immutable values, no imports from
  `interact`.
- Docker images are built by the user.
- Commits are performed by the user.

## Running
- `./docker/build_image.sh`, `./docker/run_container.sh`; in the container the repo is at
  `/home/interact/interact`, data at `~/data`, outputs at `~/outputs`.
- Tests: `python -m pytest tests` in the container.
- Claude inside the container cannot see Obsidian notes.

## End of a session
hen asked, update the Obsidian notes:
- `InterACT - Code Status.md`: a completed step gets an entry under "Completed" (Why, What changed,
  Files, Checked, Not checked) and leaves the to-do list; partly done items are split.
- New findings go into `InterACT - Findings.md` under their own heading.
`TODO.md` and `ADDITIONAL_IMPROVEMENTS.md` get no new content.
