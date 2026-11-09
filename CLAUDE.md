# CLAUDE.md

Fork of InteRACT (Kedia et al., ICRA 2024): transformer models that forecast a person's upper-body
motion, optionally conditioned on a second agent (human or robot). Remote `origin` is
`GKnerd/interact`, the working branch is `release`. The package `interact` is used by `hpp_deployment` through its public API. This repo also holds the training scripts and the training container.

## Where to find information
- `TODO.md`: what is done, what comes next, open issues. Read it first.
- `ADDITIONAL_IMPROVEMENTS.md`: findings about the model and its training data, ideas for later,
  the proposed training pipeline redesign.
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

## End of a session
When asked, update `TODO.md`: what changed, decisions with their reason, next step. New findings
go into `ADDITIONAL_IMPROVEMENTS.md`.
