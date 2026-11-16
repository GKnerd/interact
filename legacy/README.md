# Original InterACT pipeline

The training and evaluation code of InterACT (Kedia et al., ICRA 2024) as it was before the
pretraining moved to AMASS with heading and lean (`scripts/convert_amass.py`). Kept so the paper's
pipeline can still be run. It is not developed further.

## Contents
- `scripts/create_synthetic_amass_data.py`: synthetic two-person AMASS (Alice and a random Bob).
- `scripts/finetune_intent_forecaster.py`: fine-tuning on CMU-Mocap, synthetic AMASS and CoMaD.
- `scripts/hr_transfer.py`: transfer to human-robot data (CoMaD-HR).
- `scripts/eval_hh.py`, `scripts/eval_hr.py`, `scripts/eval_visualize_hr.py`, `scripts/benchmark.py`:
  evaluation of the Cornell checkpoints on CoMaD and CoMaD-HR.
- `scripts/legacy_utils/`: their datasets and helpers, moved out of the `interact` package.
- `config/`: their configs. The scripts read `../config`, so they find these.

## Run order (in the container)
```
python legacy/scripts/create_synthetic_amass_data.py
python legacy/scripts/finetune_intent_forecaster.py
python legacy/scripts/eval_hh.py
```

## State
- Synthetic AMASS was compared with the new AMASS pretraining at the git tag
  `synthetic-amass-baseline`; the synthetic data makes the model forecast motion that never happened.
- The original conversion drops heading and lean and adds a fake translation when people turn.
- Only the imports and the `Synthetic_AMASS` call in `finetune_intent_forecaster.py` were adapted
  for the move. The scripts are not tested end to end.
