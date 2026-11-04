# InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions

[Kushal Kedia](https://kushal2000.github.io/), [Atiksh Bhardwaj](https://portal-cornell.github.io/), [Prithwish Dan](https://portfolio-pdan101.vercel.app/), [Sanjiban Choudhury](https://www.sanjibanchoudhury.com/)

Cornell University

**ICRA 2024** 

[Project Page](https://portal-cornell.github.io/interact/) | [arxiv](https://arxiv.org/abs/2311.12943)


## Installation

Follow these steps to install `InteRACT`:

1. Create and activate the conda environment:
   ```bash
   cd interact
   conda create --name interact python=3.8.16
   conda activate interact
   pip install -r requirements.txt
   pip install -e . 
   ```

## Use as a package

The model code installs as the package `interact`. Running the models needs only `torch`, `numpy`, `torch-dct` and `pyyaml`; training and visualisation need the `train` extra.

```bash
pip install git+https://github.com/GKnerd/interact.git@<tag>   # models only
pip install -e .[train]                                         # development and training
```

Each released checkpoint has a config in `interact/config/<name>.yaml`:

| Config | Checkpoint | Forecast function |
|---|---|---|
| `marginal` | HH Marginal | `marginal_forecast(model, history)` |
| `marginal_2hist` | HH Marginal-2Hist | `human_human_marginal_forecast(model, history, partner_history)` |
| `interact_hh` | HH InteRACT | `human_human_conditional_forecast(model, history, partner_history, partner_future)` |
| `interact_align_hr` | HR InteRACT-Align | `human_robot_conditional_forecast(model, history, robot_history, robot_future)` |

```python
from interact import build_model, load_config, marginal_forecast

config = load_config('marginal')
model = build_model(config, weights_path='Marginal/30.model', device='cuda')
forecast = marginal_forecast(model, history)
```

- `build_model(config, weights_path=None, device='cpu')` builds the model, loads the weights strictly with `weights_only=True` and returns it in eval mode. Without `weights_path` the weights stay untrained.
- The forecast functions take `history` as (B, 15, 9, 3) joint positions in metres and return (B, 15, 9, 3) in the same frame. Partners are (B, T, 9, 3), robot points (B, T, 2, 3); of a future only the last frame is used. Centring on the last history frame and the offset back are done inside.
- Inputs must be in the frame of the training data (`up_axis` in the config) and on the model's device.

Config keys:

| Key | Meaning |
|---|---|
| `model_class`, `model` | Model class and its constructor arguments |
| `joint_names` | SMPL-H names of the 9 joints, in the model's order |
| `parents` | Index into `joint_names`, -1 = root |
| `up_axis` | Up axis of the training data |
| `rate_hz` | Frame rate |
| `history_len`, `horizon_len` | Input and forecast frames |
| `num_persons` | Persons the model takes as input |

## Preliminaries

1. Set the `base_dev_dir` to your working directory in all of the config files

2. **Create the data directory**: Create a new directory called `"data"` under interact so that the repo has the following structure:

  ### Repo Structure
  ```
  ├── config
  │   ├── *.yaml files 
  ├── interact
  |   ├── checkpoints
  |     ├── HH_checkpoints
  |     ├── HR_checkpoints
  |   ├── data
  |     ├── comad_data
  |     ├── amass
  |     ├── cmu_mocap (optional)     
  |   ├── model
  |     ├── model architecture files...
  |   ├── utils
  |     ├── utility files...
  |   ├── scripts
  |     ├── eval_hh.py / eval_hr.py        <- evaluation scripts
  |     ├── pretrain_intent_forecaster.py  <- pretraining on H-H 
  |     |── finetune_intent_forecaster.py  <- finetuning on H-H
  |     |── hr_transfer.py                 <- transferring to H-R
  |   ├── mapping
  |     ├── files for joint mapping...
  |   ├── body_models
  |     ├── SMPL skeleton file (used for AMASS data)
  |
  ├── environment.yml
  ├── README.md
  ├── setup.py


  ```

## Dataset Installation and Preprocessing

### AMASS 
Download datasets listed in ```configs/synthetic_amass.yaml``` from the official [AMASS website](https://amass.is.tue.mpg.de/). Use any of the SMPL-X or SMPL-H links to download each dataset.

The AMASS dataset contains data of single human motion. Preprocess this data to create synthetic two-human data:
```
python scripts/create_synthetic_amass.data.py
```
**Note**: Update the config file ```configs/synthetic_amass.yaml``` with the specific datasets you have in your data directory before running this script.

### CoMaD
Download the data from this link [Data](https://cornell.app.box.com/s/jb0wau30dqotcjsak78ks64ea1o88yan) into the correct data directory.

You can download the .zip file and extract the data using the following terminal commands:
```
wget https://cornell.box.com/shared/static/6ss0mfojdof8q1z9ru7go58rwxqbnel5.zip -O comad_data.zip
unzip comad_data.zip -d data/comad_data/
rm comad_data.zip
```


## Training

1. Run the pretraining script on large-scale H-H data:
   ```bash
   python scripts/pretrain_intent_forecaster.py
   ```
2. Run the finetuning script on H-H interaction data. 
    ```bash
    python scripts/finetune_intent_forecaster.py
    ```
3. Run the script to transfer the model to  the H-R setting. 
    ```bash
    python scripts/hr_transfer.py
    ```

## Evaluation

1. Run the evaluation script for H-H:
   ```bash
   python scripts/eval_hh.py
   ```
2. Run the evaluation script for H-R: 
    ```bash
    python scripts/eval_hr.py
    ```


### BibTeX
   ```bash
   @article{kedia2023interact,
    title={InteRACT: Transformer Models for Human Intent Prediction Conditioned on Robot Actions},
    author={Kedia, Kushal and Bhardwaj, Atiksh and Dan, Prithwish and Choudhury, Sanjiban},
    journal={arXiv preprint arXiv:2311.12943},
    year={2023}
  }
   ``` 

### Acknowledgement
* MRT is adapted from [MRT](https://github.com/jiashunwang/MRT)
