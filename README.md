# Comparative Analysis of Bias Mitigation Methods without Sensitive Attributes

This repository provides a unified training and evaluation pipeline for multiple
debiasing methods on binary classification datasets with spurious correlations.
It uses Hydra for configuration management and supports consistent CSV-based
evaluation outputs for downstream analysis.

## What is in this repository

- Unified entrypoint for all methods: `main.py`
- Hydra configs for methods, datasets, and sweeps: `configs/`
- Method implementations and wrappers: `methods/` and `scripts/`
- Dataset metadata and split files: `data/fairface/`, `data/waterbirds/`
- Evaluation CSVs and analysis notebook: `evaluation/`
- Training outputs/checkpoints: `results/`

## Supported methods

You can run and analyze the following methods through the same interface:

- `debiasify` (self-distillation based debiasing)
- `disent` (disentangled feature augmentation)
- `jtt` (Just Train Twice)
- `resnet` (ResNet18 baseline)

Method selection is controlled by `method=<name>` in Hydra overrides.

## Supported datasets

Current configs support:

- `fairface` (binary target + sensitive attribute metadata)
- `waterbirds` (target bird class + background/place confounder)

Default dataset in `configs/base.yaml` is `fairface`.

## Repository structure (high level)

```text
.
├── main.py                        # Hydra entrypoint
├── configs/                       # Base + method configs + sweep configs
│   ├── base.yaml
│   ├── base_debiasify.yaml
│   ├── base_disent.yaml
│   ├── base_jtt.yaml
│   ├── base_resnet.yaml
│   └── method/
│       ├── debiasify.yaml
│       ├── disent.yaml
│       ├── jtt.yaml
│       └── resnet.yaml
├── scripts/                       # Method dispatch wrappers called by main.py
├── methods/                       # Method implementations
├── data/
│   ├── fairface/
│   └── waterbirds/
├── evaluation/                    # Generated prediction CSVs + analysis notebook
└── results/                       # Checkpoints and experiment outputs
```

## Environment setup

### Option A: Conda from `env.yml` (recommended)

```bash
conda env create -f env.yml
conda activate env
```

### Option B: Existing Python environment

Create and activate your environment first (venv/conda), then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some method folders also provide their own requirement files:

- `methods/jtt/requirements.txt`
- `methods/disent/requirements.txt`

## Dataset layout and metadata

The project expects split CSV files under each dataset folder and image data
under a local `data/` subfolder.

Expected folder:

```text
data/fairface/
├── train.csv
├── val.csv
├── test.csv
├── metadata.csv
└── data/                # images
```

To regenerate `metadata.csv` (adds split ids and merges train/val/test):

```bash
cd data/fairface
python create_metadata.py
```

## How runs work

All experiments are launched through Hydra in `main.py`.

Execution flow:

1. Hydra loads `configs/base.yaml` (or a different base config you provide).
2. `method.name` chooses a wrapper module: `scripts/run_<method>.py`.
3. Wrapper script runs method-specific train/test scripts.
4. Checkpoints are written under `results/...`.
5. Evaluation writes per-sample predictions to `evaluation/*.csv`.

`stage` controls mode:

- `stage=train`: train a model
- `stage=test`: evaluate checkpoint and export prediction CSV

## Config system and key files

### Core config

- `configs/base.yaml`
	- Global defaults (`seed`, `stage`, default dataset)
	- Default method group selection

### Method configs

- `configs/method/debiasify.yaml`
- `configs/method/disent.yaml`
- `configs/method/jtt.yaml`
- `configs/method/resnet.yaml`

These define method-specific hyperparameters, output paths, and checkpoint
selection behavior for testing.

### Sweep configs

- `configs/base_debiasify.yaml`
- `configs/base_disent.yaml`
- `configs/base_jtt.yaml`
- `configs/base_resnet.yaml`

Each sweep config defines `hydra.sweeper.params` for multi-run experiments.

## Running hyperparameter sweeps

Use Hydra multirun with `-m` and select an appropriate base sweep config.

Example:

```bash
# Debiasify sweep over gamma / alpha / warmup
python main.py --config-name base_debiasify -m
```

## Outputs and experiment artifacts

### Checkpoints

Saved under method-specific `results_dir` from config, for example:

- `results/debiasify/<dataset>/<exp_name>/...`
- `results/disent/<dataset>/<exp_name>/...`
- `results/jtt/<dataset>/<exp_name>/...`
- `results/resnet18_baseline/<dataset>/<exp_name>/...`

### Evaluation CSV format

Method test scripts export per-sample prediction tables:

- `img_path`
- `y_true`
- `attr_true`
- `y_pred`
- `p_0`, `p_1`, `p_max`
- `correct`

### Evaluation files currently present

The `evaluation/` directory already contains method/dataset result CSVs for both
average and worst-group analyses, including files such as:

- `debiasify_*`
- `disent_*`
- `jtt_*`
- `resnet_*`

and the notebook:

- `evaluation/analysis.ipynb`


## Papers implemented in this repository

```bibtex
@inproceedings{disent,
      title={Learning Debiased Representation via Disentangled Feature Augmentation}, 
      author={Jungsoo Lee and Eungyeup Kim and Juyoung Lee and Jihyeon Lee and Jaegul Choo},
      year={2021},
      url={https://arxiv.org/abs/2107.01372}, 
}

@inproceedings{debiasify,
      title={Debiasify: Self-Distillation for Unsupervised Bias Mitigation}, 
      author={Nourhan Bayasi and Jamil Fayyad and Ghassan Hamarneh and Rafeef Garbi and Homayoun Najjaran},
      year={2024},
      url={https://arxiv.org/abs/2411.00711}, 
}

@inproceedings{jtt,
      title={Just Train Twice: Improving Group Robustness without Training Group Information}, 
      author={Evan Zheran Liu and Behzad Haghgoo and Annie S. Chen and Aditi Raghunathan and Pang Wei Koh and Shiori Sagawa and Percy Liang and Chelsea Finn},
      year={2021},
      eprint={2107.09044},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2107.09044}, 
}

@inproceedings{resnet,
	title={Deep Residual Learning for Image Recognition},
	author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages={770--778},
	year={2016}
}
```
