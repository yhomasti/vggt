# VGGT Evaluation

This repository contains code to reproduce the evaluation results presented in the VGGT paper.

## Table of Contents

- [Camera Pose Estimation on Co3D](#camera-pose-estimation-on-co3d)
  - [Model Weights](#model-weights)
  - [Setup](#setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Evaluation](#running-the-evaluation)
  - [Expected Results](#expected-results)
- [Checklist](#checklist)

## Camera Pose Estimation on Co3D

### Model Weights

We have addressed a minor bug in the publicly released checkpoint related to the TrackHead configuration. Specifically, the `pos_embed` flag was incorrectly set to `False`. The following checkpoint incorporates this fix by fine-tuning the tracker head with `pos_embed` as `True` while preserving all other parameters. This fix will be merged into the main branch in a future update.

```bash
wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
```

Note: The default checkpoint remains functional, though you may observe a slight performance decrease (approximately 0.3% in AUC@30) when using Bundle Adjustment (BA). If using the default checkpoint, ensure you set `pos_embed` to `False` for the TrackHead. This modification only affects tracking-based evaluations and has no impact on feed-forward estimation performance, as tracking is not utilized in the feed-forward approach.

### Setup

Install the required dependencies:

```bash
# Install VGGT as a package
pip install -e .

# Install evaluation dependencies
pip install pycolmap==3.10.0 pyceres==2.3

# Install LightGlue for keypoint detection
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .
cd ..
```

### Dataset Preparation

1. Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d)

2. Preprocess the dataset (approximately 5 minutes):
```bash
python preprocess_co3d.py --category all \
    --co3d_v2_dir /YOUR/CO3D/PATH \
    --output_dir /YOUR/CO3D/ANNO/PATH
```

   Replace `/YOUR/CO3D/PATH` with the path to your downloaded Co3D dataset, and `/YOUR/CO3D/ANNO/PATH` with the desired output directory for the processed annotations. Note that the processed data here uses the PyTorch3D camera convention, while the annotation files we provided for training on Hugging Face have already been converted to the OpenCV convention.



### Running the Evaluation

Choose one of these evaluation modes:

```bash
# Standard VGGT evaluation
python test_co3d.py \
    --model_path /YOUR/MODEL/PATH \
    --co3d_dir /YOUR/CO3D/PATH \
    --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
    --seed 0

# VGGT with Bundle Adjustment
python test_co3d.py \
    --model_path /YOUR/MODEL/PATH \
    --co3d_dir /YOUR/CO3D/PATH \
    --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
    --seed 0 \
    --use_ba
```




### Expected Results

#### Quick Evaluation
Full evaluation on Co3D can take a long time. For faster trials, you can run with ```--fast_eval```. This does exactly the same but limiting to evaluate over at most 10 sequence per category.

Use `--fast_eval` to test on a subset of data (max 10 sequences per category):

- Feed-forward estimation:
  - AUC@30: 89.98
  - AUC@15: 83.89
  - AUC@5: 67.45
  - AUC@3: 56.65

- With Bundle Adjustment (`--use_ba`):
  - AUC@30: 90.52
  - AUC@15: 85.08
  - AUC@5: 70.69
  - AUC@3: 61.32

#### Full Evaluation

- Feedforward estimation achieves a Mean AUC@30 of 89.5% (slightly higher than the 88.2% reported in the paper due to implementation differences)
- With Bundle Adjustment, you can expect a Mean AUC@30 between 90.5% and 92.5%

> **Note:** For simplicity, this script did not optimize the inference speed, so timing results may differ from those reported in the paper. For example, when using ba, keypoint extractor models are re-initialized for each sequence rather than being loaded once.
