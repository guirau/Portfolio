# CLIP Model Development

## Overview
This section focuses on exploring zero-shot and few-shot finetuning techniques for OpenAI's CLIP (Contrastive Language-Image Pre-training) model ([GitHub repo](https://github.com/openai/CLIP))

### Purpose
- Explore zero-shot learning capabilities of CLIP
- Explore image and text preprocessing techniques
- Implement zero-shot image classification
- Develop few-shot finetuning techniques
- Create a balanced batch sampling approach

## Structure
```
model_dev/
│
├── demo_interacting_with_clip.ipynb
│
├── few_shot/
│   ├── few_shot_finetuning.ipynb
│   ├── balanced_batch_sampler.py
│   ├── dataset/
│   │   └── ... (small dataset for few-shot training)
│   └── test_imgs/
│       └── ... (test images)
│
├── toy_dataset/
│   └── ... (small dataset for initial experiments)
│
├── google_open_images_dataset/
│   └── ... (images from Google Open Images dataset)
```

## Contents

### Notebooks

#### 1. `demo_interacting_with_clip.ipynb`
A self-contained notebook demonstrating fundamental interactions with CLIP models.

Downloads and runs CLIP models, calculates similarity between image and text inputs, and performs zero-shot image classifications. Example with CIFAR100 dataset and Google Open Images dataset

#### 2. `few_shot_finetuning.ipynb`
A notebook focused on fine-tuning the CLIP model with a small dataset. Performs model training and inference.

### Python Scripts

#### 1. `balanced_batch_sampler.py`
A custom batch sampler to ensure balanced representation of classes during training. Creates batches with a balanced distribution of classes.

## Datasets

- `toy_dataset/`: A very small dataset used for initial experiments and model interaction.
- `google_open_images_dataset/`: Images sourced from the Google Open Images dataset.
- `few_shot/dataset/`: A small dataset specifically prepared for few-shot training.

## Requirements

- PyTorch
- torchvision
- CLIP
- TensorBoard
- torchmetrics
- NumPy
