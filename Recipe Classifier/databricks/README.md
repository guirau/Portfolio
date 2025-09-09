# CLIP Fine-Tuning for Recipe Image Classification
This repository contains a collection of notebooks and utilities for fine-tuning OpenAI's [CLIP](https://github.com/openai/CLIP) model on step recipe images, with modular design and experiment tracking.

## Repository Structure

### Core Notebooks

#### 1. Recipe Classifier Notebooks
- `RecipeClassifier_v1.ipynb`: Base implementation of CLIP fine-tuning pipeline
- `RecipeClassifier_v2_with_GridSearch.ipynb`: Enhanced version including hyperparameter optimization

Both versions include:
- Modular architecture
- Balanced data loading and augmentation
- MLflow experiment tracking
- Early stopping and validation
- Production-ready inference

#### 2. Dataset Preparation
- `dataset_s3_pipeline.ipynb`: Scripts for preprocessing datasets in Databricks
 - Downloads and organizes images from S3
 - Handles both Recipe Classifier and Recipe Cropper datasets
 - Implements file deduplication and normalization
 - Includes augmentation utilities

### Supporting Files
- `balanced_batch_sampler.py`: Implementation of balanced batch sampling for handling class imbalance

## Core Components

### Modules
- **TrainingConfig**: Centralized configuration module
- **DataModule**: Dataset management and preprocessing
- **CLIPModule**: Model initialization and setup
- **CLIPTrainer**: Training orchestration
- **GridSearchTrainer** (v2): Hyperparameter optimization

### Features
- Centralized configuration management
- Balanced data sampling
- Custom augmentation pipeline
- MLflow integration
- Production-optimized inference
- Optional grid search (v2)

## Requirements
- PyTorch with CUDA support
- [OpenAI CLIP](https://github.com/openai/CLIP)
- MLflow
- torchmetrics
- Pillow
- numpy

## How to Use
1. Use `dataset_s3_pipeline.ipynb` to prepare the dataset
2. Choose between v1 (base) or v2 (with grid search) implementation
3. Configure parameters in TrainingConfig
4. Execute training pipeline
5. Monitor experiments via MLflow
