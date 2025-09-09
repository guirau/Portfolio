# Dataset Preparation

## Overview
This directory contains tools for preparing and analyzing the recipe image dataset, focusing on data preprocessing and exploration.

## Structure
```
dataset_prep/
├── eda.ipynb
├── preprocessing_pipeline.py
└── (generated files)
    ├── metadata.csv
    ├── invalid_filenames.txt
    ├── tmp/
    │   ├── original/
    │   └── mask/
    └── ImageFolder/
        ├── original/
        └── mask/
```

## Exploratory Data Analysis (EDA)

- `eda.ipynb`

### Purpose
- Analyze keyword distribution
- Identify data imbalances
- Standardize dataset keywords
- Visualize dataset composition

## Preprocessing Pipeline

- `preprocessing_pipeline.py` ([Documentation](preprocessing_pipeline.md))

### Core Functionality
- Download images from Google Drive
- Extract and validate metadata
- Organize images into structured dataset
- Maintain image-mask pair integrity
