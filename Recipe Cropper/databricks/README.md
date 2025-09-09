# Recipe Cropper

An ML-powered API that automates background removal from recipe images. This project explores various model architectures, training methodologies, and segmentation techniques to achieve the best results.

## Overview

**Recipe Cropper** is an image segmentation project that automates the post-production of step recipe images created at a FoodTech company's Creative Studio. Currently, this process relies on semi-automated Photoshop pipelines that require manual cropping or mask creation for each image. The goal of this project is to fully automate this pipeline.

By integrating **Recipe Cropper** with **Recipe Classifier** (a separate project developed in parallel), step images are automatically processed without manual intervention. **Recipe Classifier** first categorizes the images and routes them through a specific Photoshop pipeline, after which **Recipe Cropper** generates segmentation masks for background removal. This enables fully automated batch processing in post-production.

This project delivers an API endpoint that seamlessly integrates with existing Photoshop workflows, automating the cropping process previously done manually.

## API Integration

Code for the FastAPI application can be found under [`fastapi/`](../fastapi) 

## Repository Contents / Code Evolution

This repository demonstrates the progression of code quality, implementation improvements, and overall simplification across different versions.

### Version 1 (Proof of Concept)

- `RecipeCropper_v1.ipynb`
  - Initial proof of concept with CPU training
  - Implementation of various training strategies:
    - Standard training with train/validation/test splits
    - K-Fold Cross Validation
    - Distributed training using DDP
  - Experiment tracking with MLflow
  - Inference classes ready for deployment
  - Model deployment using Databricks

  **Notes:**
  - This version served as an experimental foundation. K-Fold Cross Validation lacked a dedicated test dataset for final evaluation, and distributed training had issues with properly broadcasting tensors across ranks.
  - This version should not be used as a reference, as later versions resolve these issues.

  **Results:**
  - The first working model achieved an **IoU of 0.9585** on the test dataset. It was deployed via Databricks and validated as a PoC by the Brand Marketing team.
  - To improve usability, a demo **Streamlit app** was developed as a frontend.
  - Example results [here](../README.md#results)

- `train_ddp_v1.py`
  - Initial attempt at distributed training
  - Known implementation issues:
    - Tensor broadcasting: Training states (learnign rates, early stopping) were not synchronized across GPU processes.
    - Evaluation workflow: Rank 0 handles evaluation, but all ranks loaded the model into GPU memory unnecessarily.
    - Model saving: Checkpoint saving was not synchronized, potentially leading to corrupted files.
    - MLflow logging: Metrics were logged independently by each process, causing redundant logging.

- `streamlit/streamlit.py` - Initial frontend prototype

### Version 2

- `RecipeCropper_v2_SegFormer_CPU.ipynb`
  - Refined training and inference pipeline for CPU
  - Improved data handling, including augmentation for artificial oversampling
  - Enhanced code structure and clarity
  - Experiment tracking with MLflow
  - Model deployment via Databricks (though no longer the preferred approach)

- `RecipeCropper_v2_SegFormer_GPU.ipynb` **(Production Release)**
  - GPU-accelerated training with optimized memory storage
  - Same structural improvements as v2 (CPU)

  **Results (Production Release):**
  - This version was selected for production after training on **5,000+ images and masks** using a base SegFormer model.
  - Achieved an **IoU of 0.9464** on the evaluation set.
  - To facilitate testing for end users, a demo frontend was built using **TypeScript and Next.js**.

  **Notes:**
    - Version **2.2**, using `briaai/RMBG-2.0`, produced good enough Zero-Shot results but was not used in production due to licensing restrictions preventing commercial use.

  - `recipeclassifier/fastapi_nextjs/demo/` - Production-ready demo frontend (built within Recipe Classifier project)
    - Built using **Next.js and TypeScript**
    - Enables users to classify and crop images via file upload or URL input.

#### Version 2.1 - SAM Variants

- `RecipeCropper_v2_SAM_points_GPU.ipynb`
  - Production-ready **SAM2** implementation
  - GPU-accelerated training with mixed precision
  - Model trained for point-based prompt segmentation
  - MLflow experiment tracking
  - Inference with point prompts and visualization

- `RecipeCropper_v2.1_SAM_points+boxes_GPU.ipynb`
  - Improved **SAM2** implementation with support for both **points and boxes** as prompts
  - Retains all features from the point-only version

  **Notes:**
  - SAM performed well in Zero-Shot settings but added excessive computational overhead, making it unsuitable for this application.

#### Version 2.2 - BRIA AI Variant

- `RecipeCropper_v2.2_BRIAAI.ipynb`
  - Production-ready implementation using **briaai/RMBG-2.0** from HuggingFace
  - Clean and maintainable codebase with improvements over previous versions
  - Supports single-CPU and multi-GPU trainng
  - Inference pipeline

- `train_ddp_v2.2_BRIAAI.py`
  - Production-ready distributed training
  - Efficient multi-GPU scaling
  - Refined, maintainable implementation with all previous features working seamlessly

  **Notes:**
  - Although this version produced good enough Zero-Shot results, it was not used in production due to commercial licensing restrictions.
