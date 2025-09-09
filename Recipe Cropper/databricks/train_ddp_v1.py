"""Distributed Data Parallel (DDP) training script for semantic segmentation using Segformer."""

import os
import sys
import logging
import warnings
from typing import Callable, Tuple

import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import mlflow
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_score, f1_score, roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torchvision.transforms as T

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# -------------------------
# Setup MLflow experiment
# -------------------------

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(
    "<PATH>/RecipeCropper-PoC-DDP-Training-Experiment"
)

# -------------------------
# Setup logging
# -------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------
# Load Dataset
# -------------------------

# Get dataset name from environment variable
dataset_name = os.environ.get("DATASET_NAME")
logger.info("Dataset: %s", dataset_name)


# Custom Dataset
class CustomSegmentationDataset(Dataset):
    """
    A custom dataset class for loading image/annotation pairs for segmentation tasks.

    This dataset class is used with a DataLoader to load images and their corresponding
    annotations, preprocess them using a feature extractor, and map the raw labels to
    corrected labels.

    Args:
      image_dir: Path to the directory containing image files.
      annotation_dir: Path to the directory containing annotation files.
      feature_extractor: The feature extractor for the images and annotations.
      label_mapper: A function to map raw lablels to corrected labels.
      image_files: List of image file names.
      annotation_files: List of annotation file names.
      agumentation: Whether to apply data augmentations.
      percentage: Percentage of the dataset to use, for testing purposes.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        feature_extractor: SegformerFeatureExtractor,
        label_mapper: Callable,
        image_files: list[str] = None,
        annotation_files: list[str] = None,
        augmentation: bool = True,
        percentage: float = 1.0,
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.feature_extractor = feature_extractor
        self.label_mapper = label_mapper

        # If no image_files/annotation_files list is provided, generate it from the image directory
        self.image_files = self.image_files = (
            sorted(
                [
                    f
                    for f in os.listdir(image_dir)
                    if f.endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            if image_files is None
            else image_files
        )
        self.annotation_files = (
            sorted([f for f in os.listdir(annotation_dir) if f.endswith(".png")])
            if annotation_files is None
            else annotation_files
        )
        assert len(self.image_files) == len(
            self.annotation_files
        ), "Number of images and annotations should be the same"

        self.augmentation = augmentation

        # For development. Randomly sample a subset of the dataset
        if percentage < 1.0:
            subset_size = int(len(self.image_files) * percentage)
            indices = random.sample(range(len(self.image_files)), subset_size)
            self.image_files = [self.image_files[i] for i in indices]
            self.annotation_files = [self.annotation_files[i] for i in indices]

    def __len__(self) -> int:
        """
        Returns the total number of image/annotation pairs.

        Returns:
          The number of image/annotation pairs.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and its corresponding annotation at the given index.

        Args:
          idx: Index of the image/annotation pair to retrieve.

        Returns:
          A tuple containing the pixel values and labels tensors.
        """
        # Get file paths
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load and convert image to RGB and annotation to grayscale
        image = Image.open(image_path).convert("RGB")
        annotation = Image.open(annotation_path).convert("L")

        # Apply augmentations using a fixed seed for reproducibility
        if self.augmentation:
            seed = torch.randint(0, 2**32, (1,)).item()
            image, annotation = augment(image, annotation, seed)

        # Correct labels using label mapper
        raw_labels = torch.tensor(np.array(annotation))
        corrected_labels = self.label_mapper(raw_labels)

        # Use feature extractor to process image and labels
        encoded_inputs = self.feature_extractor(
            image, corrected_labels, return_tensors="pt"
        )

        # Get pixel values and labels from encoded inputs
        pixel_values = encoded_inputs["pixel_values"].squeeze(0)
        labels = encoded_inputs["labels"].squeeze(0)

        return pixel_values, labels


# Data Augmentation
def augment(
    image: Image.Image, mask: Image.Image, seed: int
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply data augmentation to an image and it's mask.

    This function performs the following augmentations:
        1. Color jitter: Randomly changes brightness, contrast, saturation, and hue.
        2. Random affine transformation: Applies random rotation, translation, scaling, and shearing.
        3. Horizontal flip: Randomly flips the image and mask horizontally.
        4. Vertical flip: Randomfly flips the image and mask vertically.
        5. Rotation: Rotates the image and mask by a random angle.

    Args:
        image: The input image to be augmented.
        mask: The corresponding segmentation mask of the input image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple with the the augmented image and mask.
    """
    torch.manual_seed(seed)

    # ColorJitter
    color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    image = color_jitter(image)

    # RandomAffine
    random_affine = T.RandomAffine(
        degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
    )
    image = random_affine(image)
    mask = random_affine(mask)

    # Horizontal Flip
    if random.random() > 0.5:
        image = T.functional.hflip(image)
        mask = T.functional.hflip(mask)

    # Vertical Flip
    if random.random() > 0.5:
        image = T.functional.vflip(image)
        mask = T.functional.vflip(mask)

    # Rotation
    angle = random.uniform(-45, 45)
    image = T.functional.rotate(image, angle)
    mask = T.functional.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)

    return image, mask


# Label Mapping
def map_labels(label_tensor: torch.Tensor) -> torch.Tensor:
    """
    Maps specific labels in a PyTorch tensor to new values.

    Args:
        label_tensor: Input PyTorch tensor containing labels.

    Returns:
        Mapped labels as a PyTorch tensor.
    """
    label_tensor = label_tensor.clone().detach().type(torch.uint8)
    label_tensor[label_tensor == 38] = 1  # Map label 38 to 1
    label_tensor[label_tensor == 0] = 0  # Ensure label 0 remains 0
    return label_tensor


# -------------------------
# Creating datasets
# -------------------------

# Paths to dataset
root_dir = (
    f"<PATH>/PoC/{dataset_name}/"
)
img_dir = os.path.join(root_dir, "images")
ann_dir = os.path.join(root_dir, "annotations")

# List all image and annotation files
image_files = sorted(
    [f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
)
annotation_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".png")])

# Split data intro training, validation, and test sets
train_imgs, test_imgs, train_anns, test_anns = train_test_split(
    image_files, annotation_files, test_size=0.1, random_state=42
)
train_imgs, valid_imgs, train_anns, valid_anns = train_test_split(
    train_imgs, train_anns, test_size=0.2, random_state=42
)

# Init feature extractor
feature_extractor = SegformerFeatureExtractor(reduce_labels=False, do_rescale=False)

# Create dataset instances
dataset = CustomSegmentationDataset(
    img_dir, ann_dir, feature_extractor, map_labels, percentage=1
)
train_dataset = CustomSegmentationDataset(
    img_dir, ann_dir, feature_extractor, map_labels, train_imgs, train_anns
)
valid_dataset = CustomSegmentationDataset(
    img_dir, ann_dir, feature_extractor, map_labels, valid_imgs, valid_anns
)
test_dataset = CustomSegmentationDataset(
    img_dir, ann_dir, feature_extractor, map_labels, test_imgs, test_anns
)

logger.info("Number of training examples: %d", len(train_dataset))
logger.info("Number of validation examples: %d", len(valid_dataset))
logger.info("Number of testing examples: %d", len(test_dataset))
logger.info("Total number of examples: %d", len(dataset))

# -------------------------
# Training Functions
# -------------------------


# Training function
def train(
    model: SegformerForSemanticSegmentation,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    # Trains the model for one epoch.

    Args:
      model: The segmentation model
      dataloader: DataLoader for the training data.
      optimizer: Optimizer for training.
      device: Device to run the training on.

    Returns:
      Average loss, IoU, and F1 score over the training epoch.
    """
    model.train()  # Set model to training mode

    # To track loss
    total_loss = 0
    # To track IoU and F1-score
    all_preds = []
    all_labels = []

    # Iterate over batches of data
    for batch in tqdm(dataloader):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Forward pass: compute predictions and loss
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Backwards pass: compute gradients and update model parameters
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagate
        optimizer.step()  # Update model parameters

        total_loss += loss.item()  # Accumulate loss

        # Calculate predictions
        preds = outputs.logits.argmax(dim=1)
        preds = (
            F.interpolate(
                preds.unsqueeze(1).float(),
                size=labels.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .long()
        )  # Resize to match labels
        all_preds.append(preds.cpu().numpy().flatten())
        all_labels.append(labels.cpu().numpy().flatten())

    # Calculate average loss and metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    assert (
        all_preds.shape == all_labels.shape
    ), f"Inconsistent shapes: {all_preds.shape} and {all_labels.shape}"  # Ensure predictions and labels are of same shape

    # Calculate IoU and F1-score
    iou = jaccard_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, iou, f1


# Evaluation function
def evaluate(
    model: SegformerForSemanticSegmentation,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluates the model.

    Args:
      model: The segmentation model.
      dataloader: DataLoader for the validation data.
      device: Device to run the evaluation on.

    Returns:
      Average loss, IoU, F1 score, and ROC-AUC score over the validation epoch.
    """
    model.eval()  # Set model to evaluation mode

    # To track loss
    total_loss = 0
    # To track IoU and F1-Score
    all_preds = []
    all_labels = []
    # To track ROC-AUC score
    all_probs = []

    with torch.no_grad():  # Disable gradient computation
        # Iterate over batches of data
        for batch in tqdm(dataloader):
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            # Forward pass: compute predictions and loss
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()  # Accumulate loss

            # Calculate predictions and probabilities
            logits = outputs.logits
            preds = logits.argmax(dim=1)
            preds = (
                F.interpolate(
                    preds.unsqueeze(1).float(),
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .long()
            )  # Resize to match labels
            probs = torch.softmax(logits, dim=1)[
                :, 1, :, :
            ]  # Class 1 is the foreground
            probs = F.interpolate(
                probs.unsqueeze(1),
                size=labels.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(
                1
            )  # Resize to match labels
            all_preds.append(preds.cpu().numpy().flatten())
            all_probs.append(probs.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    # Calculate average loss and metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    assert (
        all_preds.shape == all_labels.shape
    ), f"Inconsistent shapes: {all_preds.shape} and {all_labels.shape}"  # Ensure predictions and labels are of same shape

    # Calculate IoU and F1-score
    iou = jaccard_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    # Calculate ROC-AUC score if possible
    if len(np.unique(all_labels)) == 2:
        assert (
            all_probs.shape == all_labels.shape
        ), f"Inconsistent shapes for ROC AUC: {all_probs.shape} and {all_labels.shape}"  # Ensure probabilities and labels are of same shape
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:  # ROC-AUC can't be computed if there's only one class in the batch
        roc_auc = float("nan")

    return avg_loss, iou, f1, roc_auc


# Early Stopping
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience: How long to wait after the last time validation loss improved.
        delta: Minimum change in the monitored quantity to qualify as an improvement.
        mode: One of {"min", "max}. In "min" mode, training will stop when the metric
            has stopped decreasing. In "max" mode, when the metric has stopped increasing.
        verbose: If True, prints a message for each validation loss improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        mode: str = "min",
        verbose: bool = False,
    ):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        # Metric tracked can be minimized (Loss) or maximized (IoU, F1-Score)
        self.best_metric = float("inf") if mode == "min" else -float("inf")

    def __call__(self, metric: float, model: torch.nn.Module, path: str):
        """
        Checks if early stopping conditions are met and saves the model checkpoint if validation
        loss improves.

        Args:
            metric: The current validation metric.
            model: The model to save.
            path: The path to save the model checkpoint.
        """
        score = -metric if self.mode == "min" else metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model, path)
            self.counter = 0

    def save_checkpoint(self, metric: float, model: torch.nn.Module, path: str):
        """
        Saves the model when validation loss decreases.

        Args:
            metric: The current validation metric.
            model: The model to save.
            path: The path to save the model checkpoint.
        """
        if self.verbose:
            logger.info(
                "Validation metric improved (%.6f --> %.6f). Saving model...",
                self.best_metric,
                metric,
            )
        torch.save(model.state_dict(), path)
        self.best_metric = metric


# -------------------------
# Distributed Training
# -------------------------


# Setup DDP
def setup(rank: int, world_size: int, backend: str = "gloo"):
    """
    Initialize the process group for distributed training.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        backend: Backend for distributed training. "gloo" for CPU; "nccl" for GPU.
    """
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        logger.info("Backend: %s", backend)
        dist.init_process_group(backend, rank=rank, world_size=world_size)


# Cleanup DDP
def cleanup():
    """Cleanup the process group for distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Main Distributed Training function
def main(rank: int, world_size: int):
    """
    Main function to run distributed training:

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
    """
    setup(rank, world_size)

    # Training parameters
    num_epochs = 8
    learning_rate = 6e-5
    weight_decay = 1e-5  # L2 regularization. Set to 0 to disable
    n_splits = 4  # Number of folds for K-Fold Cross-Validation
    batch_size = 4  # Number of samples processed before model is updated

    # Set validation metric
    best_val_iou = -float("inf")
    mode = "max"  # Mode for validation metric optimization

    # Adjust batch size for DDP
    batch_size = batch_size // world_size

    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
    )

    # Model setup for DDP
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=2
    ).to(rank)

    # Set label mapping in model configuration
    id2label = {"0": "background", "1": "dish"}
    label2id = {v: k for k, v in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])

    # Init optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.1, patience=2, verbose=True
    )

    # Init early stopping and set saving paths
    early_stopping = EarlyStopping(patience=5, mode=mode, verbose=True)
    checkpoint_path = "<PATH>/PoC/best_model_ddp_ex01.pt"
    model_save_path = "<PATH>/PoC/model_ddp_ex01"

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("model_name", "nvidia/mit-b0")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            logger.info("Epoch %d/%d", epoch + 1, num_epochs)

            train_sampler.set_epoch(epoch)  # DDP
            val_sampler.set_epoch(epoch)  # DDP

            # Train and evaluate model
            train_loss, train_iou, train_f1 = train(
                model, train_loader, optimizer, rank
            )
            val_loss, val_iou, val_f1, val_roc_auc = evaluate(model, val_loader, rank)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_iou", train_iou, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_iou", val_iou, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)
            mlflow.log_metric("val_roc_auc", val_roc_auc, step=epoch)

            logger.info(
                "Train Loss: %.4f, IoU: %.4f, F1-Score: %.4f",
                train_loss,
                train_iou,
                train_f1,
            )
            logger.info(
                "Validation Loss: %.4f, IoU: %.4f, F1-Score: %.4f, ROC-AUC: %.4f",
                val_loss,
                val_iou,
                val_f1,
                val_roc_auc,
            )

            # Step the scheduler
            scheduler.step(val_iou)

            # Check early stopping
            early_stopping(val_iou, model, checkpoint_path)
            if early_stopping.early_stop:
                logger.info("Early stopping.")
                break

    # Load best model checkpoint
    model.load_state_dict(torch.load(checkpoint_path))

    # Save fine-tuned model and feature extractor if model improves
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        model.module.save_pretrained(model_save_path)
        feature_extractor.save_pretrained(model_save_path)
        logger.info("Best model saved to: %s", model_save_path)

    # Final evaluation on test set
    if rank == 0:  # Only main process evaluates model
        # Final best model results
        final_results = (val_loss, val_iou, val_f1, val_roc_auc)
        print(
            f"Final best model results: Loss: {final_results[0]:.4f}, IoU: {final_results[1]:.4f}, F1-Score: {final_results[2]:.4f}, ROC-AUC: {final_results[3]:.4f}"
        )

        # Final test results
        model = SegformerForSemanticSegmentation.from_pretrained(model_save_path).to(
            rank
        )
        test_loss, test_iou, test_f1, test_roc_auc = evaluate(model, test_loader, rank)
        print(
            f"Final Test results: Loss: {test_loss:.4f}, IoU: {test_iou:.4f}, F1-Score: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}"
        )

    logger.info("Model fine-tuning and logging complete.")
    cleanup()  # DDP


if __name__ == "__main__":
    logger.info("Starting the distributed training script...")
    world_size = torch.cuda.device_count()
    if world_size > 0:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        logger.error("No CUDA devices available. Exiting...")
