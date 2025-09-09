"""Distributed Data Parallel (DDP) training script for semantic segmentation using biraai/RMBG-2.0."""

import gc
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# from safetensors import safe_open
from safetensors.torch import load_file
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

root_dir = Path("<PATH>")

# -------------------------
# Setup MLflow experiment
# -------------------------

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(
    "<PATH>/RecipeCropper-PoC-DDP-v2-Training-Experiment"
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
# GPU memory settings
# -------------------------

# PyTorch's CUDA memory allocator
# - max_split_size_mb: Maximum size of a memory block
# - expandable_segments: Allows dynamic expansion while keeping block size limit
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

# Enable cuDNN benchmark mode
torch.backends.cudnn.benchmark = True

# -------------------------
# Load Dataset
# -------------------------

# Get dataset name from environment variable
# dataset_name = os.environ.get("DATASET_NAME")
# logger.info("Dataset: %s", dataset_name)


# Custom Dataset
class BriaaiDataset(Dataset):
    """
    Dataset class for briaai/RMBG-2.0 optimized for DDP.
    Model card: https://huggingface.co/briaai/RMBG-2.0

    Args:
        image_dir: Path to dir containing images.
        mask_dir: Path to dir containing masks.
        image_files: List of image file names.
        mask_files: List of mask file names.
        augmentation: Whether to apply data augmentation.
        image_size: Size of the images to be returned.
        world_size: Total number of distributed processes.
        rank: Rank of the current process.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_files: list[str] = None,
        mask_files: list[str] = None,
        augmentation: bool = False,
        image_size: tuple = (1024, 1024),
        world_size: int = 1,
        rank: int = 0,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Validate matching files
        if len(image_files) != len(mask_files):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match"
            )

        # Distribute files across processes
        num_files = len(image_files)
        files_per_process = num_files // world_size
        start_idx = rank * files_per_process
        end_idx = start_idx + files_per_process if rank < world_size - 1 else num_files

        self.image_files = image_files[start_idx:end_idx]
        self.mask_files = mask_files[start_idx:end_idx]

        self.augmentation = augmentation
        self.image_size = image_size
        self.world_size = world_size
        self.rank = rank

        # Define preprocessing transofmrations
        self.image_transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Lambda(self._normalize_image),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.mask_transform = T.Compose([T.Resize(self.image_size), T.ToTensor()])

    # Private method to normalize image in image_transform
    # Defined outside __init__ to avoid pickling error
    @staticmethod
    def _normalize_image(x: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor."""
        return x / 255.0

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get an image, its mask, prompt points, and point labels."""
        # Load image and mask
        image_path = self.image_dir / self.image_files[idx]
        mask_path = self.mask_dir / self.mask_files[idx]

        # Convert to RGB and grayscale
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation if enabled
        if self.augmentation:
            seed = torch.randint(0, 2**32, (1,)).item()
            image, mask = augment(image, mask, seed)

        # Convert mask to binary
        binary_mask = (np.array(mask) > 0).astype(np.uint8)

        # Apply transformations
        image_tensor = self.image_transform(image)

        mask_tensor = torch.from_numpy(binary_mask)
        mask_tensor = (
            F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=self.image_size,
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )  # Remove batch and channel dims

        return image_tensor, mask_tensor


# Data Augmentation
def augment(
    image: Image.Image, mask: Image.Image, seed: int
) -> tuple[Image.Image, Image.Image]:
    """
    Applies data augmentation to an image and its mask.

    Performs the following augmentations in sequence:
        1. Color jitter (image only): Randomly changes brightness, contrast, saturation, and hue.
        2. Random affine transformation: Applies random rotation, translation, scaling, and shearing.
        3. Horizontal flip: Randomly flips the image and mask horizontally.
        4. Vertical flip: Randomfly flips the image and mask vertically.

    Args:
        image: The input image to be augmented.
        mask: The corresponding segmentation mask of the input image.
        seed: Random seed for reproducibility.
        rank: Process rank for distributed training.

    Returns:
        Tuple with the the augmented (image, mask) pair.
    """
    torch.manual_seed(seed)

    # ColorJitter
    color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    image = color_jitter(image)

    # RandomAffine
    affine = T.RandomAffine(
        degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
    )  # Set affine params once and apply to both image and mask
    params = affine.get_params(
        affine.degrees, affine.translate, affine.scale, affine.shear, image.size
    )
    image = T.functional.affine(image, *params)
    mask = T.functional.affine(mask, *params)

    # Horizontal Flip
    if random.random() > 0.5:
        image = T.functional.hflip(image)
        mask = T.functional.hflip(mask)

    # Vertical Flip
    if random.random() > 0.5:
        image = T.functional.vflip(image)
        mask = T.functional.vflip(mask)

    return image, mask


# -------------------------
# Creating datasets
# -------------------------


def create_datasets(dataset_dir: Path, world_size: int = 1, rank: int = 0):
    """Create and split datasets."""
    # Paths to images/masks in dataset
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "annotations"

    # List all image and mask files
    VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    VALID_MASK_EXTENSIONS = {".png", ".jpg"}
    image_files = sorted(
        [
            f.name
            for f in img_dir.iterdir()
            if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    )
    mask_files = sorted(
        [
            f.name
            for f in mask_dir.iterdir()
            if f.suffix.lower() in VALID_MASK_EXTENSIONS
        ]
    )

    # Split data intro train/val/test sets
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=42
    )
    train_imgs, valid_imgs, train_masks, valid_masks = train_test_split(
        train_imgs, train_masks, test_size=0.2, random_state=42
    )

    # Create dataset instances
    train_dataset = BriaaiDataset(
        img_dir,
        mask_dir,
        train_imgs,
        train_masks,
        augmentation=False,
        world_size=world_size,
        rank=rank,
    )
    valid_dataset = BriaaiDataset(
        img_dir,
        mask_dir,
        valid_imgs,
        valid_masks,
        augmentation=False,
        world_size=world_size,
        rank=rank,
    )
    test_dataset = BriaaiDataset(
        img_dir,
        mask_dir,
        test_imgs,
        test_masks,
        augmentation=False,
        world_size=world_size,
        rank=rank,
    )

    return train_dataset, valid_dataset, test_dataset


# -------------------------
# Training Functions
# -------------------------

GRADIENT_ACCUMULATION_STEPS = 2


# Training function
def train(
    model: AutoModelForImageSegmentation,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    rank: int = 0,
) -> tuple[float, float, float]:
    """
    Trains the biraai/RMBG-2.0 model for one epoch.

    Args:
        model: DDP-wrapped biraai/RMBG-2.0 model for semantic segmentation.
        dataloader: DataLoader for training data.
        optimizer: AdamW optimizer for training.
        device: Device to run training on.
        rank: Process rank.

    Returns:
        tuple: (Average loss, IoU, F1 score) for the training epoch.
    """
    torch.cuda.empty_cache()  # Free GPU memory
    model.train()  # Set to training mode

    # To track loss
    total_loss = 0
    # To track IoU and F1-score
    all_preds = []
    all_masks = []

    # Gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
        # Get batch data
        images, masks = batch
        images = images.to(
            device, non_blocking=True
        )  # non_blocking for async data transfer
        masks = masks.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            logits = outputs[0][0][0][0]
            # logits = outputs[3]
            logits = F.interpolate(
                logits, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            loss = (
                F.binary_cross_entropy_with_logits(logits, masks.unsqueeze(1).float())
                / GRADIENT_ACCUMULATION_STEPS
            )

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()  # Backpropagate

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Clip to prevent exploding gradients
            scaler.step(optimizer)  # Update model parameters
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Accumulate loss

        # Calculate predictions
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            preds = preds.squeeze(1)
            masks = masks.squeeze(1)
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())

        # Free GPU memory
        del outputs, preds
        torch.cuda.empty_cache()

    # Gather metrics from all processes
    world_size = dist.get_world_size()

    # Reduce loss
    total_loss = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(dataloader) * world_size)

    # Gather predictions
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Convert to tensors for gathering
    all_preds = torch.from_numpy(all_preds).to(device)
    all_masks = torch.from_numpy(all_masks).to(device)

    # Gather sizes
    local_size = torch.tensor([all_preds.shape[0]], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    # Pad and gather predictions
    max_size = max(size.item() for size in sizes)
    padded_preds = torch.zeros(max_size, device=device)
    padded_masks = torch.zeros(max_size, device=device)
    padded_preds[:local_size] = all_preds
    padded_masks[:local_size] = all_masks

    # Gather all data
    all_preds_list = [torch.zeros_like(padded_preds) for _ in range(world_size)]
    all_masks_list = [torch.zeros_like(padded_masks) for _ in range(world_size)]

    dist.all_gather(all_preds_list, padded_preds)
    dist.all_gather(all_masks_list, padded_masks)

    # Combine
    all_preds_combined = []
    all_masks_combined = []

    for i, size in enumerate(sizes):
        size = size.item()
        all_preds_combined.append(all_preds_list[i][:size])
        all_masks_combined.append(all_masks_list[i][:size])

    all_preds = torch.cat(all_preds_combined).cpu().numpy()
    all_masks = torch.cat(all_masks_combined).cpu().numpy()

    # Calculate metrics (only on rank 0)
    if rank == 0:
        assert (
            all_preds.shape == all_masks.shape
        ), f"Inconsistent shapes: {all_preds.shape} and {all_masks.shape}"

        # Calculate IoU and F1-score
        iou = jaccard_score(all_masks, all_preds, average="binary")
        f1 = f1_score(all_masks, all_preds, average="binary")
    else:
        iou = f1 = 0.0

    # Broadcast metrics from rank 0
    metrics = torch.tensor([iou, f1], device=device)
    dist.broadcast(metrics, src=0)
    iou, f1 = metrics.tolist()

    return avg_loss, iou, f1


# Evaluation function
def evaluate(
    model: AutoModelForImageSegmentation,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0,
) -> tuple[float, float, float, float]:
    """
    Evaluates the briaai/RMBG-2.0 model.

    Args:
        model: DDP-wrapped biraai/RMBG-2.0 model for semantic segmentation.
        dataloader: DataLoader for validation data.
        device: Device to run evaluation on.
        rank: Process rank.

    Returns:
        tuple: (Averge loss, IoU, F1 score, ROC-AUC score) for the validation epoch.
    """
    model.eval()

    # To track loss
    total_loss = 0
    # To track IoU and F1-score
    all_preds = []
    all_masks = []
    # To track ROC-AUC score
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=rank != 0):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            logits = outputs[3]
            logits = F.interpolate(
                logits, size=(1024, 1024), mode="bilinear", align_corners=False
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, masks.unsqueeze(1).float()
            )

            total_loss += loss.item()  # Accumulate loss

            # Calculate predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            preds = preds.squeeze(1)
            probs = probs.squeeze(1)
            masks = masks.squeeze(1)

            # Store predictions, masks, and probabilities
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.cpu().numpy().flatten())
            all_probs.append(probs.cpu().numpy().flatten())

            # Free GPU memory
            del outputs, preds, probs
            torch.cuda.empty_cache()

    # Gather metrics from all processes
    world_size = dist.get_world_size()

    # Convert to tensors for all_reduce
    total_loss = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(dataloader) * world_size)

    # Concatenate local predictions and gather across processes
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)
    all_probs = np.concatenate(all_probs)

    # Convert to tensors for gathering
    all_preds = torch.from_numpy(all_preds).to(device)
    all_masks = torch.from_numpy(all_masks).to(device)
    all_probs = torch.from_numpy(all_probs).to(device)

    # Gather sizes from all processes
    local_size = torch.tensor([all_preds.shape[0]], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)

    # Gather predictions from all processes
    max_size = max(size.item() for size in sizes)
    padded_preds = torch.zeros(max_size, device=device)
    padded_masks = torch.zeros(max_size, device=device)
    padded_probs = torch.zeros(max_size, device=device)

    # Pad local data
    padded_preds[:local_size] = all_preds
    padded_masks[:local_size] = all_masks
    padded_probs[:local_size] = all_probs

    # Gather all data
    all_preds_list = [torch.zeros_like(padded_preds) for _ in range(world_size)]
    all_masks_list = [torch.zeros_like(padded_masks) for _ in range(world_size)]
    all_probs_list = [torch.zeros_like(padded_probs) for _ in range(world_size)]

    dist.all_gather(all_preds_list, padded_preds)
    dist.all_gather(all_masks_list, padded_masks)
    dist.all_gather(all_probs_list, padded_probs)

    # Combine
    all_preds_combined = []
    all_masks_combined = []
    all_probs_combined = []

    for i, size in enumerate(sizes):
        size = size.item()
        all_preds_combined.append(all_preds_list[i][:size])
        all_masks_combined.append(all_masks_list[i][:size])
        all_probs_combined.append(all_probs_list[i][:size])

    all_preds = torch.cat(all_preds_combined).cpu().numpy()
    all_masks = torch.cat(all_masks_combined).cpu().numpy()
    all_probs = torch.cat(all_probs_combined).cpu().numpy()

    # Calculate metrics (only on rank 0)
    if rank == 0:
        assert (
            all_preds.shape == all_masks.shape
        ), f"Inconsistent shapes: {all_preds.shape} and {all_masks.shape}"  # Validate shapes

        # Calculate IoU and F1-score
        iou = jaccard_score(all_masks, all_preds, average="binary")
        f1 = f1_score(all_masks, all_preds, average="binary")

        # Calculate ROC-AUC score if possible
        if len(np.unique(all_masks)) == 2:
            assert (
                all_probs.shape == all_masks.shape
            ), f"Inconsistent shapes for ROC AUC: {all_probs.shape} and {all_masks.shape}"
            roc_auc = roc_auc_score(all_masks, all_probs)
        else:
            roc_auc = float("nan")
    else:
        iou = f1 = roc_auc = 0.0

    # Broadcast metrics form rank 0
    metrics = torch.tensor([iou, f1, roc_auc], device=device)
    dist.broadcast(metrics, src=0)
    iou, f1, roc_auc = metrics.tolist()

    return avg_loss, iou, f1, roc_auc


# Early Stopping
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Args:
        patience: Epochs to wait after last improvement of the validation metric.
        delta: Minimum change to qualify as an improvement.
        mode: Either "min" or "max". "min" stops when metric stops decreasing,
            "max" stops when metric stops increasing. Default: "min".
        verbose: If True, prints messages for metric improvements. Default: False.

    Example:
        early_stopping = EarlyStopping(patience=5, mode="max", verbose=True)
        for epoch in range(num_epochs):
            val_iou = validate_model()
            early_stopping(val_iou, model, optimizer, scheduler, save_path)
            if early_stopping.early_stop:
                break
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        mode: str = "min",
        verbose: bool = True,
        rank: int = 0,
    ):
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'.")

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = -float("inf") if self.mode == "max" else float("inf")
        self.early_stop = False
        self.counter = 0
        self.best_epoch = None
        self.rank = rank

    def __call__(
        self,
        epoch: int,
        metric: float,
        model: AutoModelForImageSegmentation,
        path: Path,
    ):
        """
        Checks early stopping conditions and save the model if metric improves.

        Args:
            epoch: Current epoch number
            metric: Current validation metric value
            model: briaai/RMBG-2.0 model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            path: Path to save the checkpoint
        """
        score = metric if self.mode == "max" else -metric

        # Check if metric improved
        improvement = False
        if self.mode == "min":
            improvement = score < self.best_score - self.delta
        elif self.mode == "max":
            improvement = score > self.best_score + self.delta

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.rank == 0:  # Only save on rank 0
                self.save_best_model(epoch, metric, model, path)
        elif improvement:
            if self.verbose and self.rank == 0:
                logger.info(
                    "Validation metric improved (%.4f --> %.4f). Saving model...",
                    self.best_score,
                    score,
                )
            self.best_score = score
            self.best_epoch = epoch
            if self.rank == 0:  # Only save on rank 0
                self.save_best_model(epoch, metric, model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose and self.rank == 0:
                logger.info("EarlyStopping counter: %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(
        self,
        epoch: int,
        metric: float,
        model: AutoModelForImageSegmentation,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        path: Path,
    ):
        """
        Save complete training state when validation metric improves.

        Args:
            epoch: Current epoch number.
            metric: Current validation metric value.
            model: Model to save if metric improves.
            optimizer. Optimizer state to save.
            scheduer: Scheduler state to save.
            path: Path to save the model checkpoint.
        """
        if self.rank == 0:
            path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),  # .module for DDP
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metric_value": metric,
            "best_score": self.best_score,
        }

        # Save checkpoint
        torch.save(checkpoint, path)

        if self.verbose:
            logger.info(
                "Saved checkpoint at epoch %d with metric value: %.4f", epoch, metric
            )

    def save_best_model(
        self,
        epoch: int,
        metric: float,
        model: AutoModelForImageSegmentation,
        path: Path,
    ):
        """
        Save best model.

        Args:
            epoch: Current epoch number.
            metric: Current validation metric value.
            model: Model to save.
            path: Path to save the best model.
        """
        if self.rank == 0:
            path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),  # .module for DDP
            "best_score": self.best_score,
        }

        best_path = (
            path.parent / "best_model.pt" if path.name != "best_model.pt" else path
        )

        # Save best model
        if metric == self.best_score:
            torch.save(checkpoint, best_path)
            logger.info(
                "[Best model] Saved best model at epoch %d with metric value: %.4f",
                epoch,
                metric,
            )


# -------------------------
# Helpers: Log memory usage
# -------------------------


def log_memory_usage(rank: int = 0):
    """Logs memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # in GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # in GB
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # in GB
        logger.info(
            "GPU:%d Memory: Allocated=%.2fGB, Reserved=%.2fGB, Peak=%.2fGB",
            rank,
            allocated,
            reserved,
            max_memory,
        )

        # Reset peak stats
        torch.cuda.reset_peak_memory_stats(rank)
    else:
        process = psutil.Process()
        memory_info = process.memory_info()
        rss = memory_info.rss / (1024**3)  # in GB
        vms = memory_info.vms / (1024**3)  # in GB
        logger.info("CPU Memory: RSS=%.2fGB, VMS=%.2fGB", rss, vms)

    gc.collect()  # Free CPU memory
    torch.cuda.empty_cache()  # Free GPU memory


# -------------------------
# Distributed Training
# -------------------------

models_dir = root_dir / "recipe_cropper" / "models"

# Clear sys.path of any previous models_dir and add new models_dir
sys.path = [p for p in sys.path if str(models_dir) != str(Path(p))]
sys.path.insert(0, str(models_dir))

from briaai.birefnet import BiRefNet
from briaai.BiRefNet_config import BiRefNetConfig


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
        dist.barrier()
        dist.destroy_process_group()


# Main Distributed Training function
def main(rank: int, world_size: int):
    """
    Main function to run distributed training:

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
    """
    try:
        setup(rank, world_size)

        # Create datasets
        DATASET_DIR = root_dir / "recipe_cropper" / "dataset" / "dataset_20dec"
        train_dataset, valid_dataset, test_dataset = create_datasets(
            DATASET_DIR, world_size, rank
        )

        # CONSTANTS
        MODEL_NAME = "briaai/RMBG-2.0"
        CHECKPOINT_DIR = (
            root_dir / "recipe_cropper" / "checkpoints" / "2025_02_18" / "briaai_2"
        )
        SAVE_INTERVAL = 2  # Checkpoint save interval
        NUM_EPOCHS = 50
        LEARNING_RATE = 1e-5
        WEIGHT_DECAY = 1e-4
        BATCH_SIZE = 4
        PATIENCE = 12  # Early stopping patience

        # Path to best model
        best_model_path = CHECKPOINT_DIR / "best_model.pt"

        # Create checkpoint dir (only on rank 0)
        if rank == 0:
            CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        dist.barrier()  # Wait for rank 0 to create dir

        # Adjust batch size for DDP
        BATCH_SIZE = BATCH_SIZE // world_size

        # Print dataset info
        logger.info(
            "Rank: %d - Number of training examples: %d", rank, len(train_dataset)
        )
        logger.info(
            "Rank: %d - Number of validation examples: %d", rank, len(valid_dataset)
        )
        logger.info(
            "Rank: %d - Number of testing examples: %d", rank, len(test_dataset)
        )

        # Create samplers with DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,  # Default is 2, set to 1 for better memory management
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1,
        )

        # Initialize model
        with open(models_dir / "briaai" / "config.json", "r") as f:
            config = json.load(f)

        # Initialize model from local files
        model = BiRefNet(BiRefNetConfig(**config))
        state_dict = load_file(models_dir / "briaai" / "model.safetensors")
        model.load_state_dict(state_dict)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model
        )  # Convert to sync batch norm
        model.to(f"cuda:{rank}")

        # Wrap model in DDP
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=4)

        # Initialize early stopping (only on rank 0)
        if rank == 0:
            early_stopping = EarlyStopping(patience=PATIENCE, mode="max", verbose=True)

        # Start MLflow run only on rank 0
        if rank == 0:
            mlflow.start_run()
            # Log hyperparameters
            mlflow.log_params(
                {
                    "model_name": MODEL_NAME,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE * world_size,
                    "num_epochs": NUM_EPOCHS,
                }
            )
            logger.info(
                "Learning rate: %.1e, Weight decay: %.1e, Batch size: %d",
                LEARNING_RATE,
                WEIGHT_DECAY,
                BATCH_SIZE * world_size,
            )

        # Training loop for all ranks
        for epoch in range(NUM_EPOCHS):
            if rank == 0:
                logger.info("Epoch %d/%d", epoch + 1, NUM_EPOCHS)
                log_memory_usage(rank)

            # Set epoch for all samplers
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

            # Train model
            train_loss, train_iou, train_f1 = train(
                model, train_dataloader, optimizer, f"cuda:{rank}"
            )

            # Syncronize metrics across processes
            train_metrics = torch.tensor(
                [train_loss, train_iou, train_f1], device=f"cuda:{rank}"
            )
            dist.all_reduce(train_metrics)
            train_loss, train_iou, train_f1 = train_metrics.tolist()
            train_metrics = [
                x / world_size for x in [train_loss, train_iou, train_f1]
            ]  # Avg across processes
            train_loss, train_iou, train_f1 = train_metrics

            # Evaluate model
            val_loss, val_iou, val_f1, val_roc_auc = evaluate(
                model, valid_dataloader, f"cuda:{rank}"
            )

            # Suncronize metrics across processes
            val_metrics = torch.tensor(
                [val_loss, val_iou, val_f1, val_roc_auc], device=f"cuda:{rank}"
            )
            dist.all_reduce(val_metrics)
            val_loss, val_iou, val_f1, val_roc_auc = val_metrics.tolist()
            val_metrics = [
                x / world_size for x in [val_loss, val_iou, val_f1, val_roc_auc]
            ]  # Avg across processes
            val_loss, val_iou, val_f1, val_roc_auc = val_metrics

            # Log metrics and handle checkpoint sonly on rank 0
            if rank == 0:
                log_memory_usage(rank)

                # Log metrics
                metrics = {
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc,
                }
                mlflow.log_metrics(metrics, step=epoch)

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

                # Save model checkpoint
                if epoch % SAVE_INTERVAL == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
                    early_stopping.save_checkpoint(
                        epoch, val_iou, model, optimizer, scheduler, checkpoint_path
                    )

                # Check early stopping
                early_stopping(epoch, val_iou, model, best_model_path)
                early_stop = early_stopping.early_stop
            else:
                early_stop = False

            # Broadcast early stopping decision to all processes
            early_stop_tensor = torch.tensor(
                1 if early_stop else 0, device=f"cuda:{rank}"
            )
            dist.broadcast(early_stop_tensor, src=0)
            if early_stop_tensor.item():
                if rank == 0:
                    logger.info("Early stopping triggered.")
                break

        # Evaluate best model on test set
        dist.barrier()  # Sync processes before loading best model

        # Broadcast whether we need to load best model to all processes
        if rank == 0:
            best_model_exists = os.path.exists(best_model_path)
        else:
            best_model_exists = False
        best_model_exists = torch.tensor(best_model_exists, device=f"cuda:{rank}")
        dist.broadcast(best_model_exists, src=0)

        if best_model_exists:
            # All processes load the best model
            best_model_checkpoint = torch.load(best_model_path)
            model.module.load_state_dict(best_model_checkpoint["model_state_dict"])

            # Evaluate on test set (all processes participate)
            test_loss, test_iou, test_f1, test_roc_auc = evaluate(
                model, test_dataloader, f"cuda:{rank}"
            )

            # Synchronize test metrics across processes
            test_metrics = torch.tensor(
                [test_loss, test_iou, test_f1, test_roc_auc], device=f"cuda:{rank}"
            )
            dist.all_reduce(test_metrics)
            test_loss, test_iou, test_f1, test_roc_auc = (
                x.item() / world_size for x in test_metrics
            )

            # Only rank 0 logs results
            if rank == 0:
                logger.info(
                    "Final results: Best model test results - Loss: %.4f, IoU: %.4f, F1-Score: %.4f, ROC-AUC: %.4f",
                    test_loss,
                    test_iou,
                    test_f1,
                    test_roc_auc,
                )
                # Close MLflow run
                mlflow.end_run()

            # Make sure all processes synchronize before finishing
            dist.barrier()
            logger.info(f"Rank %d: Model fine-tuning complete.", rank)

    except Exception as ex:
        logger.error(f"Rank %d: Error occured: %s", rank, str(ex))
        raise ex
    finally:
        cleanup()  # DDP cleanup


if __name__ == "__main__":
    logger.info("Starting the distributed training script...")
    world_size = torch.cuda.device_count()
    if world_size > 0:
        logger.info("World size: %d", world_size)
        try:
            mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
        except Exception as ex:
            logger.error(f"Error in main process: {str(ex)}")
            raise ex
    else:
        logger.error("No CUDA devices available. Exiting...")
