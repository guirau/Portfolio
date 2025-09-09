"""Model configuration."""

# CLIP model and fine-tuned weights
MODEL_NAME = "ViT-B/32"
MODEL_PATH = "oma_recipeclassifier/src/models/classifier_best_model_ex07.pt"

# Classification classes
CLASSES = [
    "CP",
    "chopping-board",
    "finalstep",
    "glass-bowl-large",
    "glass-bowl-medium",
    "glass-bowl-small",
    "grill-plate",
    "group_step",
    "medium",
    "oven-dish",
    "oven-tray",
    "pan",
    "pot-one-handle",
    "pot-two-handles-medium",
    "pot-two-handles-shallow",
    "pot-two-handles-small",
    "saucepan",
]

# AWS model bucket path
MODEL_BUCKET_PATH = "recipeclassifier/models"
