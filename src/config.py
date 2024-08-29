import os
import torch
# Paths
BASE_PATH = os.path.join(os.getcwd(), "input_data", "dataset", "semantic_drone_dataset")
IMAGE_PATH = os.path.join(BASE_PATH, "original_images")
TARGET_PATH = os.path.join(BASE_PATH, "label_images_semantic")
COLOR_TARGET_PATH = os.path.join(os.getcwd(), "input_data", "RGB_color_image_masks")
CSV_PATH = os.path.join(os.getcwd(), "input_data", "class_dict_seg.csv")

# Model parameters
NUM_CLASSES = 23
BATCH_SIZE = 5
NUM_EPOCHS = 75

# Training parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Data preprocessing (ImageNet Normalization)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"