# Aerial-Semantic-Segmentation
A custom YOLO_UNet approach for conducting semantic segmentation from Drone images for 22 distinct classes

<ANTARTIFACTLINK identifier="aerial-segmentation-readme-md" type="text/markdown" title="Markdown-formatted README for Aerial Segmentation Project" isClosed="true" />

Aerial Segmentation using Custom YoloUNet Architecture
This project implements a custom YoloUNet architecture for aerial image segmentation, combining a YOLOv5 encoder with a U-Net-style decoder to perform semantic segmentation on aerial imagery.
Table of Contents

Project Overview
Requirements
Setup
Dataset
Model Architecture
Training Process
Evaluation
Usage
Results

Project Overview
This project aims to perform semantic segmentation on aerial imagery using a custom deep learning architecture. The model combines the feature extraction capabilities of YOLOv5 with the precise localization of U-Net to achieve accurate segmentation results.
Requirements

Python 3.7+
PyTorch
torchvision
numpy
pandas
matplotlib
opencv-python (cv2)
albumentations
ultralytics (for YOLOv5)

Setup

Clone the repository:
bashCopygit clone https://github.com/your-username/aerial-segmentation.git
cd aerial-segmentation

Install the required packages:
bashCopypip install -r requirements.txt

Download the YOLOv5 weights:
bashCopywget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt


Dataset
The project uses the Semantic Drone Dataset. Organize your data as follows:
Copyinput_data/
├── dataset/
│   └── semantic_drone_dataset/
│       ├── original_images/
│       └── label_images_semantic/
└── class_dict_seg.csv
Model Architecture
The custom YoloUNet architecture consists of two main components:

YOLOv5 Encoder:

Utilizes the YOLOv5 model as a feature extractor
Extracts multi-scale features from the input images
Provides rich semantic information due to its pretraining on object detection tasks


Custom U-Net Decoder:

Consists of a series of upsampling blocks (Up modules)
Each Up module includes:

Bilinear upsampling
Concatenation with skip connections from the encoder
Two convolutional layers with batch normalization and ReLU activation


Gradually increases spatial resolution while decreasing channel depth
Incorporates skip connections to preserve fine-grained details



Key components of the architecture:

YoloEncoder: Wraps the YOLOv5 model and extracts features at different scales
UNetWithYoloEncoder: Combines the YOLO encoder with the U-Net decoder
Up: Custom module for upsampling and merging features
OutConv: Final convolutional layer to produce the segmentation map

The architecture is designed to leverage the strengths of both YOLO (efficient feature extraction) and U-Net (precise localization), making it well-suited for aerial image segmentation tasks.
Training Process
The training process involves several key components:

Data Preparation:

Custom AerialDataset class for loading and preprocessing images and masks
Data augmentation using Albumentations library (e.g., flips, rotations, color jittering)


Loss Function:

Combined loss using Dice Loss and Cross-Entropy Loss
Dice Loss helps with class imbalance issues
Cross-Entropy Loss provides stable gradients


Optimization:

Adam optimizer with initial learning rate of 0.001
Learning rate scheduling using ReduceLROnPlateau
Gradient clipping to prevent exploding gradients


Training Loop:

Mixed precision training using torch.cuda.amp for faster training and lower memory usage
Validation after each epoch to monitor performance
Early stopping to prevent overfitting
Model checkpointing to save the best model based on validation loss


Monitoring and Visualization:

Tracking of training/validation loss, accuracy, and mIoU
Periodic visualization of segmentation results during training



The training process is designed to be efficient and effective, with mechanisms in place to handle the challenges specific to aerial image segmentation, such as class imbalance and the need for precise boundaries.
Evaluation
The model is evaluated using:

Mean Intersection over Union (mIoU)
Pixel Accuracy
Combined Loss (Dice Loss + Cross-Entropy Loss)

Usage
To use the trained model for inference:

Load the trained model:
pythonCopymodel = UNetWithYoloEncoder(yolo_encoder, n_classes=num_classes)
model.load_state_dict(torch.load('YoloUnetV7.pt'))
model.eval()

Prepare your input image:
pythonCopyimage = cv2.imread('path_to_your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Apply necessary preprocessing

Run inference:
pythonCopywith torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

Visualize the results using the plot_results function provided in the notebook.

Results
The model's performance can be assessed using the plots generated after training, showing:

Training and Validation Loss
Training and Validation Accuracy
Training and Validation mIoU
