# Aerial-Semantic-Segmentation

This project implements a custom YoloUNet architecture for aerial image segmentation, combining a YOLOv5 encoder with a U-Net-style decoder to perform semantic segmentation on aerial imagery for 22 distinct classes. The model achieves 86% pixel accuracy and a mIoU of 0.53 when trained on a dataset of only 280 images with validation done on 80 images and testing on 40 images.

## Project Overview

This project aims to perform semantic segmentation on aerial imagery using a custom deep learning architecture. The model combines the feature extraction capabilities of YOLOv5 with the precise localization of U-Net to achieve accurate segmentation results.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- opencv-python (cv2)
- albumentations
- ultralytics (for YOLOv5)

## Dataset
The project uses the Aerial Semantic Segmentation Drone Dataset from Kaggle available at:
https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

The dataset consists of 400 images with semantic labels for 22 different classes such as people, dogs, cars, trees, vegetation, etc.
The dataset was divided into train, validation, and tests sets on a 70,20,10 split with extensive data augmentation to prevent overfitting and promote learning of undersampled classes.


## Model Architecture
### YOLOv5 Encoder:
- Utilizes the YOLOv5 model as a feature extractor
- Extracts multi-scale features from the input images
- Provides rich semantic information due to its pretraining on object detection tasks

### Custom U-Net Decoder:
- Consists of a series of upsampling blocks (Up modules)
- Each Up module includes:
    - Bilinear upsampling
    - Concatenation with skip connections from the encoder
    - Two convolutional layers with batch normalization and ReLU activation
- Gradually increases spatial resolution while decreasing channel depth
- Incorporates skip connections to preserve fine-grained details

## Key Components:
- YoloEncoder: Wraps the YOLOv5 model and extracts features at different scales
- UNetWithYoloEncoder: Combines the YOLO encoder with the U-Net decoder
- Up: Custom module for upsampling and merging features with residuals
- OutConv: Final convolutional layer to produce the segmentation map

The architecture is designed to leverage the strengths of both YOLO (efficient feature extraction) and U-Net (precise localization), making it well-suited for aerial image segmentation tasks.

## Training Process
The training process involves several key components:

### Data Preparation:
- Custom AerialDataset class for loading and preprocessing images and masks
- Data augmentation using Albumentations library (e.g., flips, rotations, distortion, brightnes)

### Loss Function:
- Combined loss using Dice Loss and Cross-Entropy Loss to balance pixel accuracy and mIoU with a higher weight given to the Dice Loss 
- Dice Loss helps with class imbalance issues while Cross-Entropy Loss provides stable gradients and aids in focusing on overall accuracy

### Optimization:
- Adam optimizer with initial learning rate of 0.001
- Learning rate scheduling using ReduceLROnPlateau

### Training Loop:
- Mixed precision training using torch.cuda.amp for faster training and lower memory usage
- Validation after each epoch to monitor performance
- Early stopping to prevent overfitting
- Model checkpointing to save the best model based on validation loss

### Monitoring and Visualization:
- Tracking of training/validation loss, accuracy, and mIoU
- Periodic visualization of segmentation results during training

The training process is designed to be efficient and effective, with mechanisms in place to handle the challenges specific to aerial image segmentation, such as class imbalance and the need for precise boundaries.

## Evaluation Metrics
The model is evaluated using:
- Mean Intersection over Union (mIoU)
- Pixel Accuracy
- Combined Loss (Dice Loss + Cross-Entropy Loss)

### Mean Intersection over Union (mIoU):
- Provides a measure of overlap between predicted and ground truth segmentation masks, penalizing both over-segmentation and under-segmentation
- More sensitive to class imbalance compared to pixel accuracy, which can be beneficial for detecting poor performance on underrepresented classes
- Gives equal importance to all classes, regardless of their frequency in the dataset

### Pixel Accuracy:
- Represents the percentage of correctly classified pixels but can be misleading in cases of severe class imbalance which is prevalent in this dataset   
    
## Results
The model's performance can be assessed using the plots generated after training, showing the loss, accuracy, and mIoU. Furthermore, the model is then tested on the test dataset which consists of 40 images.

This model acheived a peak pixel accuracy of 86% with a mIoU of .53 on the test data which boasts a significant improvement compared to other approaches to this dataset. Utilizing the UNet Mobile architecture with pre-loaded ImageNet weights, as others have in the Kaggle link to the dataset, they achieve a similar accuracy of around 81%, however have a much lower mIoU of .32, indicating that this architecture provides more accurate masks per class on average compared to typical approaches. The majority of discrepency between the mIoU and pixel accuracy comes from the classification for under-sampled classes within the dataset, which means the model can be further improved with mroe data preprocessing.

Here are a few of the results obtained from running the model on the test data:
![Test 1](results/Test%20Results.png)
![Test 2](results/Test%20Results%203.png)

## Future Work
In order to further improve the model accuracy and mIoU, I would like to implement oversampling of classes that are underepresented in teh dataset currently as they have a heavy influence on the mIoU and will aid in further increasing the overall pixel accuracy.
