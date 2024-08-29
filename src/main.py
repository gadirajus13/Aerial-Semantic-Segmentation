import config
import torch
import pandas as pd
import matplotlib.pyplot as plt
from models.yolo_encoder import YoloEncoder
from models.YoloUnet import UNetWithYoloEncoder
from data_processing.preprocessing import prepare_data
from utils.losses import CombinedLoss
from utils.visualization import create_color_map, plot_results, visualize_color_map
from train import train, EarlyStopping  
from test import test

def plot_metrics(train_losses, val_losses, train_iou, val_iou, train_acc, val_acc):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_iou, label='Training mIoU')
    plt.plot(val_iou, label='Validation mIoU')
    plt.title('mIoU over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    # Load class dictionary
    class_dict = pd.read_csv(config.CSV_PATH)

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        config.IMAGE_PATH, config.TARGET_PATH, 
        config.BATCH_SIZE, config.MEAN, config.STD
    )

    # Initialize model
    yolov5_model = torch.load('yolov5su.pt', map_location=config.DEVICE)['model']
    yolov5_model = yolov5_model.to(torch.float32)
    yolo_encoder = YoloEncoder(yolov5_model).to(config.DEVICE)
    model = UNetWithYoloEncoder(yolo_encoder, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Define loss function and optimizer
    criterion = CombinedLoss(weight=0.75)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

    # Create color map
    color_map = create_color_map(class_dict)

    # Visualize color map
    visualize_color_map(color_map, class_dict)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=8, verbose=True, path='best_model.pt')

    # Train the model
    train_losses, val_losses, train_iou, val_iou, train_acc, val_acc = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        config.NUM_EPOCHS, config.DEVICE, early_stopping
    )

    # Plot training metrics
    plot_metrics(train_losses, val_losses, train_iou, val_iou, train_acc, val_acc)

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Test the model
    test_loss, test_iou, test_accuracy = test(model, test_loader, criterion, config.DEVICE, color_map, class_dict)

    print(f"Final Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Visualize some test predictions
    model.eval()
    with torch.no_grad():
        inputs, masks = next(iter(test_loader))
        inputs, masks = inputs.to(config.DEVICE), masks.to(config.DEVICE)
        outputs = model(inputs)
        pred_masks = torch.argmax(outputs, dim=1)
        plot_results(inputs.cpu(), masks.cpu(), pred_masks.cpu(), config.NUM_CLASSES, color_map, class_dict)

if __name__ == "__main__":
    main()