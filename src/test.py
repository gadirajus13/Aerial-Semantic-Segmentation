import torch
from utils.metrics import pixel_accuracy, mIoU
from utils.visualization import plot_results

def test(model, test_loader, criterion, device, color_map, class_dict):
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            test_loss += loss.item()
            test_iou += mIoU(outputs, masks)
            test_accuracy += pixel_accuracy(outputs, masks)

            # Visualize results
            pred_masks = torch.argmax(outputs, dim=1)
            plot_results(inputs.cpu(), masks.cpu(), pred_masks.cpu(), len(class_dict), color_map, class_dict)

    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_accuracy /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_iou, test_accuracy