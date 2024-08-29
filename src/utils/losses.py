import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, f"Predictions shape {predictions.shape} doesn't match targets shape {targets.shape}"
        
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weight=0.5, num_classes=23):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        pred_probs = F.softmax(predictions, dim=1)
        
        # Create one-hot encoded target
        targets_one_hot = F.one_hot(targets.squeeze(1).long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice loss for each class and average
        dice_loss = 0
        for i in range(self.num_classes):
            dice_loss += self.dice_loss(pred_probs[:, i], targets_one_hot[:, i])
        dice_loss /= self.num_classes

        ce = self.ce_loss(predictions, targets.squeeze(1).long())
        # print(f"Dice Loss: {dice_loss.item()}, CE Loss: {ce.item()}")
        return (self.weight * dice_loss) + ((1 - self.weight) * ce)