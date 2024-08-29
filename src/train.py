import torch
import time
from torch.cuda.amp import autocast, GradScaler
from utils.metrics import pixel_accuracy, mIoU
from utils.visualization import plot_results

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping):
    scaler = GradScaler()
    train_losses = []
    val_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []

    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()  
        train_loss = 0.0
        iou_score = 0
        train_accuracy = 0

        for inputs, masks in train_loader:
            images, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                predictions = model(images)
                loss = criterion(predictions, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            with torch.no_grad():
                iou_score += mIoU(predictions, masks)
                train_accuracy += pixel_accuracy(predictions, masks)        

        epoch_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0
        val_iou_score = 0
        with torch.no_grad():
            for val_inputs, val_masks in val_loader:
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)

                with autocast():
                    val_predictions = model(val_inputs)
                    loss = criterion(val_predictions, val_masks)

                val_iou_score += mIoU(val_predictions, val_masks)
                val_accuracy += pixel_accuracy(val_predictions, val_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_iou.append(val_iou_score / len(val_loader))
        train_iou.append(iou_score / len(train_loader))
        train_acc.append(train_accuracy / len(train_loader))
        val_acc.append(val_accuracy / len(val_loader))

        train_losses.append(epoch_loss)
        epoch_time = time.time() - epoch_start_time
        epoch_minutes = int(epoch_time // 60)
        epoch_seconds = int(epoch_time % 60)

        print("\nEpoch:{}/{} |".format(epoch+1, num_epochs),
              "Train Loss: {:.3f} |".format(epoch_loss),
              "Val Loss: {:.3f} |".format(avg_val_loss),
              "Train mIoU:{:.3f} |".format(iou_score/len(train_loader)),
              "Val mIoU: {:.3f} |".format(val_iou_score/len(val_loader)),
              "Train Acc:{:.3f} |".format(train_accuracy/len(train_loader)),
              "Val Acc:{:.3f} |".format(val_accuracy/len(val_loader)),
              "Time: {:02d}:{:02d} mins".format(epoch_minutes, epoch_seconds))

        # Check Gradient Norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(avg_val_loss)
        print(f'Learning Rate: {scheduler.get_last_lr()[0]}')

    return train_losses, val_losses, train_iou, val_iou, train_acc, val_acc