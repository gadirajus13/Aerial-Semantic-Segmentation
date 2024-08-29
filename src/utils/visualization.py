import numpy as np
import matplotlib.pyplot as plt

def create_color_map(class_dict):
    color_map = {}
    for _, row in class_dict.iterrows():
        class_name = row['name']
        rgb = [row[' r'] / 255.0, row[' g'] / 255.0, row[' b'] / 255.0]
        color_map[class_name] = rgb
    return color_map

def apply_color_map(mask, color_map, class_dict):
    rgb_mask = np.zeros((*mask.shape, 3))
    for class_name, color in color_map.items():
        class_index = class_dict[class_dict['name'] == class_name].index[0]
        rgb_mask[mask == class_index] = color
    return rgb_mask

def plot_results(images, true_masks, pred_masks, n_classes, color_map, class_dict, batch_size=3):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(3):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        
        true_mask = true_masks[i].cpu().numpy()
        true_mask_rgb = apply_color_map(true_mask, color_map, class_dict)
        axes[i, 1].imshow(true_mask_rgb)
        axes[i, 1].set_title('True Mask')
        
        pred_mask = pred_masks[i].cpu().numpy()
        pred_mask_rgb = apply_color_map(pred_mask, color_map, class_dict)
        axes[i, 2].imshow(pred_mask_rgb)
        axes[i, 2].set_title('Predicted Mask')
        
    plt.tight_layout()
    plt.show()

def visualize_color_map(color_map, class_dict):
    num_classes = len(color_map)
    fig, ax = plt.subplots(figsize=(12, num_classes * 0.5))
    
    for i, (class_name, color) in enumerate(color_map.items()):
        rect = plt.Rectangle((0, i), 1, 1, facecolor=color)
        ax.add_patch(rect)
        ax.text(1.1, i + 0.5, class_name, va='center')
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, num_classes)
    ax.axis('off')
    plt.title('Color Map Verification')
    plt.tight_layout()
    plt.show()