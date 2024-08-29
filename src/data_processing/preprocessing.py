import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import AerialDataset

def create_df(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

def prepare_data(image_path, mask_path, batch_size, mean, std):
    df = create_df(image_path)
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=42)

    train_set = AerialDataset(image_path, mask_path, X_train, mean, std, is_train=True)
    val_set = AerialDataset(image_path, mask_path, X_val, mean, std)
    test_set = AerialDataset(image_path, mask_path, X_test, mean, std)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    return train_loader, val_loader, test_loader