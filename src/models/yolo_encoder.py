import torch
import torch.nn as nn

class YoloEncoder(nn.Module):
    def __init__(self, yolov5_model):
        super(YoloEncoder, self).__init__()
        self.yolov5_model = yolov5_model
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.yolov5_model.model):
            if isinstance(layer, nn.Module):
                if isinstance(layer, (nn.Sequential, nn.ModuleList)):
                    for sub_layer in layer:
                        x = sub_layer(x)
                elif hasattr(layer, 'forward'):
                    if 'Concat' in layer.__class__.__name__:
                        x = layer([x, features[-1]])
                    else:
                        x = layer(x)
                features.append(x)
            if i == 23:
                break
        return x, features