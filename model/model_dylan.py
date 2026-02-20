import torch
from torch import nn
import numpy as np

class FlexibleTNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        conv_channels: list = [25, 35, 50, 75, 125],
        fc_layers: list = [500, 250, 50],
        dropout_p: float = 0.2,
        do_sigmoid: bool = False,
        size_img: int = 128,
    ):
        
        super().__init__()
        self.do_sigmoid = do_sigmoid
 
        # Conv Layers
        conv_blocks = []
        in_ch = num_channels
        for out_ch in conv_channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            conv_blocks.append(block)
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*conv_blocks)

        # Final spatial sizye after conv and pool
        size_img = int(size_img // np.power(2, len(conv_channels)))

        # FC
        fc_blocks = []
        in_features = size_img * size_img * conv_channels[-1]

        for out_features in fc_layers:
            block = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(p=dropout_p),
                nn.ReLU(),
            )
            fc_blocks.append(block)
            in_features = out_features

        self.fc_layers = nn.Sequential(*fc_blocks)

        # Out
        self.output = nn.Linear(in_features, num_classes)
        self.final_act = nn.Sigmoid() if do_sigmoid else nn.Identity()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        x = self.output(x)
        x = self.final_act(x)
        return x
