from typing import Optional, Dict, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .base_head import BaseHead


@MODELS.register_module()
class SimpleClsHead(BaseHead):
    """Simple classification head with single hidden layer.
    
    Architecture: features → global_pool → flatten → linear → relu → dropout → linear
    """
    
    def __init__(self,
                 num_classes: int,
                 in_index: Union[int, Sequence[int]] = -1,
                 in_channels: Union[int, Sequence[int]] = None,
                 hidden_channels: int = 512,
                 dropout_rate: float = 0.5,
                 **kwargs):
        
        super().__init__(num_classes, in_index, in_channels, **kwargs)
        
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        
        # Build simple classifier
        input_channels = sum(self.in_channels_list)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, features):
        """Forward pass."""
        x = self._transform_inputs(features)
        return self.classifier(x) 