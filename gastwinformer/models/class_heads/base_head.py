from abc import ABCMeta, abstractmethod
from typing import List, Optional, Dict, Any, Union, Sequence
import torch
import torch.nn as nn
from mmengine.model import BaseModel


class BaseHead(BaseModel, metaclass=ABCMeta):
    """Base class for all heads in ConfigurableSegmentor architecture.
    
    This base class provides common functionality for heads that receive
    pre-processed features from the ConfigurableSegmentor's routing system.
    
    Unlike MMSeg's BaseDecodeHead which processes raw backbone features,
    this BaseHead works with features that have already been routed and
    potentially processed by necks.
    
    Args:
        num_classes (int): Number of output classes
        in_index (int|Sequence[int]): Input feature index. Default: -1
        in_channels (int|Sequence[int]): Input channels
        loss_cfg (dict, optional): Loss configuration. Default: CrossEntropyLoss
        init_cfg (dict, optional): Initialization configuration for head components
    """
    
    def __init__(self,
                 num_classes: int,
                 in_index: Union[int, Sequence[int]] = -1,
                 in_channels: Union[int, Sequence[int]] = None,
                 loss_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        
        # Default initialization for new classification head components
        # This only affects newly initialized layers, pretrained weights are preserved
        if init_cfg is None:
            init_cfg = dict(
                type='Xavier', 
                layer=['Linear'],
                override=dict(
                    type='Constant',
                    layer='BatchNorm1d', 
                    val=1,
                    bias=0
                )
            )
        
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        
        # Initialize input configuration
        self._init_inputs(in_index, in_channels)
        
        # Default loss configuration
        if loss_cfg is None:
            loss_cfg = dict(type='CrossEntropyLoss', loss_weight=1.0)
        
        # Build loss function
        if loss_cfg['type'] == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_cfg['type'] == 'MSELoss':
            self.loss_fn = nn.MSELoss()
        elif loss_cfg['type'] == 'BCEWithLogitsLoss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            # For more complex losses, you could use MODELS.build(loss_cfg)
            raise ValueError(f"Unsupported loss type: {loss_cfg['type']}")
        
        self.loss_weight = loss_cfg.get('loss_weight', 1.0)
    
    def _init_inputs(self, in_index: Union[int, Sequence[int]], in_channels: Union[int, Sequence[int]]):
        """Initialize input configuration similar to BaseDecodeHead._init_inputs.
        
        Args:
            in_index (int|Sequence[int]): Input feature index
            in_channels (int|Sequence[int]): Input channels
        """
        if in_channels is None:
            raise ValueError("in_channels must be provided")
        
        # Normalize inputs to lists for internal processing
        if isinstance(in_index, int):
            self.feature_indices = [in_index]
        else:
            self.feature_indices = list(in_index)
            
        if isinstance(in_channels, int):
            self.in_channels_list = [in_channels]
        else:
            self.in_channels_list = list(in_channels)
        
        # Store original format for external access (like mmseg does)
        self.in_index = in_index
        self.in_channels = in_channels
        
        # Validate inputs
        if len(self.feature_indices) != len(self.in_channels_list):
            raise ValueError(f"in_index length ({len(self.feature_indices)}) must match "
                           f"in_channels length ({len(self.in_channels_list)})")
    
    def _transform_inputs(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Transform and select input features similar to BaseDecodeHead._transform_inputs.
        
        Args:
            features: Feature tensor or list of tensors from backbone
            
        Returns:
            torch.Tensor: Selected and potentially concatenated features
        """
        # Ensure features is a list
        if isinstance(features, torch.Tensor):
            features = [features]
        
        # Select features by indices
        selected_features = []
        for idx in self.feature_indices:
            if idx >= len(features):
                raise ValueError(f"Feature index {idx} is out of range. "
                               f"Got {len(features)} features, indices: {self.feature_indices}")
            
            feat = features[idx]
            selected_features.append(feat)
        
        # Concatenate all selected features if multiple
        if len(selected_features) == 1:
            return selected_features[0]
        else:
            return torch.cat(selected_features, dim=1)  # [B, C_total, H, W]
    
    @classmethod
    def from_backbone_config(cls,
                           num_classes: int,
                           backbone_channels: List[int],
                           in_index: Union[int, Sequence[int]] = -1,
                           **kwargs):
        """Create head from backbone channel configuration.
        
        Args:
            num_classes (int): Number of output classes
            backbone_channels (List[int]): All backbone output channels [64, 128, 320, 512]
            in_index (int|Sequence[int]): Feature index(es) to use. Default: -1 (last feature)
            **kwargs: Additional arguments for the head
            
        Returns:
            BaseHead: Configured head
            
        Examples:
            # Use last feature only (default)
            head = SomeHead.from_backbone_config(3, [64, 128, 320, 512])
            
            # Use specific single feature
            head = SomeHead.from_backbone_config(3, [64, 128, 320, 512], in_index=1)
            
            # Use multiple features
            head = SomeHead.from_backbone_config(3, [64, 128, 320, 512], 
                                               in_index=[0, 3])
        """
        # Normalize in_index to list
        if isinstance(in_index, int):
            indices = [in_index]
        else:
            indices = list(in_index)
        
        # Convert negative indices to positive
        num_features = len(backbone_channels)
        normalized_indices = []
        for idx in indices:
            if idx < 0:
                normalized_indices.append(num_features + idx)
            else:
                normalized_indices.append(idx)
        
        # Get channels for selected features
        selected_channels = [backbone_channels[i] for i in normalized_indices]
        
        # Return appropriate format based on input
        if len(normalized_indices) == 1:
            return cls(
                num_classes=num_classes,
                in_index=normalized_indices[0],
                in_channels=selected_channels[0],
                **kwargs
            )
        else:
            return cls(
                num_classes=num_classes,
                in_index=normalized_indices,
                in_channels=selected_channels,
                **kwargs
            )

    @abstractmethod
    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Forward pass of the head.
        
        Args:
            features: Pre-processed features from ConfigurableSegmentor routing
            
        Returns:
            torch.Tensor: Head output (logits, predictions, etc.)
        """
        pass
    
    def _extract_ground_truth_labels(self, data_samples: List[Any]) -> torch.Tensor:
        """Extract ground truth labels from data samples.
        
        This method handles various ways labels might be stored in data_samples.
        Subclasses can override this for task-specific label extraction.
        
        Args:
            data_samples: List of data samples containing ground truth
            
        Returns:
            torch.Tensor: Ground truth labels as tensor
        """
        gt_labels = []
        
        for sample in data_samples:
            # Try different possible label attributes - prioritize diet_label for compatibility
            if hasattr(sample, 'diet_label'):
                gt_labels.append(sample.diet_label)
            elif hasattr(sample, 'gt_label'):
                gt_labels.append(sample.gt_label)
            elif hasattr(sample, 'gt_cls_label'):
                gt_labels.append(sample.gt_cls_label)
            elif hasattr(sample, 'gt_reg_target'):
                gt_labels.append(sample.gt_reg_target)
            elif hasattr(sample, 'metainfo') and 'diet_label' in sample.metainfo:
                gt_labels.append(sample.metainfo['diet_label'])
            elif hasattr(sample, 'metainfo') and 'gt_label' in sample.metainfo:
                gt_labels.append(sample.metainfo['gt_label'])
            elif hasattr(sample, 'metainfo') and 'gt_cls_label' in sample.metainfo:
                gt_labels.append(sample.metainfo['gt_cls_label'])
            else:
                raise ValueError(f"No ground truth labels found in data_samples. "
                               f"Available attributes: {dir(sample)}")
        
        # Convert to tensor - handle different data types
        if isinstance(gt_labels[0], torch.Tensor):
            return torch.stack(gt_labels, dim=0)
        else:
            return torch.tensor(gt_labels, dtype=torch.long)
    
    def loss(self, 
             features: Union[torch.Tensor, List[torch.Tensor]], 
             data_samples: List[Any], 
             train_cfg: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        """Calculate loss for the head.
        
        Args:
            features: Pre-processed features from ConfigurableSegmentor
            data_samples: List of data samples containing ground truth
            train_cfg: Training configuration (optional)
            
        Returns:
            Dict containing loss values
        """
        # Forward pass
        output = self.forward(features)
        
        # Extract ground truth
        gt_labels = self._extract_ground_truth_labels(data_samples)
        
        # Move to same device as output
        gt_labels = gt_labels.to(output.device)
        
        # Calculate loss - subclasses can override this
        loss_value = self._compute_loss(output, gt_labels)
        
        # Return loss dict with standardized name
        loss_name = self._get_loss_name()
        return {loss_name: loss_value * self.loss_weight}
    
    def _compute_loss(self, output: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """Compute the actual loss value.
        
        Subclasses can override this for custom loss computation.
        
        Args:
            output: Model output (logits, predictions, etc.)
            gt_labels: Ground truth labels
            
        Returns:
            torch.Tensor: Loss value
        """
        return self.loss_fn(output, gt_labels)
    
    def _get_loss_name(self) -> str:
        """Get the name for the loss in the returned dict.
        
        Subclasses can override this to customize loss naming.
        
        Returns:
            str: Loss name
        """
        return 'loss'
    
    def predict(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Generate predictions.
        
        Subclasses can override this for task-specific prediction formats.
        
        Args:
            features: Pre-processed features from ConfigurableSegmentor
            
        Returns:
            Dict containing predictions
        """
        # Forward pass
        output = self.forward(features)
        
        # Generate predictions - subclasses can override this
        return self._generate_predictions(output)
    
    def _generate_predictions(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions from model output.
        
        Default implementation assumes classification logits.
        Subclasses can override for task-specific prediction generation.
        
        Args:
            output: Model output
            
        Returns:
            Dict containing predictions
        """
        if output.dim() > 1 and output.shape[1] > 1:
            # Multi-class classification
            probs = torch.softmax(output, dim=1)
            pred_labels = torch.argmax(probs, dim=1)
            return {
                'logits': output,
                'probs': probs,
                'pred_labels': pred_labels
            }
        else:
            # Regression or binary classification
            return {
                'output': output,
                'predictions': output
            }
    
    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        in_index_str = self.feature_indices if len(self.feature_indices) > 1 else self.feature_indices[0]
        in_channels_str = self.in_channels_list if len(self.in_channels_list) > 1 else self.in_channels_list[0]
        return (f'num_classes={self.num_classes}, '
                f'in_index={in_index_str}, '
                f'in_channels={in_channels_str}, '
                f'loss_weight={self.loss_weight}') 