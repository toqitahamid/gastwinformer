from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch
from mmseg.structures import SegDataSample


@MODELS.register_module()
class MultitaskDataPreProcessor(BaseDataPreprocessor):
    """Data preprocessor for multitask learning using BaseDataPreprocessor.
    
    This version extends BaseDataPreprocessor directly to avoid parameter conflicts
    while still providing segmentation preprocessing capabilities.
    """
    
    def __init__(
        self,
        mean: Optional[Sequence[Number]] = None,
        std: Optional[Sequence[Number]] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        preserve_cls_labels: bool = True,
        test_cfg: Optional[dict] = None,
    ):
        super().__init__()
        
        # Store parameters
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.preserve_cls_labels = preserve_cls_labels
        self.test_cfg = test_cfg
        
        # Color space conversion
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb
        
        # Normalization setup
        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
    
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform preprocessing for multitask learning."""
        # Cast data to appropriate device
        data = self.cast_data(data)
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        
        # Extract classification labels before preprocessing
        cls_labels = []
        if self.preserve_cls_labels and data_samples is not None:
            cls_labels = self._extract_cls_labels_from_samples(data_samples)
        
        # Color space conversion
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        
        # Convert to float
        inputs = [_input.float() for _input in inputs]
        
        # Normalization
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]
        
        # Batching and padding
        if training:
            assert data_samples is not None, ('During training, ',
                                            '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)
        else:
            # Test mode
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)
        
        # Restore classification labels
        if self.preserve_cls_labels and data_samples is not None:
            if isinstance(data_samples, list):
                self._restore_cls_labels_to_samples(data_samples, cls_labels)
            else:
                # Single sample case
                single_sample_list = [data_samples]
                self._restore_cls_labels_to_samples(single_sample_list, cls_labels)
                data_samples = single_sample_list[0]
        
        return dict(inputs=inputs, data_samples=data_samples)
    
    def _extract_cls_labels_from_samples(self, data_samples: List) -> List[int]:
        """Extract classification labels from data samples."""
        cls_labels = []
        
        for i, sample in enumerate(data_samples):
            cls_label = None
            
            if isinstance(sample, SegDataSample):
                # Handle SegDataSample objects
                if hasattr(sample, 'diet_label'):
                    cls_label = sample.diet_label
                else:
                    # Check if it's in metainfo
                    metainfo = getattr(sample, 'metainfo', {})
                    cls_label = metainfo.get('diet_label', None)
                    
            elif isinstance(sample, dict):
                # Handle dict objects
                cls_label = sample.get('diet_label', None)
            
            # Handle missing labels
            if cls_label is None:
                import warnings
                warnings.warn(f"Missing diet_label in sample {i}, using default class 0")
                cls_label = 0
            
            # Validate label range (assuming 3 classes: 0, 1, 2)
            if not isinstance(cls_label, (int, float)) or cls_label < 0 or cls_label > 2:
                import warnings
                warnings.warn(f"Invalid diet_label {cls_label} in sample {i}, using default class 0")
                cls_label = 0
            
            cls_labels.append(int(cls_label))
        
        return cls_labels
    
    def _restore_cls_labels_to_samples(self, 
                                     data_samples: List, 
                                     cls_labels: List[int]) -> List:
        """Restore classification labels to processed data samples."""
        if not cls_labels or not data_samples:
            return data_samples
        
        # Handle length mismatch
        if len(cls_labels) != len(data_samples):
            import warnings
            warnings.warn(f"Length mismatch: {len(cls_labels)} labels vs {len(data_samples)} samples")
        
        # Assign labels to samples
        for i, sample in enumerate(data_samples):
            # Use corresponding label or last available label for extra samples
            label_idx = min(i, len(cls_labels) - 1) if cls_labels else 0
            cls_label = cls_labels[label_idx] if cls_labels else 0
            
            if isinstance(sample, SegDataSample):
                sample.diet_label = cls_label
            elif isinstance(sample, dict):
                sample['diet_label'] = cls_label
        
        return data_samples 