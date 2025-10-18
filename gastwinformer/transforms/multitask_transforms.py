from typing import Dict, Sequence
import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS, METRICS
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmengine.evaluator import BaseMetric
import torch.nn.functional as F


@TRANSFORMS.register_module()
class LoadDietLabel(BaseTransform):
    """Load diet label from data_info.
    
    This transform loads the diet classification label from the data_info
    and adds it to the results dict for later processing.
    """
    
    def transform(self, results: Dict) -> Dict:
        """Transform function to load diet label.
        
        Args:
            results (dict): Result dict containing data info.
            
        Returns:
            dict: Updated results with diet_label.
        """
        if 'diet_label' not in results:
            raise KeyError("'diet_label' not found in data_info")
        
        results['diet_label'] = results['diet_label']
        return results


@TRANSFORMS.register_module()
class PackMultitaskInputs(BaseTransform):
    """Pack inputs for multitask learning.
    
    This transform packages the inputs and creates SegDataSample with both
    segmentation ground truth and diet classification labels.
    
    Args:
        meta_keys (Sequence[str]): Meta keys to be packed into meta_info.
    """
    
    def __init__(self, 
                 meta_keys: Sequence[str] = ('img_path', 'seg_map_path', 'ori_shape', 
                                           'img_shape', 'pad_shape', 'scale_factor', 
                                           'flip', 'flip_direction', 'diet_type')):
        self.meta_keys = meta_keys
    
    def transform(self, results: Dict) -> Dict:
        """Transform function to pack inputs.
        
        Args:
            results (dict): Result dict from previous transforms.
            
        Returns:
            dict: Packed results with inputs and data_samples.
        """
        # Create data sample
        data_sample = SegDataSample()
        
        # Pack segmentation ground truth
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            # Ensure gt_seg_map is proper format
            if isinstance(gt_seg_map, np.ndarray):
                gt_seg_map = torch.from_numpy(gt_seg_map.copy()).long()
            
            gt_sem_seg_data = PixelData(data=gt_seg_map)
            data_sample.gt_sem_seg = gt_sem_seg_data
        
        # Pack diet label
        if 'diet_label' in results:
            data_sample.diet_label = results['diet_label']
        
        # Pack meta information
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        
        # Pack final results
        packed_results = dict()
        packed_results['data_samples'] = data_sample
        
        # Pack inputs
        if 'img' in results:
            img = results['img']
            if isinstance(img, np.ndarray):
                # Convert HWC to CHW format if needed
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img.copy()).float()
            packed_results['inputs'] = img
        
        # Preserve img_path at top level for evaluator compatibility
        if 'img_path' in results:
            packed_results['img_path'] = results['img_path']
        
        return packed_results


@TRANSFORMS.register_module()
class MultitaskResize(BaseTransform):
    """Resize images and segmentation maps for multitask learning."""
    
    def __init__(self, scale, keep_ratio=True):
        self.scale = scale
        self.keep_ratio = keep_ratio
    
    def transform(self, results):
        from mmcv.transforms import Resize
        resize_transform = Resize(scale=self.scale, keep_ratio=self.keep_ratio)
        return resize_transform(results)


# Register MMEngine's Accuracy metric in mmseg registry
try:
    from mmengine.evaluator.metrics import Accuracy
    METRICS.register_module(module=Accuracy, force=True)
except ImportError:
    pass

# Alternatively, if you need a custom accuracy metric for classification tasks
@METRICS.register_module()
class MultitaskAccuracy(BaseMetric):
    """Accuracy metric specifically for multitask learning classification tasks.
    
    This metric computes classification accuracy for the classification head
    in a multitask model.
    """
    
    default_prefix = 'cls'
    
    def __init__(self, topk=(1,), collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.topk = topk if isinstance(topk, (tuple, list)) else (topk,)
    
    def process(self, data_batch, data_samples):
        """Process one batch of data and predictions."""
        for data_sample in data_samples:
            pred = data_sample.get('pred_classification', None)
            gt = data_sample.get('gt_classification', None)
            
            if pred is not None and gt is not None:
                # Handle both logits and probabilities
                if pred.dim() > 1 and pred.size(1) > 1:
                    # Multi-class case
                    pred_label = pred.argmax(dim=1)
                else:
                    # Binary case
                    pred_label = (pred > 0.5).long().squeeze()
                
                result = {
                    'pred_label': pred_label.cpu(),
                    'gt_label': gt.cpu()
                }
                self.results.append(result)
    
    def compute_metrics(self, results):
        """Compute accuracy metrics from processed results."""
        if not results:
            return {'accuracy': 0.0}
        
        # Concatenate all predictions and ground truth
        pred_labels = torch.cat([res['pred_label'] for res in results])
        gt_labels = torch.cat([res['gt_label'] for res in results])
        
        # Compute accuracy
        correct = (pred_labels == gt_labels).float()
        accuracy = correct.mean().item() * 100.0
        
        metrics = {'accuracy': accuracy}
        
        # Add top-k accuracy if specified
        if len(self.topk) > 1:
            for k in self.topk:
                if k == 1:
                    continue
                # For top-k accuracy with k>1, we'd need the full prediction scores
                # This is a simplified version
                metrics[f'top{k}_accuracy'] = accuracy
        
        return metrics