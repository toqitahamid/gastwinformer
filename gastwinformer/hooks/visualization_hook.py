import os
import os.path as osp
import warnings
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class FlexibleSegVisualizationHook(Hook):
    """Flexible Segmentation Visualization Hook that handles variable image sizes.
    
    This hook automatically resizes predictions to match original image dimensions
    before visualization, avoiding dimension mismatch errors. It also supports
    displaying classification results for multitask models.
    
    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        show_classification (bool): Whether to display classification results.
            Defaults to True.
        cls_text_position (str): Position for classification text. Options:
            'top-left', 'top-right', 'bottom-left', 'bottom-right'. 
            Defaults to 'top-left'.
        show_confidence (bool): Whether to show confidence scores for predictions.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            Defaults to None.
    """

    # Diet class names mapping
    DIET_CLASSES = ('control', 'high_forage', 'low_forage')

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 show_classification: bool = True,
                 cls_text_position: str = 'top-left',
                 show_confidence: bool = True,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        self.show_classification = show_classification
        self.cls_text_position = cls_text_position
        self.show_confidence = show_confidence
        
        # Validate text position
        valid_positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        if self.cls_text_position not in valid_positions:
            raise ValueError(f"cls_text_position must be one of {valid_positions}")
        
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def _get_text_position(self, image_shape, text_lines):
        """Calculate text position based on image shape and text content.
        
        Args:
            image_shape (tuple): (height, width) of the image
            text_lines (list): List of text lines to display
            
        Returns:
            tuple: (x, y) position for text
        """
        height, width = image_shape[:2]
        line_height = 25
        text_height = len(text_lines) * line_height
        
        if self.cls_text_position == 'top-left':
            return (10, 30)
        elif self.cls_text_position == 'top-right':
            return (width - 200, 30)
        elif self.cls_text_position == 'bottom-left':
            return (10, height - text_height - 10)
        else:  # bottom-right
            return (width - 200, height - text_height - 10)

    def _extract_classification_info(self, data_sample):
        """Extract classification information from data sample.
        
        Args:
            data_sample: SegDataSample containing classification predictions
            
        Returns:
            dict: Dictionary containing classification information
        """
        cls_info = {}
        
        # Extract ground truth
        if hasattr(data_sample, 'diet_label'):
            gt_label = data_sample.diet_label
            if isinstance(gt_label, torch.Tensor):
                gt_label = gt_label.item()
            cls_info['gt_label'] = gt_label
            cls_info['gt_name'] = self.DIET_CLASSES[gt_label]
        elif hasattr(data_sample, 'metainfo') and 'diet_type' in data_sample.metainfo:
            gt_name = data_sample.metainfo['diet_type']
            cls_info['gt_name'] = gt_name
            if gt_name in self.DIET_CLASSES:
                cls_info['gt_label'] = self.DIET_CLASSES.index(gt_name)
        
        # Extract predictions
        if hasattr(data_sample, 'pred_cls_label'):
            pred_label = data_sample.pred_cls_label
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.item()
            cls_info['pred_label'] = pred_label
            cls_info['pred_name'] = self.DIET_CLASSES[pred_label]
        
        if hasattr(data_sample, 'pred_cls_score'):
            pred_scores = data_sample.pred_cls_score
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            cls_info['pred_scores'] = pred_scores
            cls_info['pred_confidence'] = float(pred_scores.max())
        
        return cls_info

    def _add_classification_text(self, visualizer, image, cls_info):
        """Add classification text overlay to the image.
        
        Args:
            visualizer: The visualizer instance
            image: The image array
            cls_info: Dictionary containing classification information
        """
        if not cls_info:
            return
        
        text_lines = []
        
        # Add ground truth
        if 'gt_name' in cls_info:
            text_lines.append(f"GT: {cls_info['gt_name']}")
        
        # Add prediction
        if 'pred_name' in cls_info:
            pred_text = f"Pred: {cls_info['pred_name']}"
            if self.show_confidence and 'pred_confidence' in cls_info:
                pred_text += f" ({cls_info['pred_confidence']:.3f})"
            text_lines.append(pred_text)
        
        # Add detailed scores if available
        if self.show_confidence and 'pred_scores' in cls_info:
            scores_text = "Scores: "
            score_parts = []
            for i, score in enumerate(cls_info['pred_scores']):
                score_parts.append(f"{self.DIET_CLASSES[i][:4]}:{score:.2f}")
            scores_text += " | ".join(score_parts)
            text_lines.append(scores_text)
        
        if not text_lines:
            return
        
        # Get text position
        x, y = self._get_text_position(image.shape, text_lines)
        
        # Add text lines
        for i, text in enumerate(text_lines):
            current_y = y + (i * 25)
            
            # Choose color based on content
            if text.startswith("GT:"):
                color = (0, 255, 0)  # Green for ground truth
            elif text.startswith("Pred:"):
                color = (255, 0, 0)  # Red for prediction
            else:
                color = (255, 255, 255)  # White for scores
            
            # Add text with background for better visibility
            try:
                # Try to use the visualizer's draw_texts method if available
                if hasattr(visualizer, 'draw_texts'):
                    visualizer.draw_texts(
                        text, 
                        positions=torch.tensor([[x, current_y]]),
                        colors=color,
                        font_sizes=14,
                        bboxes=dict(facecolor='black', alpha=0.7, pad=3)
                    )
                else:
                    # Fallback: add as simple text overlay
                    # This assumes the visualizer has some text drawing capability
                    pass
            except Exception as e:
                # If text drawing fails, just skip it
                warnings.warn(f"Could not add classification text: {e}")

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[SegDataSample]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for i, data_sample in enumerate(outputs):
                try:
                    # Handle different data formats
                    if isinstance(data_sample, str):
                        # Skip if data_sample is just a string
                        continue
                    
                    # Get original image path from metainfo (correct way for SegDataSample)
                    img_path = None
                    if hasattr(data_sample, 'metainfo') and 'img_path' in data_sample.metainfo:
                        img_path = data_sample.metainfo['img_path']
                    
                    if img_path is None:
                        # Skip visualization if we can't find image path
                        continue
                    
                    # The img_path might already contain the full path or just be relative to data_root
                    # Check if img_path already contains the data_root prefix
                    
                    if not osp.isabs(img_path):
                        if img_path.startswith('../../../V5/'):
                            # img_path already contains the full relative path
                            full_img_path = img_path
                        else:
                            # img_path is relative to data_root, so combine them
                            data_root = '../../../V5'  # This should match the config
                            full_img_path = osp.join(data_root, img_path)
                    else:
                        full_img_path = img_path
                        
                    # Load original image
                    image = mmcv.imread(full_img_path, channel_order='rgb', backend_args=self.backend_args)
                    
                    # Get prediction and ground truth
                    pred_sem_seg = getattr(data_sample, 'pred_sem_seg', None)
                    gt_sem_seg = getattr(data_sample, 'gt_sem_seg', None)
                
                    # Create a copy of the data sample for visualization without modifying original
                    from copy import deepcopy
                    from mmengine.structures import PixelData
                    vis_data_sample = deepcopy(data_sample)
                    
                    # Handle dimension mismatch by resizing predictions to match original image
                    if pred_sem_seg is not None:
                        pred_data = pred_sem_seg.data
                        original_h, original_w = image.shape[:2]
                        pred_h, pred_w = pred_data.shape[-2:]
                        
                        # Resize prediction if dimensions don't match
                        if (pred_h, pred_w) != (original_h, original_w):
                            # Convert to tensor if needed and add batch dimension
                            if not isinstance(pred_data, torch.Tensor):
                                pred_data = torch.from_numpy(pred_data)
                            
                            if pred_data.dim() == 2:
                                pred_data = pred_data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                            elif pred_data.dim() == 3:
                                pred_data = pred_data.unsqueeze(0)  # Add batch dim
                            
                            # Resize using nearest neighbor to preserve integer labels
                            pred_data = F.interpolate(
                                pred_data.float(), 
                                size=(original_h, original_w), 
                                mode='nearest'
                            ).long()
                            
                            # Remove batch dimensions but keep channel dimension for PixelData
                            if pred_data.dim() == 4:  # [1, 1, H, W]
                                pred_data = pred_data.squeeze(0)  # Remove batch dim → [1, H, W]
                            elif pred_data.dim() == 3 and pred_data.shape[0] == 1:  # [1, H, W]
                                pass  # Already correct format
                            else:  # [H, W]
                                pred_data = pred_data.unsqueeze(0)  # Add channel dim → [1, H, W]
                            vis_data_sample.pred_sem_seg = PixelData(data=pred_data)
                    
                    # Handle ground truth resizing if needed
                    if gt_sem_seg is not None:
                        gt_data = gt_sem_seg.data
                        original_h, original_w = image.shape[:2]
                        gt_h, gt_w = gt_data.shape[-2:]
                        
                        if (gt_h, gt_w) != (original_h, original_w):
                            if not isinstance(gt_data, torch.Tensor):
                                gt_data = torch.from_numpy(gt_data)
                            
                            if gt_data.dim() == 2:
                                gt_data = gt_data.unsqueeze(0).unsqueeze(0)
                            elif gt_data.dim() == 3:
                                gt_data = gt_data.unsqueeze(0)
                            
                            gt_data = F.interpolate(
                                gt_data.float(), 
                                size=(original_h, original_w), 
                                mode='nearest'
                            ).long()
                            
                            # Remove batch dimensions but keep channel dimension for PixelData
                            if gt_data.dim() == 4:  # [1, 1, H, W]
                                gt_data = gt_data.squeeze(0)  # Remove batch dim → [1, H, W]
                            elif gt_data.dim() == 3 and gt_data.shape[0] == 1:  # [1, H, W]
                                pass  # Already correct format
                            else:  # [H, W]
                                gt_data = gt_data.unsqueeze(0)  # Add channel dim → [1, H, W]
                            vis_data_sample.gt_sem_seg = PixelData(data=gt_data)

                    # Extract classification information if enabled
                    cls_info = {}
                    if self.show_classification:
                        cls_info = self._extract_classification_info(data_sample)

                    # Now visualize with matching dimensions using the copied data sample
                    self._visualizer.add_datasample(
                        osp.basename(full_img_path) if isinstance(full_img_path, str) else f'{mode}_{batch_idx}',
                        image,
                        vis_data_sample,  # Use the copied sample with resized data
                        show=self.show,
                        wait_time=self.wait_time,
                        step=runner.iter)
                    
                    # Add classification text overlay if enabled and info available
                    if self.show_classification and cls_info:
                        self._add_classification_text(self._visualizer, image, cls_info)
                
                except Exception as e:
                    # Skip this sample if visualization fails
                    import warnings
                    warnings.warn(f"Skipping visualization for sample {i}: {e}")
                    continue

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every validation iteration."""
        self._after_iter(runner, batch_idx, data_batch, outputs, mode='val')

    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        """Run after every test iteration."""
        self._after_iter(runner, batch_idx, data_batch, outputs, mode='test') 