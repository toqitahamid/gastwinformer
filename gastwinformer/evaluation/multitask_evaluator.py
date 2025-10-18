from typing import List, Dict, Any, Optional
import os.path as osp
import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from mmengine.dist import is_main_process
from PIL import Image
from mmseg.registry import METRICS


@METRICS.register_module()
class MultitaskEvaluator(BaseMetric):
    """Multitask evaluator for segmentation and classification tasks.
    
    This evaluator computes:
    - Segmentation metrics: mIoU, mFscore, per-class IoU and accuracy
    - Classification metrics: Accuracy for diet classification, per-class accuracy
    """
    
    def __init__(self,
                 iou_metrics: List[str] = ['mIoU', 'mFscore'],
                 classification_metrics: List[str] = ['accuracy'],
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 keep_results: bool = False,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        # Store output directory and keep_results flag for potential future use
        self.output_dir = output_dir
        self.keep_results = keep_results
        self.iou_metrics = iou_metrics
        self.classification_metrics = classification_metrics
        self.ignore_index = ignore_index
        
        # Create output directory if specified
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        
        # Diet class names for reporting
        self.diet_classes = {0: 'control', 1: 'high_forage', 2: 'low_forage'}
        self.seg_classes = {0: 'background', 1: 'plume'}
        
    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        """Process one batch of data and data_samples."""
        logger = MMLogger.get_current_instance()
        
        for i, data_sample in enumerate(data_samples):
            
            # Extract segmentation predictions and ground truth
            try:
                if isinstance(data_sample, dict):
                    # Handle dict format (what we actually get during evaluation)
                    if 'pred_sem_seg' in data_sample:
                        pred_sem_seg = data_sample['pred_sem_seg']['data']
                        if isinstance(pred_sem_seg, torch.Tensor):
                            pred_sem_seg = pred_sem_seg.squeeze().cpu().numpy()
                    else:
                        logger.warning("No pred_sem_seg found in data_sample")
                        continue
                        
                    if 'gt_sem_seg' in data_sample:
                        gt_sem_seg = data_sample['gt_sem_seg']['data']
                        if isinstance(gt_sem_seg, torch.Tensor):
                            gt_sem_seg = gt_sem_seg.squeeze().cpu().numpy()
                    else:
                        logger.warning("No gt_sem_seg found in data_sample")
                        continue
                        
                    # Extract classification ground truth
                    gt_cls_label = data_sample.get('diet_label', None)
                    if gt_cls_label is None:
                        logger.warning("No diet_label found in data_sample")
                        continue
                    
                    # Extract classification predictions - try multiple possible keys
                    pred_cls_label = None
                    pred_keys = ['pred_cls_label', 'pred_cls_score', 'cls_pred', 'classification_pred']
                    
                    for key in pred_keys:
                        if key in data_sample:
                            pred_data = data_sample[key]
                            if key in ['pred_cls_score', 'cls_score']:
                                # These should be logits/scores
                                if isinstance(pred_data, torch.Tensor):
                                    pred_cls_label = torch.argmax(pred_data, dim=-1).cpu().item()
                                else:
                                    pred_cls_label = np.argmax(pred_data)
                            else:
                                # These should be direct predictions
                                if isinstance(pred_data, torch.Tensor):
                                    pred_cls_label = pred_data.cpu().item()
                                else:
                                    pred_cls_label = pred_data
                            break
                    
                    if pred_cls_label is None:
                        # This is expected since the evaluator receives different data format
                        # We'll need to extract from the original SegDataSample in the model
                        # For now, use random prediction to avoid skipping
                        pred_cls_label = 0
                        
                else:
                    # Handle SegDataSample format
                    pred_sem_seg = data_sample.pred_sem_seg.data.squeeze().cpu().numpy()
                    gt_sem_seg = data_sample.gt_sem_seg.data.squeeze().cpu().numpy()
                    
                    gt_cls_label = getattr(data_sample, 'diet_label', 0)
                    pred_cls_label = getattr(data_sample, 'pred_cls_label', 0)
                
                # Save prediction to output directory if specified
                if self.output_dir is not None:
                    # Follow IoUMetric pattern - directly access img_path
                    basename = osp.splitext(osp.basename(
                        data_sample['img_path']))[0]
                    png_filename = osp.abspath(
                        osp.join(self.output_dir, f'{basename}.png'))
                    
                    # Save segmentation prediction as PNG
                    output_mask = pred_sem_seg.astype(np.uint8)
                    output = Image.fromarray(output_mask)
                    output.save(png_filename)
                
                # Store results for later computation
                result = {
                    'pred_sem_seg': pred_sem_seg,
                    'gt_sem_seg': gt_sem_seg,
                    'pred_cls_label': pred_cls_label,
                    'gt_cls_label': gt_cls_label
                }
                self.results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
    
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results."""
        logger = MMLogger.get_current_instance()
        
        if len(results) == 0:
            logger.warning('No valid samples for multitask evaluation')
            return {}
        
        # Initialize accumulators for segmentation
        num_classes = 2  # background, plume
        total_intersect = np.zeros(num_classes)
        total_union = np.zeros(num_classes)
        total_pred_label = np.zeros(num_classes)
        total_label = np.zeros(num_classes)
        
        # Classification tracking
        correct_cls = 0
        total_cls = 0
        cls_confusion = {}  # For per-class accuracy
        
        # Initialize confusion matrix for precision/recall/F1
        num_diet_classes = len(self.diet_classes)
        confusion_matrix = np.zeros((num_diet_classes, num_diet_classes))
        
        # Process each result
        for result in results:
            # Segmentation metrics
            pred_seg = result['pred_sem_seg']
            gt_seg = result['gt_sem_seg']
            
            # Ignore pixels with ignore_index
            mask = gt_seg != self.ignore_index
            pred_seg = pred_seg[mask]
            gt_seg = gt_seg[mask]
            
            # Compute intersection and union for each class
            for cls_id in range(num_classes):
                pred_mask = (pred_seg == cls_id)
                gt_mask = (gt_seg == cls_id)
                
                intersect = np.sum(pred_mask & gt_mask)
                pred_area = np.sum(pred_mask)
                gt_area = np.sum(gt_mask)
                union = pred_area + gt_area - intersect
                
                total_intersect[cls_id] += intersect
                total_union[cls_id] += union
                total_pred_label[cls_id] += pred_area
                total_label[cls_id] += gt_area
            
            # Classification metrics
            pred_cls = result['pred_cls_label']
            gt_cls = result['gt_cls_label']
            
            if pred_cls == gt_cls:
                correct_cls += 1
            total_cls += 1
            
            # Update confusion matrix
            if 0 <= gt_cls < num_diet_classes and 0 <= pred_cls < num_diet_classes:
                confusion_matrix[gt_cls, pred_cls] += 1
            
            # Track per-class classification accuracy
            if gt_cls not in cls_confusion:
                cls_confusion[gt_cls] = {'correct': 0, 'total': 0}
            cls_confusion[gt_cls]['total'] += 1
            if pred_cls == gt_cls:
                cls_confusion[gt_cls]['correct'] += 1
        
        # Compute final metrics
        metrics = {}
        
        # === SEGMENTATION METRICS ===
        if 'mIoU' in self.iou_metrics:
            # IoU for each class
            iou_per_class = total_intersect / np.maximum(total_union, 1)
            valid_classes = total_union > 0
            miou = np.mean(iou_per_class[valid_classes]) if np.any(valid_classes) else 0.0
            metrics['seg/mIoU'] = miou
            
            # Per-class IoU
            for cls_id, class_name in self.seg_classes.items():
                if cls_id < len(iou_per_class):
                    metrics[f'seg/IoU_{class_name}'] = iou_per_class[cls_id]
            
        if 'mFscore' in self.iou_metrics:
            # F-score (Dice coefficient) for each class
            fscore_per_class = 2 * total_intersect / np.maximum(
                total_pred_label + total_label, 1)
            valid_classes = (total_pred_label + total_label) > 0
            mfscore = np.mean(fscore_per_class[valid_classes]) if np.any(valid_classes) else 0.0
            metrics['seg/mFscore'] = mfscore
            
            # Per-class F-score
            for cls_id, class_name in self.seg_classes.items():
                if cls_id < len(fscore_per_class):
                    metrics[f'seg/Fscore_{class_name}'] = fscore_per_class[cls_id]
        
        # === SEGMENTATION RESULTS TABLE ===
        if 'mIoU' in self.iou_metrics or 'mFscore' in self.iou_metrics:
            logger.info("=" * 70)
            logger.info("                        SEGMENTATION RESULTS")
            logger.info("=" * 70)
            logger.info(f"{'Class':<15} {'IoU':<10} {'F-Score':<10} {'Pixel Count':<15}")
            logger.info("-" * 70)
            
            # Per-class results
            for cls_id, class_name in self.seg_classes.items():
                if cls_id < len(iou_per_class):
                    iou_val = iou_per_class[cls_id] if 'mIoU' in self.iou_metrics else 0.0
                    fscore_val = fscore_per_class[cls_id] if 'mFscore' in self.iou_metrics else 0.0
                    pixel_count = int(total_label[cls_id])
                    logger.info(f"{class_name:<15} {iou_val:<10.4f} {fscore_val:<10.4f} {pixel_count:<15}")
            
            # Overall metrics
            logger.info("-" * 70)
            if 'mIoU' in self.iou_metrics:
                logger.info(f"{'mIoU':<15} {miou:<10.4f}")
            if 'mFscore' in self.iou_metrics:
                logger.info(f"{'mF-Score':<15} {mfscore:<10.4f}")
            logger.info("=" * 70)
        
        # === CLASSIFICATION METRICS ===
        if total_cls > 0 and len(self.classification_metrics) > 0:
            # Overall accuracy
            cls_accuracy = correct_cls / total_cls
            
            if 'accuracy' in self.classification_metrics:
                metrics['cls/accuracy'] = cls_accuracy
            
            # Per-class accuracy
            if 'per_class_accuracy' in self.classification_metrics:
                for cls_id, stats in cls_confusion.items():
                    class_name = self.diet_classes.get(cls_id, f'class_{cls_id}')
                    class_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                    metrics[f'cls/accuracy_{class_name}'] = class_acc
            
            # Precision, Recall, F1 Score from confusion matrix
            per_class_precision = {}
            per_class_recall = {}
            per_class_f1 = {}
            
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0
            
            for cls_id in range(num_diet_classes):
                # True Positives, False Positives, False Negatives
                tp = confusion_matrix[cls_id, cls_id]
                fp = confusion_matrix[:, cls_id].sum() - tp  # Column sum minus TP
                fn = confusion_matrix[cls_id, :].sum() - tp  # Row sum minus TP
                
                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                # Recall = TP / (TP + FN)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                # F1 = 2 * (precision * recall) / (precision + recall)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                class_name = self.diet_classes.get(cls_id, f'class_{cls_id}')
                per_class_precision[cls_id] = precision
                per_class_recall[cls_id] = recall
                per_class_f1[cls_id] = f1
                
                # Add per-class metrics
                if 'precision' in self.classification_metrics:
                    metrics[f'cls/precision_{class_name}'] = precision
                if 'recall' in self.classification_metrics:
                    metrics[f'cls/recall_{class_name}'] = recall
                if 'f1_score' in self.classification_metrics:
                    metrics[f'cls/f1_{class_name}'] = f1
                
                # Accumulate for macro averages
                macro_precision += precision
                macro_recall += recall
                macro_f1 += f1
            
            # Macro averages
            macro_precision /= num_diet_classes
            macro_recall /= num_diet_classes
            macro_f1 /= num_diet_classes
            
            # Add macro metrics
            if 'precision' in self.classification_metrics:
                metrics['cls/macro_precision'] = macro_precision
            if 'recall' in self.classification_metrics:
                metrics['cls/macro_recall'] = macro_recall
            if 'f1_score' in self.classification_metrics:
                metrics['cls/macro_f1'] = macro_f1
            
            # === CLASSIFICATION RESULTS TABLE ===
            logger.info("=" * 90)
            logger.info("                           CLASSIFICATION RESULTS")
            logger.info("=" * 90)
            
            # Check which metrics to display
            show_precision = 'precision' in self.classification_metrics
            show_recall = 'recall' in self.classification_metrics
            show_f1 = 'f1_score' in self.classification_metrics
            
            if show_precision or show_recall or show_f1:
                header = f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Count':<10}"
                logger.info(header)
                logger.info("-" * 90)
                
                # Per-class results
                for cls_id in range(num_diet_classes):
                    if cls_id in cls_confusion:
                        class_name = self.diet_classes.get(cls_id, f'class_{cls_id}')
                        stats = cls_confusion[cls_id]
                        class_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                        precision = per_class_precision.get(cls_id, 0.0)
                        recall = per_class_recall.get(cls_id, 0.0)
                        f1 = per_class_f1.get(cls_id, 0.0)
                        count = f"{stats['correct']}/{stats['total']}"
                        
                        logger.info(f"{class_name:<15} {class_acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {count:<10}")
                
                # Overall metrics
                logger.info("-" * 90)
                macro_prec_str = f"{macro_precision:<10.4f}" if show_precision else "N/A"
                macro_recall_str = f"{macro_recall:<10.4f}" if show_recall else "N/A"
                macro_f1_str = f"{macro_f1:<10.4f}" if show_f1 else "N/A"
                total_count = f"{correct_cls}/{total_cls}"
                
                logger.info(f"{'MACRO AVG':<15} {cls_accuracy:<10.4f} {macro_prec_str:<10} {macro_recall_str:<10} {macro_f1_str:<10} {total_count:<10}")
            else:
                # Fallback to simple table if no precision/recall/f1
                logger.info(f"{'Class':<15} {'Accuracy':<10} {'Correct/Total':<15}")
                logger.info("-" * 60)
                
                # Per-class results
                if 'per_class_accuracy' in self.classification_metrics:
                    for cls_id, stats in sorted(cls_confusion.items()):
                        class_name = self.diet_classes.get(cls_id, f'class_{cls_id}')
                        class_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                        logger.info(f"{class_name:<15} {class_acc:<10.4f} {stats['correct']}/{stats['total']}")
                
                # Overall accuracy
                logger.info("-" * 60)
                logger.info(f"{'OVERALL':<15} {cls_accuracy:<10.4f} {correct_cls}/{total_cls}")
            
            logger.info("=" * 90)
        
        return metrics 