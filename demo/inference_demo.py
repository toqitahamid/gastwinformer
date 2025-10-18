#!/usr/bin/env python
"""
GasTwinFormer Inference Demo

This script demonstrates how to use GasTwinFormer for inference on gas leak images.

Usage:
    python demo/inference_demo.py --config CONFIG_FILE --checkpoint CHECKPOINT_FILE \
        --image IMAGE_FILE [--device DEVICE] [--output OUTPUT_DIR]

Example:
    python demo/inference_demo.py \
        --config gastwinformer/configs/gastwinformer_80k.py \
        --checkpoint checkpoints/gastwinformer_80k.pth \
        --image demo/sample_image.jpg \
        --device cuda:0 \
        --output demo/results/
"""

import argparse
import os
import os.path as osp
import warnings

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model
from mmseg.utils import register_all_modules

import gastwinformer  # Register gastwinformer modules


def parse_args():
    parser = argparse.ArgumentParser(description='GasTwinFormer inference demo')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--image', required=True, help='Input image file or directory')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--output', default='demo/results', help='Output directory for results')
    parser.add_argument('--show', action='store_true', help='Show results')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of segmentation overlay')
    args = parser.parse_args()
    return args


def visualize_result(image, result, output_path, opacity=0.5, palette=None):
    """Visualize segmentation result on the input image.
    
    Args:
        image (np.ndarray): Input image (H, W, 3) in RGB format
        result (dict): Prediction result containing 'pred_sem_seg'
        output_path (str): Path to save the visualization
        opacity (float): Opacity of the segmentation overlay
        palette (list): Color palette for different classes
    """
    if palette is None:
        # Default palette: background (black), plume (red)
        palette = [[0, 0, 0], [255, 0, 0]]
    
    # Extract prediction
    if hasattr(result, 'pred_sem_seg'):
        pred_mask = result.pred_sem_seg.data.cpu().numpy()
    else:
        pred_mask = result['pred_sem_seg']['data'].cpu().numpy()
    
    # Remove channel and batch dimensions if present
    if pred_mask.ndim == 4:
        pred_mask = pred_mask[0, 0]
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask[0]
    
    # Create colored mask
    h, w = pred_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        colored_mask[pred_mask == class_id] = color
    
    # Resize if needed
    if image.shape[:2] != pred_mask.shape:
        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1 - opacity, colored_mask, opacity, 0)
    
    # Create side-by-side visualization
    vis = np.hstack([image, overlay, colored_mask])
    
    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output_path}")
    
    return vis


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    print(f"Loading model from {args.config} and {args.checkpoint}")
    model = init_model(args.config, args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    
    # Get image paths
    if osp.isdir(args.image):
        # Process all images in directory
        image_files = [
            osp.join(args.image, f) for f in os.listdir(args.image)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        print(f"Found {len(image_files)} images in {args.image}")
    else:
        # Process single image
        image_files = [args.image]
    
    # Process each image
    for img_path in image_files:
        print(f"\nProcessing {img_path}...")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        result = inference_model(model, img_path)
        
        # Extract classification result if available
        if hasattr(result, 'pred_cls_label'):
            diet_classes = ['control', 'high_forage', 'low_forage']
            pred_label = result.pred_cls_label
            if isinstance(pred_label, torch.Tensor):
                pred_label = pred_label.item()
            print(f"  Classification: {diet_classes[pred_label]}")
            
            if hasattr(result, 'pred_cls_score'):
                scores = result.pred_cls_score
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                print(f"  Confidence scores: {dict(zip(diet_classes, scores))}")
        
        # Visualize and save
        output_name = osp.splitext(osp.basename(img_path))[0] + '_result.png'
        output_path = osp.join(args.output, output_name)
        vis = visualize_result(image_rgb, result, output_path, opacity=args.opacity)
        
        if args.show:
            cv2.imshow('GASTwinFormer Result', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
    
    if args.show:
        cv2.destroyAllWindows()
    
    print(f"\nAll results saved to {args.output}")


if __name__ == '__main__':
    main()

