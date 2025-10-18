import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss


def _expand_onehot_labels_dice(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target, num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


@MODELS.register_module()
class GaussianPlumeWeightedDiceLoss(nn.Module):
    """
    Gaussian Plume Weighted Dice Loss for gas leak segmentation.

    Incorporates physical constraints from gas dispersion behavior using
    Gaussian plume model weights in the Dice loss calculation.

    Based on the paper: "High-accuracy combustible gas cloud imaging system
    using YOLO-plume classification network" (Frontiers in Physics, 2025)
    """

    def __init__(
        self,
        use_sigmoid=True,
        activate=True,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=255,
        eps=1e-6,
        gas_class_idx=1,
        loss_name="loss_gaussian_plume_dice",
    ):
        """Compute Gaussian plume weighted dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-6.
            gas_class_idx (int): Index of gas class in multi-class predictions.
                Defaults to 1.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_gaussian_plume_dice'.
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.eps = eps
        self.gas_class_idx = gas_class_idx
        self._loss_name = loss_name

    def gaussian_plume_weights(self, mask_shape, center_x, center_y, sigma_x, sigma_y):
        """
        Generate Gaussian plume weights for each pixel.

        Args:
            mask_shape: Tuple of (H, W) for mask dimensions
            center_x: X coordinate of plume center (μx)
            center_y: Y coordinate of plume center (μy)
            sigma_x: Horizontal diffusion scale (σx)
            sigma_y: Vertical diffusion scale (σy)

        Returns:
            Weight tensor of shape (H, W)
        """
        H, W = mask_shape

        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        x_coords = torch.arange(W, dtype=torch.float32).unsqueeze(0)

        # Move to same device as inputs
        device = center_x.device
        y_coords = y_coords.to(device)
        x_coords = x_coords.to(device)

        # Calculate Gaussian weights using Equation 2 from paper
        # w(x,y) = exp(-((x-μx)²/(2σx²) + (y-μy)²/(2σy²)))
        x_term = (x_coords - center_x) ** 2 / (2 * sigma_x**2)
        y_term = (y_coords - center_y) ** 2 / (2 * sigma_y**2)

        weights = torch.exp(-(x_term + y_term))

        return weights

    def estimate_plume_parameters(self, pred_mask):
        """
        Estimate plume center and spread parameters from predicted mask.
        Uses center of mass for location and standard deviation for spread.

        Args:
            pred_mask: Predicted segmentation mask (B, H, W) with values in [0, 1]

        Returns:
            Dictionary with center_x, center_y, sigma_x, sigma_y for each batch
        """
        B, H, W = pred_mask.shape
        device = pred_mask.device

        params = {
            "center_x": torch.zeros(B, device=device),
            "center_y": torch.zeros(B, device=device),
            "sigma_x": torch.ones(B, device=device) * W / 4,  # Default to 1/4 of width
            "sigma_y": torch.ones(B, device=device) * H / 4,  # Default to 1/4 of height
        }

        # Create coordinate grids
        y_coords = (
            torch.arange(H, dtype=torch.float32, device=device)
            .unsqueeze(1)
            .expand(H, W)
        )
        x_coords = (
            torch.arange(W, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .expand(H, W)
        )

        for b in range(B):
            mask = pred_mask[b]

            # Apply threshold to create binary mask for parameter estimation
            binary_mask = (mask > 0.5).float()

            # Skip if mask is empty
            total_mass = binary_mask.sum()
            if total_mass < 1.0:
                # Keep default values for empty masks
                continue

            # Calculate center of mass
            params["center_x"][b] = (binary_mask * x_coords).sum() / total_mass
            params["center_y"][b] = (binary_mask * y_coords).sum() / total_mass

            # Calculate spread (standard deviation) using continuous mask values
            cx, cy = params["center_x"][b], params["center_y"][b]

            # Use the continuous mask values for weighted standard deviation
            weighted_mass = mask.sum()
            if weighted_mass > 1e-6:
                params["sigma_x"][b] = torch.sqrt(
                    ((x_coords - cx) ** 2 * mask).sum() / weighted_mass
                )
                params["sigma_y"][b] = torch.sqrt(
                    ((y_coords - cy) ** 2 * mask).sum() / weighted_mass
                )

            # Ensure minimum sigma values to avoid too narrow Gaussians
            params["sigma_x"][b] = torch.clamp(
                params["sigma_x"][b], min=W / 20, max=W / 2
            )
            params["sigma_y"][b] = torch.clamp(
                params["sigma_y"][b], min=H / 20, max=H / 2
            )

        return params

    def gaussian_plume_dice_loss(self, pred, target, plume_params=None):
        """
        Calculate weighted Dice loss with Gaussian plume constraints.

        Args:
            pred: Predicted segmentation (B, H, W) with values in [0, 1]
            target: Ground truth masks (B, H, W)
            plume_params: Optional dict with plume parameters. If None, estimates from pred.

        Returns:
            Weighted Dice loss tensor for each sample in batch
        """
        B, H, W = pred.shape

        # Ensure pred and target are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        target = target.float()

        # Estimate plume parameters if not provided
        if plume_params is None:
            plume_params = self.estimate_plume_parameters(pred.detach())

        losses = []

        for b in range(B):
            # Get plume weights for this sample
            weights = self.gaussian_plume_weights(
                (H, W),
                plume_params["center_x"][b],
                plume_params["center_y"][b],
                plume_params["sigma_x"][b],
                plume_params["sigma_y"][b],
            )

            # Apply weights to predictions and targets
            pred_b = pred[b]
            target_b = target[b].float()

            # Weighted Dice loss calculation (Equation 4)
            # L_weighted = 1 - 2*Σ(w*y*ŷ) / (Σ(w*y) + Σ(w*ŷ))
            weighted_intersection = weights * pred_b * target_b
            weighted_pred = weights * pred_b
            weighted_target = weights * target_b

            numerator = 2.0 * weighted_intersection.sum()
            denominator = weighted_target.sum() + weighted_pred.sum()

            dice_score = (numerator + self.eps) / (denominator + self.eps)
            loss = 1.0 - dice_score

            losses.append(loss)

        return torch.stack(losses)

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=255,
        **kwargs,
    ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Defaults to 255.
            **kwargs: Additional keyword arguments. Can include:
                - plume_params (dict): Dictionary with plume parameters
                  containing 'center_x', 'center_y', 'sigma_x', 'sigma_y'

        Returns:
            torch.Tensor: The calculated loss
        """
        # Handle one-hot target expansion
        one_hot_target = target
        if pred.shape != target.shape:
            one_hot_target = _expand_onehot_labels_dice(pred, target)

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        # Apply activation if needed
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)

        # Handle different input formats for plume weighting
        if pred.dim() == 4 and pred.shape[1] > 1:
            # Multi-class case - take gas class
            pred_plume = pred[:, self.gas_class_idx]
            target_plume = one_hot_target[:, self.gas_class_idx]
        elif pred.dim() == 4 and pred.shape[1] == 1:
            # Single channel with batch
            pred_plume = pred.squeeze(1)
            target_plume = (
                one_hot_target.squeeze(1)
                if one_hot_target.dim() == 4
                else one_hot_target
            )
        elif pred.dim() == 3:
            # Already (B, H, W)
            pred_plume = pred
            target_plume = target
        else:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")

        # Get plume parameters from kwargs
        plume_params = kwargs.get("plume_params", None)

        # Calculate Gaussian plume weighted dice loss
        loss = self.gaussian_plume_dice_loss(pred_plume, target_plume, plume_params)

        # Apply sample weights and reduction
        if weight is not None:
            assert weight.ndim == loss.ndim
            assert len(weight) == len(pred)

        loss = (
            weight_reduce_loss(loss, weight, reduction, avg_factor) * self.loss_weight
        )

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
