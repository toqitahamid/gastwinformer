from typing import List, Optional, Dict, Any, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


@MODELS.register_module()
class ConfigurableSegmentor(BaseModel):
    """Template-based segmentor that builds components from configurations.
    
    This segmentor is a pure template that orchestrates components:
    - Builds backbone, necks, and heads from config
    - Supports 4 neck strategies (none, shared, separate, hybrid)
    - Routes features between components based on configuration
    - Aggregates losses from multiple heads
    - Handles pretrained weight loading at model level
    
    Args:
        backbone (dict): Backbone configuration
        decode_head (dict): Segmentation head configuration
        cls_head (dict, optional): Classification head configuration
        auxiliary_head (dict, optional): Auxiliary head configuration for deep supervision
        neck (dict, optional): Single neck configuration (Option 2)
        necks (dict, optional): Multiple neck configurations (Options 3&4)
        routing (dict, optional): Feature routing configuration
        loss_weights (dict, optional): Loss weights for different heads
        pretrained (str, optional): Path to pretrained backbone weights
        data_preprocessor (dict, optional): Data preprocessor configuration
        train_cfg (dict, optional): Training configuration
        test_cfg (dict, optional): Testing configuration
        init_cfg (dict, optional): Initialization configuration for full model
    """
    
    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 cls_head: Optional[dict] = None,
                 auxiliary_head: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 necks: Optional[dict] = None,
                 routing: Optional[dict] = None,
                 loss_weights: Optional[Dict[str, float]] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        
        # Handle backbone pretraining (MMSegmentation style)
        if pretrained is not None:
            backbone = backbone.copy()
            backbone['init_cfg'] = dict(type='Pretrained', checkpoint=pretrained)
        
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # Build backbone
        self.backbone = self._build_backbone(backbone)
        
        # Determine and build neck strategy
        self.neck_strategy = self._determine_neck_strategy(neck, necks)
        self.necks = self._build_necks(neck, necks)
        
        # Build heads
        self.heads = self._build_heads(decode_head, cls_head, auxiliary_head)
        
        # Configuration
        self.routing_config = routing or self._default_routing_config()
        self.loss_weights = loss_weights or self._get_default_loss_weights(cls_head, auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Validate configuration
        self._validate_configuration()
    
    def _build_backbone(self, backbone: dict) -> nn.Module:
        """Build backbone from configuration."""
        return MODELS.build(backbone)
    
    def _determine_neck_strategy(self, 
                                neck_config: Optional[dict], 
                                neck_configs: Optional[dict]) -> str:
        """Determine which neck strategy is being used."""
        if neck_config is None and neck_configs is None:
            return 'none'  # Option 1: No necks
        elif neck_config is not None and neck_configs is None:
            return 'shared'  # Option 2: Single shared neck
        elif neck_config is None and neck_configs is not None:
            if 'shared' in neck_configs and 'task_specific' in neck_configs:
                return 'hybrid'  # Option 4: Hybrid necks
            else:
                return 'separate'  # Option 3: Separate necks
        else:
            raise ValueError("Cannot specify both neck_config and neck_configs")
    
    def _build_necks(self, 
                    neck_config: Optional[dict], 
                    neck_configs: Optional[dict]) -> nn.ModuleDict:
        """Build necks based on strategy."""
        necks = nn.ModuleDict()
        
        if self.neck_strategy == 'none':
            pass  # No necks to build
        
        elif self.neck_strategy == 'shared':
            necks['shared'] = MODELS.build(neck_config)
        
        elif self.neck_strategy == 'separate':
            for neck_name, neck_cfg in neck_configs.items():
                necks[neck_name] = MODELS.build(neck_cfg)
        
        elif self.neck_strategy == 'hybrid':
            # Build shared neck
            necks['shared'] = MODELS.build(neck_configs['shared'])
            # Build task-specific necks
            for neck_name, neck_cfg in neck_configs['task_specific'].items():
                necks[f'task_{neck_name}'] = MODELS.build(neck_cfg)
        
        return necks
    
    def _build_heads(self, decode_head: dict, cls_head: Optional[dict] = None, auxiliary_head: Optional[dict] = None) -> nn.ModuleDict:
        """Build heads from configurations."""
        head_modules = nn.ModuleDict()
        
        # Build decode head (segmentation)
        head_modules['segmentation'] = MODELS.build(decode_head)
        
        # Build classification head if provided
        if cls_head is not None:
            head_modules['classification'] = MODELS.build(cls_head)
        
        # Build auxiliary head if provided (for deep supervision)
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                # Multiple auxiliary heads
                head_modules['auxiliary'] = nn.ModuleList()
                for aux_cfg in auxiliary_head:
                    head_modules['auxiliary'].append(MODELS.build(aux_cfg))
            else:
                # Single auxiliary head
                head_modules['auxiliary'] = MODELS.build(auxiliary_head)
            
        return head_modules
    
    def _get_default_loss_weights(self, cls_head: Optional[dict] = None, auxiliary_head: Optional[dict] = None) -> Dict[str, float]:
        """Get default loss weights based on available heads."""
        weights = {'segmentation': 1.0}
        if cls_head is not None:
            weights['classification'] = 1.0
        if auxiliary_head is not None:
            weights['auxiliary'] = 0.4  # Standard auxiliary loss weight
        return weights
    
    def _default_routing_config(self) -> dict:
        """Generate default routing configuration based on neck strategy."""
        config = {
            'backbone_to_necks': {},
            'necks_to_heads': {}
        }
        
        if self.neck_strategy == 'none':
            # Direct routing from backbone to heads - always pass all features
            # Let heads decide which features to use via their own selection mechanism
            for head_name in self.heads.keys():
                config['necks_to_heads'][head_name] = {
                    'source': 'backbone',
                    'features': 'all'  # Always pass all features, heads select what they need
                }
        
        elif self.neck_strategy == 'shared':
            config['backbone_to_necks']['shared'] = 'all'
            for head_name in self.heads.keys():
                config['necks_to_heads'][head_name] = {
                    'source': 'shared',
                    'features': 'all'
                }
        
        elif self.neck_strategy == 'separate':
            # Only route to necks that actually exist, others go directly from backbone
            available_necks = set(self.necks.keys())
            for head_name in self.heads.keys():
                if head_name in available_necks:
                    # Head has a corresponding neck
                    config['backbone_to_necks'][head_name] = 'all'
                    config['necks_to_heads'][head_name] = {
                        'source': head_name,
                        'features': 'all'
                    }
                else:
                    # Head goes directly from backbone - pass all features, let head select
                    config['necks_to_heads'][head_name] = {
                        'source': 'backbone',
                        'features': 'all'  # Let heads select what they need
                    }
        
        elif self.neck_strategy == 'hybrid':
            config['backbone_to_necks']['shared'] = 'all'
            available_task_necks = {k for k in self.necks.keys() if k.startswith('task_')}
            for head_name in self.heads.keys():
                task_neck_name = f'task_{head_name}'
                if task_neck_name in available_task_necks:
                    # Head has a task-specific neck
                    config['backbone_to_necks'][task_neck_name] = 'shared'
                    config['necks_to_heads'][head_name] = {
                        'source': task_neck_name,
                        'features': 'all'
                    }
                else:
                    # Head uses shared neck
                    config['necks_to_heads'][head_name] = {
                        'source': 'shared',
                        'features': 'all'
                    }
        
        return config
    
    def _validate_configuration(self):
        """Validate that configuration is consistent."""
        # Check that routing config matches available necks
        routing_necks = set(self.routing_config.get('backbone_to_necks', {}).keys())
        available_necks = set(self.necks.keys())
        
        if self.neck_strategy != 'none':
            invalid_necks = routing_necks - available_necks
            if invalid_necks:
                raise ValueError(f"Routing references non-existent necks: {invalid_necks}")
        
        # Check that all heads have routing configuration
        for head_name in self.heads.keys():
            if head_name not in self.routing_config.get('necks_to_heads', {}):
                raise ValueError(f"No routing configuration for head: {head_name}")
        
        # Validate that all routing sources exist
        necks_to_heads = self.routing_config.get('necks_to_heads', {})
        for head_name, routing_info in necks_to_heads.items():
            source = routing_info['source']
            if source != 'backbone' and source not in available_necks:
                raise ValueError(f"Head '{head_name}' routes from non-existent source: {source}")
    
    @property
    def with_necks(self):
        """Check if any necks exist."""
        return len(self.necks) > 0
    
    @property
    def with_auxiliary_head(self):
        """Check if auxiliary head exists."""
        return 'auxiliary' in self.heads
    
    def extract_feat(self, inputs: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], tuple]:
        """Extract features from backbone."""
        features = self.backbone(inputs)
        
        # Handle different backbone output formats
        # Some backbones (like DDRNet) return different formats for training vs inference
        if isinstance(features, torch.Tensor):
            # Single tensor output (e.g., DDRNet during inference)
            return features
        elif isinstance(features, (list, tuple)):
            # Multiple features output (e.g., DDRNet during training, FPN outputs)
            return features
        else:
            # Fallback: wrap in list
            return [features]
    
    def route_features(self, 
                      features: Union[torch.Tensor, List[torch.Tensor]], 
                      routing_rule: Union[str, List[int], int]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Route features based on routing rule."""
        if routing_rule == 'all':
            # For 'all' rule, preserve the original format (crucial for heads like DDRHead)
            return features
        
        # Handle single tensor case
        if isinstance(features, torch.Tensor):
            if routing_rule == 'last':
                return features
            elif isinstance(routing_rule, int):
                if routing_rule == 0 or routing_rule == -1:
                    return features
                else:
                    raise IndexError(f"Single tensor provided but routing rule {routing_rule} requires index access")
            else:
                return features
        
        # Handle list/tuple of tensors case
        if isinstance(features, (list, tuple)):
            features = list(features)  # Ensure it's a list for indexing
        else:
            features = [features]
        
        if routing_rule == 'last':
            return features[-1] 
        elif isinstance(routing_rule, int):
            return features[routing_rule]
        elif isinstance(routing_rule, (list, tuple)):
            return [features[i] for i in routing_rule]
        else:
            raise ValueError(f"Unknown routing rule: {routing_rule}")
    
    def process_through_necks(self, backbone_features: Union[torch.Tensor, List[torch.Tensor], tuple]) -> Dict[str, Any]:
        """Process features through necks based on strategy."""
        neck_outputs = {}
        
        # Always add backbone features for direct routing
        neck_outputs['backbone'] = backbone_features
        
        if self.neck_strategy == 'none':
            pass  # Only backbone features available
        
        elif self.neck_strategy == 'shared':
            routing_rule = self.routing_config['backbone_to_necks']['shared']
            routed_features = self.route_features(backbone_features, routing_rule)
            neck_outputs['shared'] = self.necks['shared'](routed_features)
        
        elif self.neck_strategy == 'separate':
            for neck_name, neck in self.necks.items():
                routing_rule = self.routing_config['backbone_to_necks'][neck_name]
                routed_features = self.route_features(backbone_features, routing_rule)
                neck_outputs[neck_name] = neck(routed_features)
        
        elif self.neck_strategy == 'hybrid':
            # Process through shared neck first
            shared_routing = self.routing_config['backbone_to_necks']['shared']
            shared_features = self.route_features(backbone_features, shared_routing)
            shared_output = self.necks['shared'](shared_features)
            neck_outputs['shared'] = shared_output
            
            # Process through task-specific necks
            for neck_name, neck in self.necks.items():
                if neck_name.startswith('task_'):
                    routing_rule = self.routing_config['backbone_to_necks'][neck_name]
                    if routing_rule == 'shared':
                        input_features = shared_output
                    else:
                        input_features = self.route_features(backbone_features, routing_rule)
                    neck_outputs[neck_name] = neck(input_features)
        
        return neck_outputs
    
    def route_to_heads(self, neck_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Route neck outputs to appropriate heads."""
        head_inputs = {}
        
        for head_name in self.heads.keys():
            routing_config = self.routing_config['necks_to_heads'][head_name]
            source = routing_config['source']
            features_rule = routing_config['features']
            
            source_features = neck_outputs[source]
            routed_features = self.route_features(source_features, features_rule)
            head_inputs[head_name] = routed_features
        
        return head_inputs
    
    def forward(self, 
                inputs: torch.Tensor,
                data_samples: Optional[List[SegDataSample]] = None,
                mode: str = 'tensor') -> Any:
        """Forward function supporting multiple modes."""
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            return self.tensor_forward(inputs)
    
    def loss(self, 
             inputs: torch.Tensor, 
             data_samples: List[SegDataSample]) -> Dict[str, torch.Tensor]:
        """Calculate losses from all heads."""
        losses = {}
        
        # Extract backbone features
        backbone_features = self.extract_feat(inputs)
        
        # Process through necks
        neck_outputs = self.process_through_necks(backbone_features)
        
        # Route to heads
        head_inputs = self.route_to_heads(neck_outputs)
        
        # Calculate losses for each head
        for head_name, head in self.heads.items():
            head_features = head_inputs[head_name]
            
            # Handle auxiliary heads (can be single head or ModuleList)
            if head_name == 'auxiliary':
                if isinstance(head, nn.ModuleList):
                    # Multiple auxiliary heads
                    for idx, aux_head in enumerate(head):
                        aux_losses = aux_head.loss(head_features, data_samples, self.train_cfg)
                        weight = self.loss_weights.get(head_name, 1.0)
                        for loss_name, loss_value in aux_losses.items():
                            losses[f'{head_name}_{idx}.{loss_name}'] = loss_value * weight
                else:
                    # Single auxiliary head
                    aux_losses = head.loss(head_features, data_samples, self.train_cfg)
                    weight = self.loss_weights.get(head_name, 1.0)
                    for loss_name, loss_value in aux_losses.items():
                        losses[f'{head_name}.{loss_name}'] = loss_value * weight
            else:
                # Regular heads (segmentation, classification)
                head_losses = head.loss(head_features, data_samples, self.train_cfg)
                weight = self.loss_weights.get(head_name, 1.0)
                for loss_name, loss_value in head_losses.items():
                    losses[f'{head_name}.{loss_name}'] = loss_value * weight
        
        return losses
    
    def predict(self, 
                inputs: torch.Tensor,
                data_samples: List[SegDataSample]) -> List[SegDataSample]:
        """Forward function for prediction."""
        # Extract backbone features
        backbone_features = self.extract_feat(inputs)
        
        # Process through necks
        neck_outputs = self.process_through_necks(backbone_features)
        
        # Route to heads
        head_inputs = self.route_to_heads(neck_outputs)
        
        # Get predictions from each head (skip auxiliary heads during inference)
        head_predictions = {}
        for head_name, head in self.heads.items():
            if head_name == 'auxiliary':
                continue  # Skip auxiliary heads during inference
                
            head_features = head_inputs[head_name]
            
            if head_name == 'segmentation':
                # Handle segmentation head
                batch_img_metas = [dict(sample.metainfo) for sample in data_samples]
                head_predictions[head_name] = head.predict(head_features, batch_img_metas, self.test_cfg)
            else:
                # Handle other heads (e.g., classification)
                if hasattr(head, 'predict'):
                    head_predictions[head_name] = head.predict(head_features)
                else:
                    # Fallback for heads without predict method
                    head_predictions[head_name] = head(head_features)
        
        # Update data_samples with predictions
        for i, sample in enumerate(data_samples):
            # Handle segmentation predictions
            if 'segmentation' in head_predictions:
                seg_result = head_predictions['segmentation'][i]
                if isinstance(seg_result, torch.Tensor):
                    if seg_result.dim() == 3 and seg_result.shape[0] > 1:
                        seg_pred = torch.argmax(seg_result, dim=0, keepdim=True)
                    else:
                        seg_pred = seg_result
                    sample.pred_sem_seg = PixelData(data=seg_pred)
                else:
                    sample.pred_sem_seg = seg_result.pred_sem_seg
            
            # Handle classification predictions
            if 'classification' in head_predictions:
                cls_result = head_predictions['classification']
                if isinstance(cls_result, torch.Tensor):
                    cls_pred = torch.softmax(cls_result, dim=1)
                    sample.pred_cls_score = cls_pred[i]
                    sample.pred_cls_label = cls_pred[i].argmax().item()
                else:
                    # Handle structured classification output
                    try:
                        if isinstance(cls_result, dict):
                            # Dictionary format with cls_score and cls_label keys
                            if 'cls_score' in cls_result:
                                sample.pred_cls_score = cls_result['cls_score'][i]
                            elif len(cls_result) > 0:
                                # Try to get first available result
                                first_key = list(cls_result.keys())[0]
                                sample.pred_cls_score = cls_result[first_key][i]
                            
                            if 'cls_label' in cls_result:
                                sample.pred_cls_label = cls_result['cls_label'][i]
                            elif 'cls_score' in cls_result:
                                sample.pred_cls_label = cls_result['cls_score'][i].argmax().item()
                        elif isinstance(cls_result, (list, tuple)):
                            # List/tuple format
                            if len(cls_result) > i:
                                cls_item = cls_result[i]
                                if isinstance(cls_item, torch.Tensor):
                                    sample.pred_cls_score = cls_item
                                    sample.pred_cls_label = cls_item.argmax().item()
                                else:
                                    sample.pred_cls_score = cls_item
                                    sample.pred_cls_label = cls_item
                        else:
                            # Single result for entire batch or unknown format
                            if hasattr(cls_result, '__len__') and len(cls_result) > i:
                                sample.pred_cls_score = cls_result[i]
                                if isinstance(cls_result[i], torch.Tensor):
                                    sample.pred_cls_label = cls_result[i].argmax().item()
                                else:
                                    sample.pred_cls_label = cls_result[i]
                            else:
                                # Fallback: assume single prediction for all samples
                                sample.pred_cls_score = cls_result
                                if isinstance(cls_result, torch.Tensor):
                                    sample.pred_cls_label = cls_result.argmax().item()
                                else:
                                    sample.pred_cls_label = cls_result
                    except (IndexError, KeyError, AttributeError) as e:
                        # Fallback handling for any indexing/access errors
                        print(f"Warning: Error accessing classification result for sample {i}: {e}")
                        print(f"cls_result type: {type(cls_result)}, shape/length: {getattr(cls_result, 'shape', len(cls_result) if hasattr(cls_result, '__len__') else 'N/A')}")
                        # Set default values
                        sample.pred_cls_score = torch.tensor([0.0])
                        sample.pred_cls_label = 0
        
        return data_samples
    
    def tensor_forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward for tensor mode."""
        # Extract backbone features
        backbone_features = self.extract_feat(inputs)
        
        # Process through necks
        neck_outputs = self.process_through_necks(backbone_features)
        
        # Route to heads
        head_inputs = self.route_to_heads(neck_outputs)
        
        # Get outputs from each head (skip auxiliary heads during inference)
        outputs = {}
        for head_name, head in self.heads.items():
            if head_name == 'auxiliary':
                continue  # Skip auxiliary heads during inference
                
            head_features = head_inputs[head_name]
            outputs[head_name] = head(head_features)
        
        return outputs