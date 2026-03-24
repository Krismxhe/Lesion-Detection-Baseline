"""Backbone sub-package.

Currently all backbones are provided by upstream packages:
  ResNet-50     → mmdet.ResNet
  Swin-T        → mmpretrain.SwinTransformer
  PVT-v2-B2    → mmpretrain.PyramidVisionTransformerV2
  EfficientNet  → mmdet.TIMMBackbone  (timm backend)

Add custom backbone classes here if needed and register them with:
    from mmyolo.registry import MODELS
    @MODELS.register_module()
    class MyBackbone: ...
"""
