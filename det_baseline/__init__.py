"""det_baseline — RTMDet multi-backbone detection framework.

Importing this package ensures that all mmdet, mmpretrain, and mmyolo
modules are registered in the mmengine registry before any config is
parsed. This is required so that:
  - mmdet.ResNet, mmdet.TIMMBackbone, mmdet.GIoULoss, etc. resolve correctly
  - mmpretrain.SwinTransformer, mmpretrain.PyramidVisionTransformerV2 resolve
  - mmyolo's YOLODetector, CSPNeXtPAFPN, RTMDetSepBNHead resolve

Always import det_baseline before calling Runner.from_cfg(cfg).
"""

from mmdet.utils import register_all_modules as register_mmdet_modules
from mmpretrain.utils import register_all_modules as register_mmpretrain_modules
from mmyolo.utils import register_all_modules as register_mmyolo_modules

# Registration order matters: mmdet first (mmyolo depends on it),
# then mmpretrain, finally mmyolo which sets the default scope so
# that type='YOLODetector' / 'CSPNeXtPAFPN' / 'RTMDetSepBNHead'
# resolve correctly when building models from config.
# mmdet first
register_mmdet_modules(init_default_scope=False)

# then mmpretrain
register_mmpretrain_modules(init_default_scope=False)

# finally mmyolo, and let it own the default scope
register_mmyolo_modules(init_default_scope=True)
