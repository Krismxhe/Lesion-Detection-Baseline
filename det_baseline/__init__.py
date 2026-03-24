"""det_baseline — RTMDet multi-backbone detection framework.

Importing this package ensures that all mmdet, mmpretrain, and mmyolo
modules are registered in the mmengine registry before any config is
parsed. This is required so that:
  - mmdet.ResNet, mmdet.TIMMBackbone, mmdet.GIoULoss, etc. resolve correctly
  - mmpretrain.SwinTransformer, mmpretrain.PyramidVisionTransformerV2 resolve
  - mmyolo's YOLODetector, CSPNeXtPAFPN, RTMDetSepBNHead resolve

Always import det_baseline before calling Runner.from_cfg(cfg).
"""

import mmdet
import mmpretrain
import mmyolo

# Registration order matters: mmdet first (mmyolo depends on it),
# then mmpretrain, finally mmyolo which sets the default scope so
# that type='YOLODetector' / 'CSPNeXtPAFPN' / 'RTMDetSepBNHead'
# resolve correctly when building models from config.
mmdet.register_all_modules(init_default_scope=False)
mmpretrain.register_all_modules(init_default_scope=False)
mmyolo.register_all_modules(init_default_scope=True)
