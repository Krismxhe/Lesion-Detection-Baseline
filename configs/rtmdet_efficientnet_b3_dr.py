# ============================================================
# RTMDet + EfficientNet-B3 (timm) — DR Detection Baseline
#
# Backbone:  EfficientNet-B3 via mmdet.TIMMBackbone (timm backend)
# Neck:      CSPNeXtPAFPN  in=[48,136,384]  out=128
# Head:      RTMDetSepBNHead  feat=128
#
# EfficientNet-B3 feature channel map (timm features_only=True):
#   index 0 → stride 2,  channels: 24
#   index 1 → stride 4,  channels: 32
#   index 2 → stride 8,  channels: 48   ← C3 (P3)
#   index 3 → stride 16, channels: 136  ← C4 (P4)
#   index 4 → stride 32, channels: 384  ← C5 (P5)
#
# Note: TIMMBackbone is registered in mmdet scope. mmdet must be
# imported before building the model (handled by det_baseline/__init__.py).
# ============================================================

_base_ = [
    './_base_/dr_detection.py',
    './_base_/schedule_100e.py',
    './_base_/default_runtime.py',
]

_backbone_out_channels = [48, 136, 384]
_neck_out_channels = 128
_num_classes = {{_base_.num_classes}}

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
    ),
    backbone=dict(
        type='mmdet.TIMMBackbone',
        model_name='efficientnet_b3',
        features_only=True,
        pretrained=True,            # loads timm ImageNet-1k weights
        out_indices=(2, 3, 4),      # P3/P4/P5 at strides 8/16/32
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=_backbone_out_channels,
        out_channels=_neck_out_channels,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=_num_classes,
        in_channels=_neck_out_channels,
        stacked_convs=2,
        feat_channels=_neck_out_channels,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32],
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=_num_classes,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300,
    ),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=85,
        switch_pipeline={{_base_.train_pipeline_stage2}},
    ),
]

work_dir = 'work_dirs/rtmdet_efficientnet_b3_dr'
