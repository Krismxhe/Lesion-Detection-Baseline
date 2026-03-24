# ============================================================
# RTMDet + ConvNeXt-Small — DR Detection Baseline
#
# Backbone:  ConvNeXt-Small (ImageNet-1k pretrained, from mmpretrain)
# Neck:      CSPNeXtPAFPN  in=[192,384,768]  out=128
# Head:      RTMDetSepBNHead  feat=128
#
# ConvNeXt-S stage output channels (out_indices=(1,2,3)):
#   stage1 → C3: 192   (stride 8)
#   stage2 → C4: 384   (stride 16)
#   stage3 → C5: 768   (stride 32)
#
# Key parameters:
#   gap_before_final_norm=False — must be False for detection; setting it
#     True would apply global-average-pooling before the final LayerNorm
#     and collapse the spatial dimensions, breaking the neck.
#   layer_scale_init_value=1e-6 — per the ConvNeXt paper for IN-1k models.
#   drop_path_rate=0.4 — standard stochastic depth rate for ConvNeXt-S.
# ============================================================

_base_ = [
    './_base_/dr_detection.py',
    './_base_/schedule_100e.py',
    './_base_/default_runtime.py',
]

_backbone_out_channels = [192, 384, 768]
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
        type='mmpretrain.ConvNeXt',
        arch='small',
        out_indices=(1, 2, 3),
        drop_path_rate=0.4,
        layer_scale_init_value=1e-6,
        gap_before_final_norm=False,   # preserve spatial maps for detection
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/mmclassification/v0/'
                'convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth'
            ),
            prefix='backbone.',
        ),
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

work_dir = 'work_dirs/rtmdet_convnext_s_dr'
