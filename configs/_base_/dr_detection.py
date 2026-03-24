# ============================================================
# Base dataset config: Diabetic Retinopathy Detection
#
# Annotation format: COCO JSON (converted from YOLO txt via
#   python tools/convert_yolo_to_coco.py --data-root dataset-example
#
# Class index mapping (0-based, matching original YOLO labels):
#   0: cotton_wool_spots
#   1: epiretinal_membranes
#   2: hard_exudates
#   3: hemorrhages_or_microaneurysms
#   4: intraretinal_microvascular_anomalies
#   5: neovascularization_nv
#   6: venous_beading
#   7: vitreous_or_preretinal_hemorrhage
# ============================================================

dataset_type = 'YOLOv5CocoDataset'
data_root = 'dataset-example/'

metainfo = dict(
    classes=(
        'cotton_wool_spots',
        'epiretinal_membranes',
        'hard_exudates',
        'hemorrhages_or_microaneurysms',
        'intraretinal_microvascular_anomalies',
        'neovascularization_nv',
        'venous_beading',
        'vitreous_or_preretinal_hemorrhage',
    ),
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
    ]
)

num_classes = 8
img_scale = (1024, 1024)   # (H, W) — shared by all pipelines

# Max cached images for Mosaic / MixUp in-memory augmentation.
# Reduce if RAM is limited.
_max_cached_images = 20

# ── Stage-1 train pipeline (with Mosaic + MixUp) ─────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=_max_cached_images,
        pad_val=114.0,
    ),
    dict(
        type='mmdet.RandomResize',
        scale=(2048, 2048),   # 2 × img_scale; scales with mosaic output
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_max_cached_images,
    ),
    dict(type='mmdet.PackDetInputs'),
]

# ── Stage-2 train pipeline (no Mosaic / MixUp, last N epochs) ────────────────
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs'),
]

# ── Val / Test pipeline ───────────────────────────────────────────────────────
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        serialize_data=False,   # required for Mosaic use_cached=True
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=val_pipeline,
    ),
)

# ── Evaluators ────────────────────────────────────────────────────────────────
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(1, 10, 100),
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(1, 10, 100),
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
)
