# ============================================================
# Training schedule — 100 epochs (shared by all backbone configs)
#
# Strategy:
#   - Warmup: linear LR ramp for first 1000 iterations
#   - Main:   cosine annealing from epoch 50 → 100
#   - Last 15 epochs: Mosaic is switched off (via PipelineSwitchHook
#     defined in each backbone config), val every epoch
# ============================================================

max_epochs = 100
num_last_epochs = 15  # switch to stage-2 pipeline at epoch (max_epochs - num_last_epochs)
base_lr = 4e-4

# ── Optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05,
    ),
)

# ── LR schedulers ─────────────────────────────────────────────────────────────
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000,           # linear warmup for first 1000 iters
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# ── Training / val loops ──────────────────────────────────────────────────────
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    # val every epoch once Mosaic is off and we're in the final stretch
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Hooks ─────────────────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='auto',   # saves best coco/bbox_mAP checkpoint
    ),
)
