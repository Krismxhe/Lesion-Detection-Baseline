#!/usr/bin/env python3
"""Training entry point for det-baseline.

Usage:
    # Single-GPU training
    python tools/train.py configs/rtmdet_resnet50_dr.py

    # With custom work-dir
    python tools/train.py configs/rtmdet_swin_t_dr.py --work-dir work_dirs/exp_swin

    # Resume from last checkpoint
    python tools/train.py configs/rtmdet_pvtv2_b2_dr.py --resume

    # AMP (mixed precision, saves ~30% VRAM)
    python tools/train.py configs/rtmdet_efficientnet_b3_dr.py --amp

    # Override any config option on the fly
    python tools/train.py configs/rtmdet_resnet50_dr.py \
        --cfg-options train_dataloader.batch_size=8 train_cfg.max_epochs=50

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 tools/train.py configs/rtmdet_resnet50_dr.py
"""

import argparse
import os
import sys

# Add project root to PYTHONPATH so `import det_baseline` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import det_baseline  # noqa: F401 — registers all mmdet/mmpretrain/mmyolo modules

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detection model')
    parser.add_argument('config', help='Path to mmyolo config file')
    parser.add_argument('--work-dir', help='Directory for logs and checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint in work-dir')
    parser.add_argument('--amp', action='store_true',
                        help='Enable Automatic Mixed Precision (FP16) training')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config key=value pairs, e.g. '
             'train_dataloader.batch_size=8 train_cfg.max_epochs=50',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir:
        cfg.work_dir = args.work_dir

    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
