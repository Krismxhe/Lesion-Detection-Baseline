#!/usr/bin/env python3
"""Evaluation (val or test split) entry point for det-baseline.

Usage:
    # Evaluate on validation set
    python tools/eval.py configs/rtmdet_resnet50_dr.py \
        work_dirs/rtmdet_resnet50_dr/best_coco_bbox_mAP_epoch_100.pth

    # Evaluate on test set
    python tools/eval.py configs/rtmdet_resnet50_dr.py \
        work_dirs/rtmdet_resnet50_dr/best_coco_bbox_mAP_epoch_100.pth \
        --split test

    # Save results JSON
    python tools/eval.py configs/rtmdet_resnet50_dr.py checkpoint.pth \
        --out results/resnet50_val.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import det_baseline  # noqa: F401

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a detection model')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to checkpoint (.pth)')
    parser.add_argument('--split', default='val', choices=['val', 'test'],
                        help='Which split to evaluate (default: val)')
    parser.add_argument('--work-dir', help='Directory to save evaluation outputs')
    parser.add_argument('--out', help='Save per-image COCO-format predictions to this path, e.g. results/resnet50_val.bbox.json')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override config options',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    if args.work_dir:
        cfg.work_dir = args.work_dir

    # Route the correct split to test_dataloader / test_evaluator
    if args.split == 'test':
        # test_dataloader / test_evaluator are already defined in _base_
        pass
    else:
        # Point test loop at the val split
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    # Write per-image prediction JSON when --out is given.
    # mmdet CocoMetric dumps results when outfile_prefix is set.
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg.test_evaluator.outfile_prefix = str(out.with_suffix(''))

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
