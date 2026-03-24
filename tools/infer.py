#!/usr/bin/env python3
"""Single-image inference + visualization for det-baseline.

Usage:
    python tools/infer.py configs/rtmdet_resnet50_dr.py checkpoint.pth image.jpg

    # Custom output path and score threshold
    python tools/infer.py configs/rtmdet_resnet50_dr.py checkpoint.pth image.jpg \
        --out result.jpg --score-thr 0.3

    # CPU inference
    python tools/infer.py configs/rtmdet_resnet50_dr.py checkpoint.pth image.jpg \
        --device cpu

    # Batch inference on a folder
    python tools/infer.py configs/rtmdet_resnet50_dr.py checkpoint.pth \
        dataset-example/images/test/ --out-dir results/
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import det_baseline  # noqa: F401

import cv2
from mmdet.apis import DetInferencer

CLASS_NAMES = [
    'cotton_wool_spots',
    'epiretinal_membranes',
    'hard_exudates',
    'hemorrhages_or_microaneurysms',
    'intraretinal_microvascular_anomalies',
    'neovascularization_nv',
    'venous_beading',
    'vitreous_or_preretinal_hemorrhage',
]

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on image(s)')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--out', default=None,
                        help='Output path for a single image (default: result.jpg)')
    parser.add_argument('--out-dir', default='results',
                        help='Output directory for batch inference (default: results/)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold for visualization (default: 0.3)')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device (default: cuda:0)')
    return parser.parse_args()


def print_detections(predictions: dict, score_thr: float):
    bboxes = predictions.get('bboxes', [])
    scores = predictions.get('scores', [])
    labels = predictions.get('labels', [])

    kept = [(b, s, l) for b, s, l in zip(bboxes, scores, labels) if s >= score_thr]
    print(f'  Detected {len(kept)} object(s) (score >= {score_thr}):')
    for bbox, score, label in kept:
        x1, y1, x2, y2 = bbox
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'cls_{label}'
        print(f'    [{name}]  score={score:.3f}  '
              f'bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})')


def main():
    args = parse_args()

    inferencer = DetInferencer(
        model=args.config,
        weights=args.checkpoint,
        device=args.device,
        palette='random',
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch mode
        img_paths = [
            p for p in sorted(input_path.iterdir())
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        print(f'Found {len(img_paths)} images in {input_path}')
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            result = inferencer(
                inputs=str(img_path),
                out_dir=str(out_dir),
                pred_score_thr=args.score_thr,
                show=False,
                return_vis=False,
                no_save_pred=True,
            )
            print(f'\n{img_path.name}:')
            print_detections(result['predictions'][0], args.score_thr)

        print(f'\nVisualizations saved to {out_dir}/vis/')

    else:
        # Single image mode — get vis back and save to the requested path.
        # Do NOT pass out_dir here; we handle saving ourselves to avoid
        # inferencer writing an extra copy under out_dir/vis/.
        out_path = args.out or 'result.jpg'

        result = inferencer(
            inputs=str(input_path),
            pred_score_thr=args.score_thr,
            show=False,
            return_vis=True,
            no_save_pred=True,
        )

        # DetInferencer returns visualization in RGB (matplotlib convention).
        # cv2.imwrite expects BGR — convert before saving.
        vis_rgb = result['visualization'][0]
        cv2.imwrite(out_path, cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
        print(f'Saved visualization → {out_path}')
        print(f'\n{input_path.name}:')
        print_detections(result['predictions'][0], args.score_thr)


if __name__ == '__main__':
    main()
