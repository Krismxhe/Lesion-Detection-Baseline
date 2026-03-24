#!/usr/bin/env python3
"""Convert YOLO-format annotations to COCO JSON for MMDetection/MMYolo.

YOLO format (per-image .txt file):
    <class_id>  <cx_norm>  <cy_norm>  <w_norm>  <h_norm>

Output COCO JSON is written to <data_root>/annotations/{split}.json.

Usage:
    python tools/convert_yolo_to_coco.py
    python tools/convert_yolo_to_coco.py --data-root dataset-example --splits train val test
"""

import argparse
import json
from pathlib import Path

import cv2

# Must match the order in dataset.yaml (0-indexed)
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

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def convert_split(data_root: Path, split: str) -> dict:
    img_dir = data_root / 'images' / split
    label_dir = data_root / 'labels' / split

    categories = [
        {'id': i, 'name': name, 'supercategory': 'lesion'}
        for i, name in enumerate(CLASS_NAMES)
    ]

    images, annotations = [], []
    ann_id = 0

    img_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not img_paths:
        print(f'  [warn] No images found in {img_dir}')

    for img_id, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [warn] Cannot read {img_path}, skipping.')
            continue
        h, w = img.shape[:2]

        images.append({
            'id': img_id,
            'file_name': img_path.name,
            'height': h,
            'width': w,
        })

        label_path = label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue  # image with no annotations is fine

        lines = label_path.read_text().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            # YOLO normalized → COCO absolute (x_min, y_min, w, h)
            abs_w = bw * w
            abs_h = bh * h
            x_min = max(0.0, (cx - bw / 2) * w)
            y_min = max(0.0, (cy - bh / 2) * h)
            abs_w = min(abs_w, w - x_min)
            abs_h = min(abs_h, h - y_min)

            if abs_w <= 0 or abs_h <= 0:
                continue

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cls_id,
                'bbox': [round(x_min, 2), round(y_min, 2),
                         round(abs_w, 2), round(abs_h, 2)],
                'area': round(abs_w * abs_h, 2),
                'iscrowd': 0,
            })
            ann_id += 1

    return {
        'info': {'description': 'DR Detection (converted from YOLO format)'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='dataset-example',
                        help='Dataset root directory (default: dataset-example)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Splits to convert (default: train val test)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = data_root / 'annotations'
    out_dir.mkdir(exist_ok=True)

    for split in args.splits:
        print(f'Converting [{split}] ...')
        coco = convert_split(data_root, split)
        out_path = out_dir / f'{split}.json'
        with open(out_path, 'w') as f:
            json.dump(coco, f, indent=2)
        print(f'  Saved {out_path}  '
              f'({len(coco["images"])} images, {len(coco["annotations"])} annotations)')

    print('\nDone. Run training next:')
    print('  python tools/train.py configs/rtmdet_resnet50_dr.py')


if __name__ == '__main__':
    main()
