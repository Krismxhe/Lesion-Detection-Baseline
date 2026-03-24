#!/usr/bin/env python3
"""
Raw feature map visualization for det-baseline models.

Hooks backbone (C3/C4/C5) and neck (P3/P4/P5) outputs of a trained
RTMDet model and renders three types of visualizations:

  overview  — 2-row × 4-col heatmap overlay for all 6 feature layers
  channels  — top-K individual channel responses for one layer
  compare   — same layer, different backbones side-by-side

Aggregation modes (channel dim → 2-D spatial map):
  l2norm  feature-vector magnitude: feat.norm(dim=0)   [recommended]
  mean    average activation:        feat.mean(0)
  max     strongest channel:         feat.max(0).values

Usage — single model:
    python tools/visualize_features.py \\
        configs/rtmdet_resnet50_dr.py checkpoint.pth image.jpg

    python tools/visualize_features.py ... --agg mean
    python tools/visualize_features.py ... --mode channels --layer neck_p4 --top-k 32

Usage — multi-backbone comparison:
    python tools/visualize_features.py --compare \\
        --models resnet50:configs/rtmdet_resnet50_dr.py:ckpt1.pth \\
                 swin_t:configs/rtmdet_swin_t_dr.py:ckpt2.pth \\
        --compare-image image.jpg --layer neck_p4
"""

import argparse
import os
import sys
from pathlib import Path

# ── project root on path so `import det_baseline` works ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import det_baseline  # noqa — registers mmdet / mmpretrain / mmyolo modules

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmyolo.registry import MODELS

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SCALE = (1024, 1024)        # (H, W) — must match training config
MEAN_BGR  = [103.53, 116.28, 123.675]
STD_BGR   = [57.375, 57.12,  58.395]

LAYER_KEYS = [
    'backbone_c3', 'backbone_c4', 'backbone_c5',
    'neck_p3',     'neck_p4',     'neck_p5',
]
LAYER_LABELS = {
    'backbone_c3': 'Backbone C3  stride 8',
    'backbone_c4': 'Backbone C4  stride 16',
    'backbone_c5': 'Backbone C5  stride 32',
    'neck_p3':     'Neck P3  stride 8',
    'neck_p4':     'Neck P4  stride 16',
    'neck_p5':     'Neck P5  stride 32',
}


# ════════════════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════════════════

def load_model(config_path: str, checkpoint_path: str,
               device: str = 'cuda:0') -> torch.nn.Module:
    """
    Build the detector from *config_path* and load weights from
    *checkpoint_path*.  Always returns model in eval mode.
    """
    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ════════════════════════════════════════════════════════════════════════════

def preprocess_image(img_bgr: np.ndarray,
                     img_scale: tuple = IMG_SCALE,
                     device: str = 'cuda:0'):
    """
    Resize *img_bgr* to *img_scale* and apply the training-time
    normalisation (BGR mean/std, no channel swap).

    Returns
    -------
    img_batch   : [1, 3, H, W] float32 tensor on *device*
    img_resized : [H, W, 3] BGR uint8 numpy array  (for overlay)
    """
    h, w = img_scale
    img_resized = cv2.resize(img_bgr, (w, h))
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # [3,H,W]
    mean = torch.tensor(MEAN_BGR).view(3, 1, 1)
    std  = torch.tensor(STD_BGR ).view(3, 1, 1)
    img_t = (img_t - mean) / std
    return img_t.unsqueeze(0).to(device), img_resized


# ════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(model: torch.nn.Module,
                     img_bgr: np.ndarray,
                     img_scale: tuple = IMG_SCALE) -> tuple:
    """
    Run a single forward pass through backbone + neck (head skipped).

    Returns
    -------
    features   : dict  keys = LAYER_KEYS,  values = [1,C,H,W] cpu tensors
    img_resized: [H,W,3] BGR uint8  (already resized, use for overlay)
    """
    device = next(model.parameters()).device
    img_batch, img_resized = preprocess_image(img_bgr, img_scale, str(device))

    backbone_out = model.backbone(img_batch)  # tuple (C3, C4, C5)
    neck_out     = model.neck(backbone_out)   # tuple (P3, P4, P5)

    features = {}
    for i, key in enumerate(['backbone_c3', 'backbone_c4', 'backbone_c5']):
        features[key] = backbone_out[i].cpu()
    for i, key in enumerate(['neck_p3', 'neck_p4', 'neck_p5']):
        features[key] = neck_out[i].cpu()

    return features, img_resized


# ════════════════════════════════════════════════════════════════════════════
#  Channel aggregation  →  2-D spatial map
# ════════════════════════════════════════════════════════════════════════════

def aggregate(feat: torch.Tensor, mode: str = 'l2norm') -> np.ndarray:
    """
    Collapse the channel dimension of *feat* [1, C, H, W] → [H, W] float32.

    Parameters
    ----------
    mode : 'l2norm' | 'mean' | 'max'
    """
    f = feat.squeeze(0)          # [C, H, W]
    if mode == 'l2norm':
        out = f.norm(dim=0)
    elif mode == 'mean':
        out = f.mean(0)
    elif mode == 'max':
        out = f.max(0).values
    else:
        raise ValueError(f'Unknown aggregation mode "{mode}". '
                         'Choose from: l2norm, mean, max')
    return out.numpy().astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Rendering utilities
# ════════════════════════════════════════════════════════════════════════════

def normalize_to_uint8(m: np.ndarray) -> np.ndarray:
    """Min-max normalise *m* to [0, 255] uint8."""
    m = m.astype(np.float32)
    lo, hi = m.min(), m.max()
    m = (m - lo) / (hi - lo + 1e-7)
    return (m * 255).astype(np.uint8)


def make_heatmap_overlay(img_bgr: np.ndarray,
                          act_map: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Resize *act_map* to match *img_bgr*, apply JET colormap, blend.

    Returns RGB uint8 array suitable for ``plt.imshow``.
    """
    h, w = img_bgr.shape[:2]
    heat_u8     = normalize_to_uint8(act_map)
    heat_rsz    = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_color  = cv2.applyColorMap(heat_rsz, cv2.COLORMAP_JET)       # BGR
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════════════════
#  Plot 1 — 6-layer overview  (2 rows × 4 cols)
# ════════════════════════════════════════════════════════════════════════════

def plot_feature_overview(features: dict,
                           img_resized: np.ndarray,
                           agg:   str   = 'l2norm',
                           title: str   = '',
                           alpha: float = 0.5) -> plt.Figure:
    """
    Two-row grid:
      Row 0 → original | backbone C3 | C4 | C5
      Row 1 → original | neck     P3 | P4 | P5

    Channel count and spatial size are shown in each cell title.
    Color scale is *per-layer* (each layer normalised independently).
    """
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))

    rows = [
        ('backbone', ['c3', 'c4', 'c5'], ['C3 stride 8', 'C4 stride 16', 'C5 stride 32']),
        ('neck',     ['p3', 'p4', 'p5'], ['P3 stride 8', 'P4 stride 16', 'P5 stride 32']),
    ]

    for row_i, (prefix, suffixes, slabels) in enumerate(rows):
        axes[row_i, 0].imshow(img_rgb)
        axes[row_i, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[row_i, 0].axis('off')

        for col_i, (sfx, slabel) in enumerate(zip(suffixes, slabels), start=1):
            key  = f'{prefix}_{sfx}'
            feat = features[key]                # [1, C, H, W]
            act  = aggregate(feat, agg)         # [H, W]
            ovl  = make_heatmap_overlay(img_resized, act, alpha)
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]

            axes[row_i, col_i].imshow(ovl)
            layer_tag = 'Backbone' if prefix == 'backbone' else 'Neck'
            axes[row_i, col_i].set_title(
                f'{layer_tag} {slabel}\n[{C} ch  {H}×{W}]',
                fontsize=10,
            )
            axes[row_i, col_i].axis('off')

    suptitle = f'Feature Map Overview  |  aggregation = {agg}'
    if title:
        suptitle = f'{title}   —   {suptitle}'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Plot 2 — individual channel grid
# ════════════════════════════════════════════════════════════════════════════

def plot_channel_grid(feat:       torch.Tensor,
                      layer_name: str   = '',
                      top_k:      int   = 32,
                      ncols:      int   = 8) -> plt.Figure:
    """
    Show the *top_k* most-activated channels from *feat* [1, C, H, W].
    Channels are ranked by mean absolute activation.
    Each channel is independently min-max normalised (VIRIDIS colormap).
    The activation score (mean |act|) is shown as the sub-title.
    """
    f    = feat.squeeze(0).numpy()      # [C, H, W]
    C    = f.shape[0]
    top_k = min(top_k, C)

    scores  = np.abs(f).mean(axis=(1, 2))           # [C]
    top_idx = np.argsort(scores)[::-1][:top_k]

    nrows  = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.0, nrows * 2.2))
    axes = np.array(axes).flatten()

    for i, ch in enumerate(top_idx):
        ch_u8     = normalize_to_uint8(f[ch])
        ch_color  = cv2.applyColorMap(ch_u8, cv2.COLORMAP_VIRIDIS)
        axes[i].imshow(cv2.cvtColor(ch_color, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'ch {ch}\n{scores[ch]:.3f}', fontsize=7)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    head = f'Top-{top_k} Channels  (ranked by mean |activation|)'
    if layer_name:
        head = f'{LAYER_LABELS.get(layer_name, layer_name)}   —   {head}'
    fig.suptitle(head, fontsize=11, fontweight='bold')
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Plot 3 — multi-backbone comparison
# ════════════════════════════════════════════════════════════════════════════

def plot_backbone_comparison(models_and_names: list,
                              img_bgr: np.ndarray,
                              layer:   str   = 'neck_p4',
                              agg:     str   = 'l2norm',
                              alpha:   float = 0.5) -> plt.Figure:
    """
    Load and compare *layer* across N models on the same image.

    *models_and_names* : list of ``(name: str, model: nn.Module)``

    Color scale is UNIFIED across all subplots for fair visual comparison.
    """
    n   = len(models_and_names)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.5 * (n + 1), 5.5))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # ── first pass: collect activation maps + metadata ─────────────────────
    act_maps, subtitles, img_for_overlay = [], [], None
    for name, model in models_and_names:
        feats, img_rsz = extract_features(model, img_bgr)
        if img_for_overlay is None:
            img_for_overlay = img_rsz
        act = aggregate(feats[layer], agg)
        act_maps.append(act)
        C, H, W = feats[layer].shape[1], feats[layer].shape[2], feats[layer].shape[3]
        subtitles.append(f'{name}\n[{C} ch  {H}×{W}]')

    # ── unified colour scale ───────────────────────────────────────────────
    g_min = min(m.min() for m in act_maps)
    g_max = max(m.max() for m in act_maps)

    for i, (act, subtitle) in enumerate(zip(act_maps, subtitles)):
        act_norm = (act - g_min) / (g_max - g_min + 1e-7)
        act_u8   = (act_norm * 255).astype(np.uint8)
        h, w     = img_for_overlay.shape[:2]
        act_rsz  = cv2.resize(act_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        heat     = cv2.applyColorMap(act_rsz, cv2.COLORMAP_JET)
        ovl      = cv2.addWeighted(img_for_overlay, 1 - alpha, heat, alpha, 0)

        axes[i + 1].imshow(cv2.cvtColor(ovl, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(subtitle, fontsize=11)
        axes[i + 1].axis('off')

    layer_label = LAYER_LABELS.get(layer, layer)
    fig.suptitle(
        f'Backbone Comparison   |   {layer_label}   |   agg = {agg}\n'
        f'(unified colour scale)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  Diagnostic utility — print feature map stats (also used by notebook)
# ════════════════════════════════════════════════════════════════════════════

def print_feature_stats(features: dict) -> None:
    """
    Print a table of shapes and basic activation statistics for all 6 layers.
    Useful for a quick sanity-check before plotting.

    Columns: layer | shape | l2norm mean | l2norm max | dead channels (|act|<1e-4)
    """
    hdr = f"{'layer':15s}  {'shape':20s}  {'l2norm mean':>12}  {'l2norm max':>10}  {'dead ch':>8}"
    print(hdr)
    print('─' * len(hdr))
    for key in LAYER_KEYS:
        f   = features[key]                          # [1, C, H, W]
        act = aggregate(f, 'l2norm')                 # [H, W]
        ch_activity = f.squeeze(0).abs().mean((1, 2))  # [C]
        dead = int((ch_activity < 1e-4).sum())
        print(
            f'{key:15s}  {str(tuple(f.shape)):20s}  '
            f'{act.mean():>12.4f}  {act.max():>10.4f}  '
            f'{dead:>4d}/{f.shape[1]}'
        )


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Visualize raw feature maps of a trained RTMDet model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # single-model positional args
    p.add_argument('config',     nargs='?', help='Config file path')
    p.add_argument('checkpoint', nargs='?', help='Checkpoint file (.pth)')
    p.add_argument('image',      nargs='?', help='Input image path')

    # visualization control
    p.add_argument('--mode',  default='overview',
                   choices=['overview', 'channels'],
                   help='Visualization mode (default: overview)')
    p.add_argument('--layer', default='neck_p4', choices=LAYER_KEYS,
                   help='Target layer for --mode channels (default: neck_p4)')
    p.add_argument('--agg',   default='l2norm',
                   choices=['l2norm', 'mean', 'max'],
                   help='Channel aggregation (default: l2norm)')
    p.add_argument('--top-k', type=int, default=32,
                   help='Top-K channels in channel-grid mode (default: 32)')
    p.add_argument('--alpha', type=float, default=0.5,
                   help='Heatmap alpha blend (default: 0.5)')

    # multi-backbone compare
    p.add_argument('--compare', action='store_true',
                   help='Enable multi-backbone comparison mode')
    p.add_argument('--models', nargs='+', metavar='NAME:CFG:CKPT',
                   help='Models spec for --compare: name:config:checkpoint')
    p.add_argument('--compare-image', metavar='IMG',
                   help='Image path for --compare mode')

    # output
    p.add_argument('--out-dir', default='vis_output',
                   help='Output directory (default: vis_output/)')
    p.add_argument('--device',  default='cuda:0',
                   help='Inference device (default: cuda:0)')
    p.add_argument('--dpi',     type=int, default=150,
                   help='Figure DPI (default: 150)')
    return p


def _run_single(args: argparse.Namespace) -> None:
    if not all([args.config, args.checkpoint, args.image]):
        print('Error: config, checkpoint, and image are required.')
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    print(f'Loading model  …  {args.checkpoint}')
    model = load_model(args.config, args.checkpoint, args.device)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f'Error: cannot read {args.image}')
        sys.exit(1)

    print('Extracting features …')
    features, img_resized = extract_features(model, img_bgr)
    print_feature_stats(features)

    if args.mode == 'overview':
        fig = plot_feature_overview(
            features, img_resized,
            agg=args.agg, title=Path(args.config).stem, alpha=args.alpha,
        )
        out = out_dir / f'{stem}_overview_{args.agg}.png'
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved → {out}')

    elif args.mode == 'channels':
        fig = plot_channel_grid(
            features[args.layer],
            layer_name=args.layer, top_k=args.top_k,
        )
        out = out_dir / f'{stem}_{args.layer}_top{args.top_k}.png'
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved → {out}')


def _run_compare(args: argparse.Namespace) -> None:
    if not (args.models and args.compare_image):
        print('Error: --models and --compare-image are required with --compare.')
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_and_names = []
    for spec in args.models:
        parts = spec.split(':')
        if len(parts) != 3:
            print(f'Error: expected name:config:ckpt, got "{spec}"')
            sys.exit(1)
        name, cfg_path, ckpt_path = parts
        print(f'  Loading [{name}] …')
        models_and_names.append((name, load_model(cfg_path, ckpt_path, args.device)))

    img_bgr = cv2.imread(args.compare_image)
    if img_bgr is None:
        print(f'Error: cannot read {args.compare_image}')
        sys.exit(1)

    fig = plot_backbone_comparison(
        models_and_names, img_bgr,
        layer=args.layer, agg=args.agg, alpha=args.alpha,
    )
    stem = Path(args.compare_image).stem
    out  = out_dir / f'{stem}_compare_{args.layer}_{args.agg}.png'
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def main() -> None:
    # Switch to non-interactive Agg backend before any figure is created.
    # switch_backend() is safe to call after pyplot has been imported.
    plt.switch_backend('agg')
    args = _build_parser().parse_args()
    if args.compare:
        _run_compare(args)
    else:
        _run_single(args)


if __name__ == '__main__':
    main()
