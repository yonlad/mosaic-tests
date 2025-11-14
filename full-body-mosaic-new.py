import os
import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image

# Optional deps
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import boto3
except Exception:
    boto3 = None

try:
    from skimage import color
    SKIMAGE_AVAILABLE = True
except Exception:
    color = None
    SKIMAGE_AVAILABLE = False


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class MosaicConfig:
    # Source
    image: str = "Jasmine copy.jpg"
    is_s3_key: bool = False

    # S3
    aws_region: str = os.getenv("REACT_APP_AWS_REGION", "us-east-2")
    aws_access_key_id: Optional[str] = os.getenv("REACT_APP_AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("REACT_APP_AWS_SECRET_ACCESS_KEY")
    s3_bucket: str = os.getenv("REACT_APP_S3_BUCKET", "eternity-mirror-project")
    s3_prefix: str = "selected-images/"
    thumbnail_limit: int = 6000
    max_concurrent_downloads: int = 24

    # Tiles
    base_cell_size: int = 80
    min_cell_size: int = 2
    max_cell_size: int = 48
    foreground_min_cell_size: int = 2
    background_min_cell_size: int = 8
    boundary_min_cell_size: int = 40  # enforce larger tiles along FG boundary to avoid micro tiles

    # Thumbnails (hi-res building blocks)
    internal_thumbnail_size: Tuple[int, int] = (128, 128)

    # Matching
    reuse_penalty: float = 0.15
    max_thumbnail_usage: int = 1000
    color_space: str = "lab"

    # Foreground handling
    use_foreground_mask: bool = True
    foreground_inclusion_bias: float = 0.15
    exclude_white_background: bool = True
    white_bg_luma_threshold: int = 245
    fg_mask_erosion_radius: int = 0  # 0 disables; >0 erodes FG to remove 1-2 px fringes
    fg_mask_fill_holes: bool = True
    fg_mask_fill_holes_max_area: int = 2500  # 0 disables size filtering

    # Adaptive detail controls
    detail_threshold: float = 0.10
    grad_weight: float = 0.7
    var_weight: float = 0.3

    # Output
    output_dir: str = "output"
    jpeg_quality: int = 80
    save_thumbnail: bool = True
    output_long_side: int = 8000
    max_source_image_pixels: Optional[int] = None  # None disables Pillow's decompression bomb guard

    # Blending
    blend_with_source: bool = True
    blend_alpha: float = 0.12


# ----------------------------
# Utilities
# ----------------------------

def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _to_lab(rgb_arr: np.ndarray) -> np.ndarray:
    if not SKIMAGE_AVAILABLE:
        return rgb_arr.astype(np.float32) / 255.0
    rgb_norm = (rgb_arr.astype(np.float32) / 255.0)[None, ...]
    lab = color.rgb2lab(rgb_norm)[0]
    return lab


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Thumb:
    s3_key: str
    image: Image.Image
    avg_color_rgb: np.ndarray
    avg_color_lab: Optional[np.ndarray]
    reuse_count: int = 0


# ----------------------------
# Thumbnail loading (S3/local)
# ----------------------------

def _new_s3_client(cfg: MosaicConfig):
    if boto3 is None:
        raise RuntimeError("boto3 not available for S3 operations")
    return boto3.client(
        "s3",
        region_name=cfg.aws_region,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
    )


def _download_single_image_s3(s3_client, bucket: str, key: str, internal_size: Tuple[int, int]) -> Optional[Thumb]:
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        img = Image.open(BytesIO(body))
        img = _ensure_rgb(img)
        hi = img.resize(internal_size, Image.LANCZOS)
        arr = np.array(hi, dtype=np.uint8)
        avg_rgb = arr.reshape(-1, 3).mean(axis=0)
        avg_lab = _to_lab(arr).reshape(-1, 3).mean(axis=0) if SKIMAGE_AVAILABLE else None
        return Thumb(s3_key=key, image=hi, avg_color_rgb=avg_rgb, avg_color_lab=avg_lab)
    except Exception:
        return None


def fetch_thumbnails_from_s3(cfg: MosaicConfig) -> List[Thumb]:
    if boto3 is None:
        print("Warning: boto3 not available. S3 fetch disabled.")
        return []

    s3_client = _new_s3_client(cfg)

    keys: List[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg.s3_bucket, Prefix=cfg.s3_prefix):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                keys.append(key)
    if not keys:
        return []

    rng = np.random.default_rng(42)
    rng.shuffle(keys)
    keys = keys[: cfg.thumbnail_limit]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    thumbs: List[Thumb] = []
    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_downloads) as ex:
        futures = {ex.submit(_download_single_image_s3, s3_client, cfg.s3_bucket, k, cfg.internal_thumbnail_size): k for k in keys}
        completed = 0
        for fut in as_completed(futures):
            t = fut.result()
            if t is not None:
                thumbs.append(t)
            completed += 1
            if completed % 100 == 0:
                print(f"Downloaded {completed}/{len(futures)} thumbnails...")

    print(f"Collected {len(thumbs)} thumbnails")
    return thumbs


# ----------------------------
# Source image loading
# ----------------------------

def load_source_image(cfg: MosaicConfig) -> Optional[Image.Image]:
    if cfg.is_s3_key:
        if boto3 is None:
            print("boto3 not available; cannot load source from S3")
            return None
        try:
            s3 = _new_s3_client(cfg)
            data = s3.get_object(Bucket=cfg.s3_bucket, Key=cfg.image)["Body"].read()
            img = Image.open(BytesIO(data))
            return _ensure_rgb(img)
        except Exception as e:
            print(f"Failed to load source from S3: {e}")
            return None
    else:
        try:
            img = Image.open(cfg.image)
            return _ensure_rgb(img)
        except Exception as e:
            print(f"Failed to load local image: {e}")
            return None


# ----------------------------
# Foreground estimation with white background exclusion and optional erosion
# ----------------------------

def detect_foreground_mask(src_rgb: np.ndarray, cfg: MosaicConfig) -> np.ndarray:
    if not cfg.use_foreground_mask:
        return np.ones((src_rgb.shape[0], src_rgb.shape[1]), dtype=bool)

    gray = (0.299 * src_rgb[..., 0] + 0.587 * src_rgb[..., 1] + 0.114 * src_rgb[..., 2]).astype(np.float32)
    h, w = gray.shape

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.hypot(gx, gy)

    bright_thresh = np.percentile(gray, 60)
    bright_mask = gray >= bright_thresh

    if cfg.exclude_white_background:
        white_bg = gray >= float(cfg.white_bg_luma_threshold)
    else:
        white_bg = np.zeros_like(gray, dtype=bool)

    y_grid, x_grid = np.ogrid[:h, :w]
    yc, xc = h / 2.0, w / 2.0
    dist = np.sqrt((y_grid - yc) ** 2 + (x_grid - xc) ** 2)
    center = 1.0 - (dist / (dist.max() + 1e-6))
    center_mask = center > 0.25

    edge_mask = grad > np.percentile(grad, 70)

    mask = ((bright_mask & center_mask) | edge_mask) & (~white_bg)

    inflated = np.zeros_like(mask)
    inflated[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        | mask[:-2, 1:-1]
        | mask[2:, 1:-1]
        | mask[1:-1, :-2]
        | mask[1:-1, 2:]
    )
    final_mask = (inflated | mask) & (~white_bg)

    # Optional erosion to remove 1-2 px fringes along silhouette
    if cfg.fg_mask_erosion_radius > 0:
        r = int(cfg.fg_mask_erosion_radius)
        if r > 0:
            eroded = np.zeros_like(final_mask)
            # simple square structuring element erosion
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = np.zeros_like(final_mask)
                    src = final_mask
                    y1 = max(0, 0 - dy)
                    y2 = min(final_mask.shape[0], final_mask.shape[0] - dy)
                    x1 = max(0, 0 - dx)
                    x2 = min(final_mask.shape[1], final_mask.shape[1] - dx)
                    shifted[y1:y2, x1:x2] = src[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
                    if dy == -r and dx == -r:
                        eroded = shifted
                    else:
                        eroded = eroded & shifted if (dy != -r or dx != -r) else shifted
            final_mask = final_mask & eroded

    if cfg.fg_mask_fill_holes:
        final_mask = fill_mask_holes(final_mask, cfg.fg_mask_fill_holes_max_area)

    return final_mask


def fill_mask_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    q: deque[Tuple[int, int]] = deque()

    def enqueue(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and not mask[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    for x in range(w):
        enqueue(0, x)
        enqueue(h - 1, x)
    for y in range(h):
        enqueue(y, 0)
        enqueue(y, w - 1)

    while q:
        y, x = q.popleft()
        outside[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            enqueue(ny, nx)

    filled = mask.copy()
    holes = (~mask) & (~outside)
    if not np.any(holes):
        return filled

    if max_area <= 0:
        filled[holes] = True
        return filled

    visited_holes = np.zeros_like(mask, dtype=bool)
    hole_indices = np.argwhere(holes)
    for y, x in hole_indices:
        if visited_holes[y, x]:
            continue
        component: List[Tuple[int, int]] = []
        q = deque([(y, x)])
        visited_holes[y, x] = True
        while q:
            cy, cx = q.popleft()
            component.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and holes[ny, nx] and not visited_holes[ny, nx]:
                    visited_holes[ny, nx] = True
                    q.append((ny, nx))
        if len(component) <= max_area:
            for cy, cx in component:
                filled[cy, cx] = True

    return filled

# ----------------------------
# Integral images and region stats
# ----------------------------

def _integral(img: np.ndarray) -> np.ndarray:
    return img.cumsum(axis=0).cumsum(axis=1)


def _sum_rect(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    x2 = x + w - 1
    y2 = y + h - 1
    s = ii[y2, x2]
    if x > 0:
        s -= ii[y2, x - 1]
    if y > 0:
        s -= ii[y - 1, x2]
    if x > 0 and y > 0:
        s += ii[y - 1, x - 1]
    return float(s)


def build_integral_images(src_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    gray = (0.299 * src_rgb[..., 0] + 0.587 * src_rgb[..., 1] + 0.114 * src_rgb[..., 2]).astype(np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.hypot(gx, gy)

    ii_1 = _integral(np.ones_like(gray, dtype=np.float32))
    ii_gray = _integral(gray)
    ii_gray2 = _integral(gray * gray)
    ii_grad = _integral(grad)

    return {
        "ii_1": ii_1,
        "ii_gray": ii_gray,
        "ii_gray2": ii_gray2,
        "ii_grad": ii_grad,
    }


def region_stats(ii: Dict[str, np.ndarray], x: int, y: int, w: int, h: int) -> Tuple[float, float, float]:
    n = _sum_rect(ii["ii_1"], x, y, w, h)
    if n <= 0:
        return 0.0, 0.0, 0.0
    s = _sum_rect(ii["ii_gray"], x, y, w, h)
    s2 = _sum_rect(ii["ii_gray2"], x, y, w, h)
    sg = _sum_rect(ii["ii_grad"], x, y, w, h)

    mean = s / n
    var = max(0.0, (s2 / n) - (mean * mean))
    mean_grad = sg / n
    return mean, var, mean_grad


# ----------------------------
# Adaptive quadtree tiling with boundary guard
# ----------------------------

def adaptive_tile_quadtree(src_rgb: np.ndarray, cfg: MosaicConfig, fg_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
    h, w, _ = src_rgb.shape
    tiles: List[Tuple[int, int, int, int]] = []
    ii = build_integral_images(src_rgb)

    if fg_mask is None:
        fg_mask = np.ones((h, w), dtype=bool)

    fg_float = fg_mask.astype(np.float32)
    ii_fg = _integral(fg_float)

    def fg_ratio(x: int, y: int, tw: int, th: int) -> float:
        count = _sum_rect(ii["ii_1"], x, y, tw, th)
        if count <= 0:
            return 0.0
        fgs = _sum_rect(ii_fg, x, y, tw, th)
        return float(fgs / count)

    min_fore = max(cfg.min_cell_size, cfg.foreground_min_cell_size)
    min_back = max(cfg.min_cell_size, cfg.background_min_cell_size)

    def split(x: int, y: int, tw: int, th: int, depth: int) -> None:
        fr = fg_ratio(x, y, tw, th)
        if fr < 0.01:
            return

        # Boundary guard: if region straddles FG boundary, enforce a larger minimum
        # by checking if interior is mixed FG/BG.
        is_boundary = (fr > 0.0) and (fr < 1.0)
        boundary_min = max(cfg.boundary_min_cell_size, min_fore)

        min_sz = int(min_fore if fr >= cfg.foreground_inclusion_bias else min_back)
        if is_boundary:
            min_sz = max(min_sz, boundary_min)

        if tw <= min_sz or th <= min_sz:
            tiles.append((x, y, tw, th))
            return
        if tw <= cfg.min_cell_size and th <= cfg.min_cell_size:
            tiles.append((x, y, tw, th))
            return

        _, var, mean_grad = region_stats(ii, x, y, tw, th)
        detail = cfg.grad_weight * (mean_grad / (mean_grad + 1e-6)) + cfg.var_weight * (var / (var + 50.0))

        if detail < cfg.detail_threshold or (tw <= cfg.max_cell_size and th <= cfg.max_cell_size and (tw <= min_sz * 2 or th <= min_sz * 2)):
            tiles.append((x, y, tw, th))
            return

        hw = max(min_sz, tw // 2)
        hh = max(min_sz, th // 2)
        if hw == tw and hh == th:
            tiles.append((x, y, tw, th))
            return

        split(x, y, hw, hh, depth + 1)
        if x + hw < w:
            split(x + hw, y, min(tw - hw, w - (x + hw)), hh, depth + 1)
        if y + hh < h:
            split(x, y + hh, hw, min(th - hh, h - (y + hh)), depth + 1)
        if x + hw < w and y + hh < h:
            split(x + hw, y + hh, min(tw - hw, w - (x + hw)), min(th - hh, h - (y + hh)), depth + 1)

    step = max(cfg.base_cell_size, cfg.min_cell_size)
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            tw = min(step, w - x0)
            th = min(step, h - y0)
            split(x0, y0, tw, th, 0)

    return tiles


# ----------------------------
# Matching and placement with usage cap
# ----------------------------

def match_and_place(src_rgb: np.ndarray, tiles: List[Tuple[int, int, int, int]], thumbs: List[Thumb], cfg: MosaicConfig) -> Image.Image:
    long_side = max(src_rgb.shape[1], src_rgb.shape[0])
    scale_block = cfg.internal_thumbnail_size[0] / max(cfg.min_cell_size, 1)
    scale_output = cfg.output_long_side / max(1, long_side)
    scale = min(scale_block, scale_output)

    out_w = max(1, int(src_rgb.shape[1] * scale))
    out_h = max(1, int(src_rgb.shape[0] * scale))
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    # Precompute palette in chosen color space
    if cfg.color_space.lower() == "lab" and SKIMAGE_AVAILABLE and all(t.avg_color_lab is not None for t in thumbs):
        thumb_colors = np.stack([t.avg_color_lab for t in thumbs], axis=0)
        thumb_list = thumbs
    else:
        thumb_colors = np.stack([t.avg_color_rgb for t in thumbs], axis=0)
        thumb_list = thumbs

    # Track reuse penalty and hard usage cap
    reuse_counts = np.zeros(len(thumb_list), dtype=np.int32)

    for (x, y, w, h) in tiles:
        patch = src_rgb[y : y + h, x : x + w]
        patch_avg = patch.reshape(-1, 3).mean(axis=0)
        if cfg.color_space.lower() == "lab" and SKIMAGE_AVAILABLE:
            patch_lab = _to_lab(patch.astype(np.uint8)).reshape(-1, 3).mean(axis=0)
            target = patch_lab
        else:
            target = patch_avg

        # compute distances
        d = ((thumb_colors - target[None, :]) ** 2).sum(axis=1)
        # apply reuse penalty
        d = d + cfg.reuse_penalty * reuse_counts.astype(np.float32)

        # enforce hard cap by inflating distance for exhausted tiles
        exhausted = reuse_counts >= int(cfg.max_thumbnail_usage)
        if np.any(exhausted):
            d[exhausted] = d[exhausted] + 1e9

        idx = int(np.argmin(d))
        chosen = thumb_list[idx]
        reuse_counts[idx] += 1

        # placement size proportional to tile size
        ox = int(x * scale)
        oy = int(y * scale)
        tw_px = max(1, int(w * scale))
        th_px = max(1, int(h * scale))
        if chosen.image.size != (tw_px, th_px):
            tile_img = chosen.image.resize((tw_px, th_px), Image.LANCZOS)
        else:
            tile_img = chosen.image
        canvas.paste(tile_img, (ox, oy))

    # Optional blending with upscaled source to reduce blockiness
    if cfg.blend_with_source:
        src_img = Image.fromarray(src_rgb)
        src_up = src_img.resize((out_w, out_h), Image.BILINEAR)
        canvas = Image.blend(src_up, canvas, alpha=1.0 - cfg.blend_alpha)

    return canvas


# ----------------------------
# Orchestration
# ----------------------------

def run(cfg: MosaicConfig) -> Optional[Image.Image]:
    start = time.time()

    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    if cfg.max_source_image_pixels is None or cfg.max_source_image_pixels < 0:
        Image.MAX_IMAGE_PIXELS = None
    else:
        Image.MAX_IMAGE_PIXELS = cfg.max_source_image_pixels

    try:
        src_img = load_source_image(cfg)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
    if src_img is None:
        return None

    # Downscale source for tractability if huge; target longest side ~1000 px
    w, h = src_img.size
    max_side = max(w, h)
    if max_side > 1200:
        scale = 1200.0 / max_side
        nw, nh = int(w * scale), int(h * scale)
        src_img = src_img.resize((nw, nh), Image.BILINEAR)
        w, h = nw, nh

    src_rgb = np.array(src_img, dtype=np.uint8)

    thumbs = fetch_thumbnails_from_s3(cfg)
    if not thumbs:
        print("No thumbnails available.")
        return None

    fg_mask = detect_foreground_mask(src_rgb, cfg)

    tiles = adaptive_tile_quadtree(src_rgb, cfg, fg_mask)
    mosaic = match_and_place(src_rgb, tiles, thumbs, cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"mosaic_new_{ts}.jpg")
    mosaic.save(out_path, format="JPEG", quality=cfg.jpeg_quality, optimize=True, progressive=True)

    if cfg.save_thumbnail:
        thumb = mosaic.copy()
        thumb.thumbnail((800, 1200), Image.LANCZOS)
        thumb_path = os.path.join(cfg.output_dir, f"mosaic_new_thumb_{ts}.jpg")
        thumb.save(thumb_path, format="JPEG", quality=70, optimize=True, progressive=True)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.2f}s -> {out_path}")
    return mosaic


# ----------------------------
# CLI
# ----------------------------

def _parse_args() -> MosaicConfig:
    p = argparse.ArgumentParser(description="Adaptive full-body mosaic generator (new)")
    p.add_argument("--image", type=str, default=MosaicConfig.image)
    p.add_argument("--is-s3-key", action="store_true")
    p.add_argument("--s3-prefix", type=str, default=MosaicConfig.s3_prefix)
    p.add_argument("--thumbnail-limit", type=int, default=MosaicConfig.thumbnail_limit)
    p.add_argument("--max-concurrent-downloads", type=int, default=MosaicConfig.max_concurrent_downloads)
    p.add_argument("--base-cell-size", type=int, default=MosaicConfig.base_cell_size)
    p.add_argument("--min-cell-size", type=int, default=MosaicConfig.min_cell_size)
    p.add_argument("--max-cell-size", type=int, default=MosaicConfig.max_cell_size)
    p.add_argument("--foreground-min-cell-size", type=int, default=MosaicConfig.foreground_min_cell_size)
    p.add_argument("--background-min-cell-size", type=int, default=MosaicConfig.background_min_cell_size)
    p.add_argument("--boundary-min-cell-size", type=int, default=MosaicConfig.boundary_min_cell_size)
    p.add_argument("--internal-thumbnail-size", type=int, nargs=2, default=list(MosaicConfig.internal_thumbnail_size))
    p.add_argument("--reuse-penalty", type=float, default=MosaicConfig.reuse_penalty)
    p.add_argument("--max-thumbnail-usage", type=int, default=MosaicConfig.max_thumbnail_usage)
    p.add_argument("--color-space", type=str, default=MosaicConfig.color_space, choices=["lab", "rgb"])
    p.add_argument("--exclude-white-background", action="store_true", default=MosaicConfig.exclude_white_background)
    p.add_argument("--white-bg-luma-threshold", type=int, default=MosaicConfig.white_bg_luma_threshold)
    p.add_argument("--fg-mask-erosion-radius", type=int, default=MosaicConfig.fg_mask_erosion_radius)
    p.add_argument("--no-fg-mask-fill-holes", dest="fg_mask_fill_holes", action="store_false")
    p.add_argument("--fg-mask-fill-holes", dest="fg_mask_fill_holes", action="store_true")
    p.set_defaults(fg_mask_fill_holes=MosaicConfig.fg_mask_fill_holes)
    p.add_argument(
        "--fg-mask-fill-holes-max-area",
        type=int,
        default=MosaicConfig.fg_mask_fill_holes_max_area,
        help="Only fill foreground holes with pixel area <= this (set <=0 to fill everything)",
    )
    p.add_argument("--detail-threshold", type=float, default=MosaicConfig.detail_threshold)
    p.add_argument("--grad-weight", type=float, default=MosaicConfig.grad_weight)
    p.add_argument("--var-weight", type=float, default=MosaicConfig.var_weight)
    p.add_argument("--output-dir", type=str, default=MosaicConfig.output_dir)
    p.add_argument("--jpeg-quality", type=int, default=MosaicConfig.jpeg_quality)
    p.add_argument("--save-thumbnail", action="store_true", default=MosaicConfig.save_thumbnail)
    p.add_argument("--no-blend", action="store_true")
    p.add_argument("--blend-alpha", type=float, default=MosaicConfig.blend_alpha)
    p.add_argument("--output-long-side", type=int, default=MosaicConfig.output_long_side)
    p.add_argument(
        "--max-source-image-pixels",
        type=int,
        default=MosaicConfig.max_source_image_pixels,
        help="Override Pillow's decompression bomb limit for the source image (set to -1 to disable)",
    )
    args = p.parse_args()

    cfg = MosaicConfig(
        image=args.image,
        is_s3_key=args.is_s3_key,
        s3_prefix=args.s3_prefix,
        thumbnail_limit=args.thumbnail_limit,
        max_concurrent_downloads=args.max_concurrent_downloads,
        base_cell_size=args.base_cell_size,
        min_cell_size=args.min_cell_size,
        max_cell_size=args.max_cell_size,
        foreground_min_cell_size=args.foreground_min_cell_size,
        background_min_cell_size=args.background_min_cell_size,
        boundary_min_cell_size=args.boundary_min_cell_size,
        internal_thumbnail_size=tuple(args.internal_thumbnail_size),
        reuse_penalty=args.reuse_penalty,
        max_thumbnail_usage=args.max_thumbnail_usage,
        color_space=args.color_space,
        exclude_white_background=args.exclude_white_background,
        white_bg_luma_threshold=args.white_bg_luma_threshold,
        fg_mask_erosion_radius=args.fg_mask_erosion_radius,
        fg_mask_fill_holes=args.fg_mask_fill_holes,
        fg_mask_fill_holes_max_area=args.fg_mask_fill_holes_max_area,
        detail_threshold=args.detail_threshold,
        grad_weight=args.grad_weight,
        var_weight=args.var_weight,
        output_dir=args.output_dir,
        jpeg_quality=args.jpeg_quality,
        save_thumbnail=args.save_thumbnail,
        blend_with_source=(not args.no_blend),
        blend_alpha=args.blend_alpha,
        output_long_side=args.output_long_side,
        max_source_image_pixels=args.max_source_image_pixels,
    )
    return cfg


def main() -> None:
    cfg = _parse_args()
    # Print config summary
    print(json.dumps(asdict(cfg), indent=2, default=str))
    run(cfg)


if __name__ == "__main__":
    main()
