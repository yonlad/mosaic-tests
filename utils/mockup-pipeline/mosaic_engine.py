"""
Reusable mosaic generation engine.

Loads thumbnails from S3 once into memory, then can generate multiple mosaics
efficiently without re-downloading. Based on the logic in mosaics-grey.py but
refactored as a class for batch use.
"""

import gc
import math
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import boto3
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

try:
    from skimage.filters import threshold_otsu, sobel
    from skimage import color
    from skimage.morphology import binary_closing, binary_opening, binary_dilation, binary_erosion, disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    threshold_otsu = None

# Default mosaic configuration matching mosaics-grey.py
DEFAULT_CONFIG = {
    "thumbnail_size": (2, 2),
    "internal_thumbnail_size": (64, 64),
    "cell_size": (2, 2),
    "brightness_threshold": 240,
    "foreground_threshold": 0.01,
    "detail_sensitivity": 2.0,
    "max_thumbnail_usage": 50,
    "max_dimension": 800,
}


class MosaicEngine:
    """Generates mosaics from source images using pre-loaded thumbnails."""

    def __init__(self, aws_region, aws_access_key_id, aws_secret_access_key,
                 s3_bucket, thumbnail_prefix="selected-images/",
                 thumbnail_limit=4000, max_concurrent_downloads=50,
                 config=None):
        self.aws_region = aws_region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_bucket = s3_bucket
        self.thumbnail_prefix = thumbnail_prefix
        self.max_concurrent_downloads = max_concurrent_downloads
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.thumbnails = []

        self._load_thumbnails(thumbnail_limit)

    def _get_s3_client(self):
        return boto3.client(
            "s3",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )

    def _download_single_image(self, key):
        """Download and process a single thumbnail from S3."""
        try:
            s3 = self._get_s3_client()
            resp = s3.get_object(Bucket=self.s3_bucket, Key=key)
            image_data = resp["Body"].read()
            img = Image.open(BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")

            internal_size = self.config["internal_thumbnail_size"]
            thumb_size = self.config["thumbnail_size"]

            hi_res = img.resize(internal_size, Image.LANCZOS)
            preview = img.resize(thumb_size, Image.BILINEAR)
            avg_color = np.mean(np.array(preview), axis=(0, 1)).astype(int)
            del preview

            return {
                "s3_key": key,
                "avg_color": avg_color,
                "image": hi_res,
                "display_size": thumb_size,
            }
        except Exception as e:
            print(f"Error downloading {key}: {e}")
            return None

    def _load_thumbnails(self, limit):
        """Fetch thumbnails from S3 into memory."""
        s3 = self._get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")

        all_keys = []
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.thumbnail_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                    all_keys.append(key)

        print(f"Found {len(all_keys)} images in s3://{self.s3_bucket}/{self.thumbnail_prefix}")

        random.shuffle(all_keys)
        selected = all_keys[:limit]

        print(f"Downloading {len(selected)} thumbnails with {self.max_concurrent_downloads} workers...")

        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            futures = {executor.submit(self._download_single_image, k): k for k in selected}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.thumbnails.append(result)
                completed += 1
                if completed % 200 == 0:
                    print(f"  Downloaded {completed}/{len(selected)}...")

        print(f"Loaded {len(self.thumbnails)} thumbnails into memory")

    def generate(self, source_image: Image.Image, background: Image.Image) -> Image.Image:
        """Create a mosaic of source_image using loaded thumbnails on the given background.

        Args:
            source_image: The photograph to turn into a mosaic.
            background: Background image (e.g. mirror texture). Resized to fit.

        Returns:
            The composited mosaic image (RGB).
        """
        cfg = self.config
        max_dim = cfg["max_dimension"]
        cell_w, cell_h = cfg["cell_size"]
        internal_size = cfg["internal_thumbnail_size"]
        max_usage = cfg["max_thumbnail_usage"]

        # Reset usage counts
        for thumb in self.thumbnails:
            thumb["usage_count"] = 0

        img = source_image.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if too large
        w, h = img.size
        if w > max_dim or h > max_dim:
            if w > h:
                new_w, new_h = max_dim, int(h * max_dim / w)
            else:
                new_h, new_w = max_dim, int(w * max_dim / h)
            img = img.resize((new_w, new_h), Image.BILINEAR)

        original_img = img.copy()

        # Detect foreground
        foreground_mask = self._detect_foreground(original_img)

        # Detect background color for halo removal
        bg_color = self._detect_background_color(original_img)

        # Edge detection for detail map
        gray = img.convert("L")
        edges_array = np.array(gray.filter(ImageFilter.FIND_EDGES))
        detail_map = ndimage.gaussian_filter(edges_array.astype(float), sigma=5.0)
        if detail_map.max() > 0:
            detail_map /= detail_map.max()

        # Grid dimensions
        n_cols = math.floor(img.width / cell_w)
        n_rows = math.floor(img.height / cell_h)
        max_cells = 10000
        if n_cols * n_rows > max_cells:
            scale = math.sqrt(max_cells / (n_cols * n_rows))
            n_cols = max(1, math.floor(n_cols * scale))
            n_rows = max(1, math.floor(n_rows * scale))

        grid_w, grid_h = n_cols * cell_w, n_rows * cell_h
        original_img = original_img.resize((grid_w, grid_h), Image.BILINEAR)
        original_array = np.array(original_img)

        # Resize foreground mask to grid
        mask_img = Image.fromarray((foreground_mask.astype(np.uint8) * 255))
        mask_resized = np.array(mask_img.resize((grid_w, grid_h), Image.NEAREST)) > 127

        # Hi-res scale
        scale_factor = internal_size[0] / cfg["thumbnail_size"][0]
        hi_w = int(grid_w * scale_factor)
        hi_h = int(grid_h * scale_factor)

        # Prepare background canvas
        bg_resized = background.resize((hi_w, hi_h), Image.LANCZOS).convert("RGB")
        mosaic = bg_resized.copy()

        # Pre-compute cell info
        bg_color_f = bg_color.astype(float)
        halo_threshold = 40.0

        cells = []
        for y in range(n_rows):
            for x in range(n_cols):
                cx, cy = x * cell_w, y * cell_h
                cell_mask = mask_resized[cy:cy + cell_h, cx:cx + cell_w]
                is_fg = np.mean(cell_mask) > cfg["foreground_threshold"]
                if not is_fg:
                    continue

                cell_region = original_array[cy:cy + cell_h, cx:cx + cell_w]
                avg_color = np.mean(cell_region, axis=(0, 1)).astype(int)

                # Halo removal
                dist = np.sqrt(np.sum((avg_color.astype(float) - bg_color_f) ** 2))
                if dist < halo_threshold:
                    continue

                cell_edges = edges_array[cy:cy + cell_h, cx:cx + cell_w] if cy + cell_h <= edges_array.shape[0] and cx + cell_w <= edges_array.shape[1] else np.zeros((cell_h, cell_w))
                detail = min(1.0, (np.mean(cell_edges) / 255.0) * cfg["detail_sensitivity"] * 2.0)

                cells.append((x, y, avg_color, detail))

        print(f"  Placing {len(cells)} foreground tiles on {n_cols}x{n_rows} grid...")

        # Place tiles
        for x, y, avg_color, detail in cells:
            best = self._find_best_match(avg_color, max_usage)
            if not best:
                continue
            best["usage_count"] += 1

            hi_x = int(x * cell_w * scale_factor)
            hi_y = int(y * cell_h * scale_factor)
            hi_x = max(0, min(hi_x, hi_w - internal_size[0]))
            hi_y = max(0, min(hi_y, hi_h - internal_size[1]))

            mosaic.paste(best["image"], (hi_x, hi_y))

        # Rotate to portrait if landscape
        if mosaic.width > mosaic.height:
            mosaic = mosaic.transpose(Image.ROTATE_270)

        return mosaic

    def _find_best_match(self, target_color, max_usage):
        """Find the thumbnail with closest average color."""
        target = target_color.astype(float)
        best = None
        best_dist = float("inf")

        if SKIMAGE_AVAILABLE:
            try:
                target_lab = color.rgb2lab(target.reshape(1, 1, 3) / 255.0).reshape(3)
                for thumb in self.thumbnails:
                    if thumb.get("usage_count", 0) >= max_usage:
                        continue
                    t_lab = color.rgb2lab(thumb["avg_color"].reshape(1, 1, 3) / 255.0).reshape(3)
                    dist = np.sum((target_lab - t_lab) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best = thumb
                return best
            except Exception:
                pass

        # Fallback: weighted RGB
        weights = np.array([0.3, 0.6, 0.1])
        for thumb in self.thumbnails:
            if thumb.get("usage_count", 0) >= max_usage:
                continue
            diff = (target - thumb["avg_color"].astype(float)) * weights
            dist = np.sum(diff ** 2)
            if dist < best_dist:
                best_dist = dist
                best = thumb
        return best

    def _detect_foreground(self, image):
        """Detect foreground mask. Returns boolean array."""
        gray = np.array(image.convert("L"))
        rgb = np.array(image)

        if SKIMAGE_AVAILABLE and threshold_otsu is not None:
            try:
                threshold = threshold_otsu(gray)
                dark_mask = gray < (threshold + 20)
                bright_threshold = min(255, threshold + 80)
                bright_mask = gray < bright_threshold

                r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
                skin1 = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (np.abs(r.astype(int) - g.astype(int)) > 15)
                skin2 = (r > 80) & (g > 50) & (b > 30) & (r > b) & (g > b) & ((r - b) > 10)
                skin_mask = skin1 | skin2

                edges = sobel(gray)
                edge_regions = binary_dilation(edges > np.percentile(edges, 60), disk(15))

                h, w = gray.shape
                yc, xc = h // 2, w // 2
                yy, xx = np.ogrid[:h, :w]
                center_dist = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
                max_dist = np.sqrt(h ** 2 + w ** 2) / 2
                center_mask = (1.0 - center_dist / max_dist) > 0.2

                combined = skin_mask | dark_mask | (bright_mask & edge_regions & center_mask)
                dark_dilated = binary_dilation(dark_mask, disk(20))
                combined = combined | (bright_mask & dark_dilated)

                mask = binary_closing(combined, disk(8))
                mask = binary_opening(mask, disk(3))
                mask = binary_erosion(mask, disk(2))
                return mask
            except Exception:
                pass

        # Simple fallback
        return gray < 240

    def _detect_background_color(self, image, margin=0.05):
        """Detect background color from image edges."""
        arr = np.array(image)
        h, w = arr.shape[:2]
        my, mx = max(1, int(h * margin)), max(1, int(w * margin))
        edge_px = np.concatenate([
            arr[:my, :].reshape(-1, 3),
            arr[-my:, :].reshape(-1, 3),
            arr[:, :mx].reshape(-1, 3),
            arr[:, -mx:].reshape(-1, 3),
        ])
        return np.median(edge_px, axis=0).astype(int)
