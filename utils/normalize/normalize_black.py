"""
Batch image normalization pipeline for S3 — BLACK BACKGROUND variant.

Same normalization as normalize.py but adapted for images with black
backgrounds instead of white:
  - Body detection finds "non-black" pixels (bright pixels above threshold)
  - Output canvas uses white background for consistency

Usage
-----
  # Dry run with debug
  python normalize_black.py --limit 3 --dry-run --debug

  # Full run on sanitized bucket
  python normalize_black.py --bucket pistoletto.sanitized \
    --prefix selected-images/resized_black_bg/ \
    --dest-prefix normalized/ --head-position 0.40
"""

import os
import json
import argparse
import logging
import time
import threading
from pathlib import Path
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from pose_utils import (
    detect_pose,
    estimate_head_top_y,
    estimate_person_center_x,
)

# ── Config ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", os.getenv("REACT_APP_AWS_REGION", "us-east-2"))
AWS_ACCESS_KEY_ID = os.getenv(
    "AWS_ACCESS_KEY_ID", os.getenv("REACT_APP_AWS_ACCESS_KEY_ID")
)
AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY", os.getenv("REACT_APP_AWS_SECRET_ACCESS_KEY")
)

DEFAULT_BUCKET = "pistoletto.sanitized"
DEFAULT_PREFIX = "selected-images/resized_black_bg/"
DEFAULT_DEST_PREFIX = "normalized/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
JPEG_QUALITY = 95

SCRIPT_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("normalize_black")

# Thread-local S3 clients
_thread_local = threading.local()


def _get_s3_client():
    """Get or create a thread-local S3 client."""
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return _thread_local.s3


# ── S3 Scanning ─────────────────────────────────────────────────────────────


def scan_source_images(bucket, prefix):
    """Find all images under the given prefix."""
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            ext = os.path.splitext(key)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                keys.append(key)

    return sorted(keys)


# ── Body Bounds Detection (BLACK background) ──────────────────────────────


def detect_body_bounds(pil_image, black_threshold=30):
    """
    Find bounding box of the person using non-black pixel detection.

    For black background images, a pixel is "non-black" (i.e. part of the
    person) if ANY channel is ABOVE the threshold.

    Returns: (top_y, bottom_y, left_x, right_x) in pixels,
             or None if no person found.
    """
    rgb = np.array(pil_image.convert("RGB"))

    # A pixel is "non-black" if ANY channel is above threshold
    non_black = np.any(rgb > black_threshold, axis=2)

    if not non_black.any():
        return None

    rows_with_content = np.any(non_black, axis=1)
    cols_with_content = np.any(non_black, axis=0)

    row_indices = np.where(rows_with_content)[0]
    col_indices = np.where(cols_with_content)[0]

    top_y = int(row_indices[0])
    bottom_y = int(row_indices[-1])
    left_x = int(col_indices[0])
    right_x = int(col_indices[-1])

    return (top_y, bottom_y, left_x, right_x)


# ── Core Normalization ──────────────────────────────────────────────────────


def normalize_image(pil_image, canvas_w=2100, canvas_h=3000,
                    crown_margin=2.6, black_threshold=30,
                    head_position=0.40):
    """
    Normalize a participant image on a black background so that:
    - Head top (scalp) is at head_position fraction from top of canvas
    - Body bottom is flush with canvas bottom
    - Person is horizontally centered
    - Output is canvas_w x canvas_h on WHITE background

    Returns: (canvas_pil, metadata_dict) or (None, error_dict)
    """
    img_w, img_h = pil_image.size

    # Step 1: Detect body bounds via thresholding (non-black pixels)
    bounds = detect_body_bounds(pil_image, black_threshold=black_threshold)
    if bounds is None:
        return None, {"error": "no_person_detected", "method": "threshold_empty"}

    thresh_top_y, thresh_bottom_y, thresh_left_x, thresh_right_x = bounds

    # Step 2: Detect head top via pose estimation
    landmarks = detect_pose(pil_image)
    head_top_y = None
    person_center_x = None
    detection_method = "threshold_only"

    if landmarks is not None:
        head_top_y, method = estimate_head_top_y(landmarks, img_h, crown_margin)
        if head_top_y is not None:
            detection_method = method
            person_center_x = estimate_person_center_x(landmarks, img_w)

    # Step 3: Fallback if pose detection failed
    if head_top_y is None:
        person_height_thresh = thresh_bottom_y - thresh_top_y
        crown_margin_px = person_height_thresh * 0.03
        head_top_y = max(0, thresh_top_y - crown_margin_px)
        detection_method = "threshold_only"

    if person_center_x is None:
        person_center_x = (thresh_left_x + thresh_right_x) / 2

    body_bottom_y = thresh_bottom_y

    # Step 4: Compute scale factor
    person_height = body_bottom_y - head_top_y
    if person_height <= 0:
        return None, {"error": "invalid_person_height", "person_height": person_height}

    target_person_height = canvas_h * (1.0 - head_position)
    scale_factor = target_person_height / person_height

    # Step 5: Resize the original image
    new_w = round(img_w * scale_factor)
    new_h = round(img_h * scale_factor)
    if new_w < 1 or new_h < 1:
        return None, {"error": "invalid_scale", "scale_factor": scale_factor}

    resized = pil_image.resize((new_w, new_h), Image.LANCZOS)

    # Step 6: Compute placement on black canvas
    scaled_body_bottom_y = body_bottom_y * scale_factor
    scaled_center_x = person_center_x * scale_factor

    paste_y = round(canvas_h - scaled_body_bottom_y)
    paste_x = round(canvas_w / 2 - scaled_center_x)

    # Step 7: Composite onto black canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    canvas.paste(resized, (paste_x, paste_y))

    # Step 9: Compute verification metrics
    scaled_head_top_y = head_top_y * scale_factor
    actual_head_top_on_canvas = paste_y + scaled_head_top_y
    expected_head_top = canvas_h * head_position
    deviation_pct = abs(actual_head_top_on_canvas - expected_head_top) / canvas_h * 100

    # QA flags
    review_reasons = []
    if scale_factor > 3.0:
        review_reasons.append("extreme_upscale")
    if scale_factor < 0.3:
        review_reasons.append("extreme_downscale")
    if detection_method == "threshold_only":
        review_reasons.append("no_pose_detected")
    if paste_x + new_w < 0 or paste_x > canvas_w:
        review_reasons.append("person_outside_canvas")

    metadata = {
        "original_size": (img_w, img_h),
        "scale_factor": round(scale_factor, 4),
        "person_height_original_px": round(person_height),
        "head_top_on_canvas_y": round(actual_head_top_on_canvas, 1),
        "expected_head_top_y": round(expected_head_top, 1),
        "deviation_percent": round(deviation_pct, 3),
        "detection_method": detection_method,
        "paste_offset": (paste_x, paste_y),
        "bg_conversion": "black_to_white",
        "review_needed": review_reasons if review_reasons else None,
    }

    return canvas, metadata


# ── Debug Visualization ─────────────────────────────────────────────────────


def draw_debug(original, normalized, metadata, canvas_h, head_position=0.40):
    """
    Create a side-by-side debug visualization.
    Left: original with annotations. Right: normalized with head_position guideline.
    """
    orig_w, orig_h = original.size
    norm_w, norm_h = normalized.size

    scale = norm_h / orig_h
    disp_orig_w = round(orig_w * scale)
    disp_orig = original.resize((disp_orig_w, norm_h), Image.LANCZOS)

    gap = 20
    total_w = disp_orig_w + gap + norm_w
    debug_canvas = Image.new("RGB", (total_w, norm_h + 40), (40, 40, 40))

    debug_canvas.paste(disp_orig, (0, 40))
    debug_canvas.paste(normalized, (disp_orig_w + gap, 40))

    draw = ImageDraw.Draw(debug_canvas)

    guideline_y = round(canvas_h * head_position) + 40
    right_x_start = disp_orig_w + gap
    draw.line(
        [(right_x_start, guideline_y), (right_x_start + norm_w, guideline_y)],
        fill="red", width=2,
    )

    method = metadata.get("detection_method", "?")
    sf = metadata.get("scale_factor", 0)
    dev = metadata.get("deviation_percent", 0)
    orig_size = metadata.get("original_size", (0, 0))
    label = f"Method: {method} | Scale: {sf:.2f}x | Dev: {dev:.2f}% | Orig: {orig_size[0]}x{orig_size[1]} | BG: black→white"
    draw.text((10, 10), label, fill="white")

    draw.text((right_x_start + 5, guideline_y - 15), f"{head_position:.0%} line", fill="red")

    return debug_canvas


# ── Per-Image Processing ────────────────────────────────────────────────────


def process_one(src_key, bucket, dest_prefix, args):
    """Process a single image. Returns result dict."""
    result = {
        "src_key": src_key,
        "dest_key": None,
        "status": "error",
        "error": None,
        "metadata": None,
    }

    s3 = _get_s3_client()

    dest_key = f"{dest_prefix}{src_key}"
    dest_key = os.path.splitext(dest_key)[0] + ".jpg"
    result["dest_key"] = dest_key

    try:
        # Skip if already processed
        if not args.dry_run:
            try:
                s3.head_object(Bucket=bucket, Key=dest_key)
                result["status"] = "skipped"
                return result
            except ClientError:
                pass

        # Download
        response = s3.get_object(Bucket=bucket, Key=src_key)
        image_data = response["Body"].read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")

        # Normalize
        normalized, metadata = normalize_image(
            pil_image,
            canvas_w=args.output_width,
            canvas_h=args.output_height,
            crown_margin=args.crown_margin,
            black_threshold=args.black_threshold,
            head_position=args.head_position,
        )

        if normalized is None:
            result["error"] = metadata.get("error", "normalization_failed")
            result["metadata"] = metadata
            return result

        result["metadata"] = metadata

        # Encode as JPEG
        jpeg_buf = BytesIO()
        normalized.save(jpeg_buf, format="JPEG", quality=JPEG_QUALITY)
        jpeg_bytes = jpeg_buf.getvalue()

        if args.dry_run:
            dry_dir = SCRIPT_DIR / "dry_run_output"
            dry_dir.mkdir(exist_ok=True)
            flat_name = src_key.replace("/", "__")
            flat_name = os.path.splitext(flat_name)[0] + ".jpg"
            (dry_dir / flat_name).write_bytes(jpeg_bytes)
            result["status"] = "dry-run"
            result["local_path"] = str(dry_dir / flat_name)

            if args.debug:
                debug_img = draw_debug(
                    pil_image, normalized, metadata,
                    args.output_height, args.head_position,
                )
                debug_name = os.path.splitext(flat_name)[0] + "_debug.jpg"
                debug_buf = BytesIO()
                debug_img.save(debug_buf, format="JPEG", quality=90)
                (dry_dir / debug_name).write_bytes(debug_buf.getvalue())
        else:
            s3.put_object(
                Bucket=bucket, Key=dest_key,
                Body=jpeg_bytes, ContentType="image/jpeg",
            )
            result["status"] = "ok"

    except Exception as exc:
        result["error"] = str(exc)
        logger.error(f"Error processing {src_key}: {exc}")

    return result


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Normalize black-background images to consistent framing (white output)"
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET,
                        help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"S3 prefix to scan (default: {DEFAULT_PREFIX})")
    parser.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX,
                        help=f"Destination prefix (default: {DEFAULT_DEST_PREFIX})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max images to process (0 = all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Thread pool size (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Save locally instead of uploading to S3")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug visualization images")
    parser.add_argument("--output-width", type=int, default=2100,
                        help="Output canvas width in pixels (default: 2100)")
    parser.add_argument("--output-height", type=int, default=3000,
                        help="Output canvas height in pixels (default: 3000)")
    parser.add_argument("--crown-margin", type=float, default=2.6,
                        help="Crown margin multiplier (default: 2.6)")
    parser.add_argument("--black-threshold", type=int, default=30,
                        help="Non-black pixel threshold (default: 30)")
    parser.add_argument("--head-position", type=float, default=0.40,
                        help="Scalp position as fraction from top (default: 0.40)")
    args = parser.parse_args()

    logger.info(f"Scanning s3://{args.bucket}/{args.prefix}*...")
    all_keys = scan_source_images(args.bucket, args.prefix)
    logger.info(f"Found {len(all_keys)} source images")

    if args.limit > 0:
        all_keys = all_keys[:args.limit]
        logger.info(f"Limited to {len(all_keys)} images")

    if not all_keys:
        logger.warning("No images to process")
        return

    logger.info(
        f"Processing {len(all_keys)} images → "
        f"{args.output_width}x{args.output_height} canvas | "
        f"{'DRY RUN' if args.dry_run else f's3://{args.bucket}/{args.dest_prefix}'}"
    )

    results = []
    start_time = time.time()

    if HAS_TQDM:
        progress = tqdm(total=len(all_keys), desc="Normalizing (black bg)", unit="img")
    else:
        progress = None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, key, args.bucket, args.dest_prefix, args): key
            for key in all_keys
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            status = result["status"]
            if status == "error":
                logger.warning(f"FAIL {result['src_key']}: {result['error']}")

            if progress:
                progress.update(1)
                progress.set_postfix_str(f"last: {status}")

    if progress:
        progress.close()

    elapsed = time.time() - start_time

    # ── Summary ──
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    pose_detected = sum(
        1 for r in results
        if r.get("metadata") and r["metadata"].get("detection_method") != "threshold_only"
    )
    threshold_only = sum(
        1 for r in results
        if r.get("metadata") and r["metadata"].get("detection_method") == "threshold_only"
    )
    scale_factors = [
        r["metadata"]["scale_factor"]
        for r in results if r.get("metadata") and "scale_factor" in r["metadata"]
    ]
    deviations = [
        r["metadata"]["deviation_percent"]
        for r in results if r.get("metadata") and "deviation_percent" in r["metadata"]
    ]
    review_needed = [
        r for r in results
        if r.get("metadata") and r["metadata"].get("review_needed")
    ]

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.1f}s")
    logger.info(f"  Status: {counts}")
    logger.info(f"  Pose detected: {pose_detected}, Threshold-only: {threshold_only}")
    if scale_factors:
        logger.info(f"  Scale factor: avg={sum(scale_factors)/len(scale_factors):.2f}, "
                     f"min={min(scale_factors):.2f}, max={max(scale_factors):.2f}")
    if deviations:
        logger.info(f"  Deviation: avg={sum(deviations)/len(deviations):.3f}%, "
                     f"max={max(deviations):.3f}%")
    if review_needed:
        logger.info(f"  Flagged for review: {len(review_needed)}")
        for r in review_needed:
            logger.info(f"    {r['src_key']}: {r['metadata']['review_needed']}")

    # ── Write JSON log ──
    log_data = {
        "started_at": datetime.now().astimezone().isoformat(),
        "bucket": args.bucket,
        "source_prefix": args.prefix,
        "dest_prefix": args.dest_prefix,
        "output_dimensions": [args.output_width, args.output_height],
        "crown_margin": args.crown_margin,
        "head_position": args.head_position,
        "black_threshold": args.black_threshold,
        "total_scanned": len(all_keys),
        "elapsed_seconds": round(elapsed, 1),
        "summary": {
            **counts,
            "pose_detected": pose_detected,
            "threshold_fallback": threshold_only,
            "avg_scale_factor": round(sum(scale_factors) / len(scale_factors), 3) if scale_factors else None,
            "avg_deviation_pct": round(sum(deviations) / len(deviations), 4) if deviations else None,
        },
        "results": results,
    }

    log_name = f"normalize_black_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path = SCRIPT_DIR / log_name
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
