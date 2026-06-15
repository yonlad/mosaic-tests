"""
Phase 1: Generate mockup images by replacing white mosaic backgrounds with mirror texture.

Four methods available:
  --method rembg        Use rembg on source photo for a clean person mask (default)
  --method flood-fill   Remove connected white background from mosaic edges
  --method threshold    Make ALL white pixels transparent (simple global threshold)
  --method regenerate   Re-create the mosaic from scratch with mirror background
"""

import argparse
import csv
import io
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MOCKUP_S3_BUCKET = os.getenv("MOCKUP_S3_BUCKET", "pistoletto.moe.mockups")

S3_BASE_URL = f"https://s3.{AWS_REGION}.amazonaws.com"

MIRROR_BG_PATH = Path(__file__).resolve().parent.parent / "backgrounds" / "mirror-background.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def parse_name_from_email(email: str) -> str:
    """Extract a first name from an email address."""
    local = email.split("@")[0] if "@" in email else email
    tokens = re.split(r"[._\-]", local)
    first = tokens[0] if tokens else ""
    if not first or first.isdigit() or len(first) <= 1:
        return "there"
    return first.title()


# ── Method: flood-fill ─────────────────────────────────────────────────────

def remove_white_background_flood(mosaic: Image.Image, threshold: int) -> Image.Image:
    """Remove connected white background from image edges via flood-fill.

    Preserves white pixels inside the mosaic (grid lines between tiles)
    so only the outer background is replaced.
    """
    rgba = mosaic.convert("RGBA")
    data = np.array(rgba)

    white_mask = (data[:, :, 0] > threshold) & (data[:, :, 1] > threshold) & (data[:, :, 2] > threshold)
    labeled, _ = ndimage.label(white_mask)

    border_labels = set()
    border_labels.update(labeled[0, :].flat)
    border_labels.update(labeled[-1, :].flat)
    border_labels.update(labeled[:, 0].flat)
    border_labels.update(labeled[:, -1].flat)
    border_labels.discard(0)

    bg_mask = np.isin(labeled, list(border_labels))
    data[bg_mask, 3] = 0

    return Image.fromarray(data, "RGBA")


# ── Method: threshold (global white removal) ───────────────────────────────

def remove_white_background_threshold(mosaic: Image.Image, threshold: int) -> Image.Image:
    """Make ALL white pixels transparent (simple global threshold)."""
    rgba = mosaic.convert("RGBA")
    data = np.array(rgba)
    white_mask = (data[:, :, 0] > threshold) & (data[:, :, 1] > threshold) & (data[:, :, 2] > threshold)
    data[white_mask, 3] = 0
    return Image.fromarray(data, "RGBA")


# ── Method: rembg ──────────────────────────────────────────────────────────

_rembg_remove = None

def _get_rembg():
    global _rembg_remove
    if _rembg_remove is None:
        from rembg import remove as rr
        _rembg_remove = rr
    return _rembg_remove


def get_person_mask_rembg(source_image: Image.Image, target_size: tuple) -> Image.Image:
    """Use rembg on the source photo to get a person-shaped alpha mask."""
    rembg_fn = _get_rembg()
    result = rembg_fn(source_image)
    mask = result.split()[3]
    mask = mask.resize(target_size, Image.LANCZOS)
    return mask


# ── S3 helpers ─────────────────────────────────────────────────────────────

def s3_object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def upload_to_s3_with_retry(s3, bucket: str, key: str, data: bytes, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/jpeg")
            return
        except ClientError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── Image download helper ─────────────────────────────────────────────────

def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


# ── Row processors per method ─────────────────────────────────────────────

def process_row_rembg(row, mirror_bg, s3, dry_run):
    """rembg method: mask from source photo applied to mosaic."""
    result = dict(row)
    mosaic_url = row.get("mosaic_url", "").strip()
    source_url = row.get("source_image_url", "").strip()
    user_id = row.get("user_id", "").strip()
    result["parsed_name"] = parse_name_from_email(row.get("email", ""))

    if not mosaic_url or not source_url:
        log.warning(f"Skipping {user_id}: missing mosaic_url or source_image_url")
        result["mockup_url"] = ""
        return result

    s3_key = f"mockups/{user_id}/mockup.jpg"
    if not dry_run and s3_object_exists(s3, MOCKUP_S3_BUCKET, s3_key):
        log.info(f"Skipping {user_id}: already exists")
        result["mockup_url"] = f"{S3_BASE_URL}/{MOCKUP_S3_BUCKET}/{s3_key}"
        return result

    try:
        mosaic = download_image(mosaic_url).convert("RGB")
        source_img = download_image(source_url).convert("RGB")

        mask = get_person_mask_rembg(source_img, mosaic.size)
        mosaic_rgba = mosaic.convert("RGBA")
        mosaic_rgba.putalpha(mask)

        bg = mirror_bg.resize(mosaic.size, Image.LANCZOS).convert("RGBA")
        composite = Image.alpha_composite(bg, mosaic_rgba).convert("RGB")

        result["mockup_url"] = _save_or_upload(composite, user_id, s3_key, s3, dry_run)
    except Exception as e:
        log.error(f"Failed {user_id}: {e}")
        result["mockup_url"] = ""
    return result


def process_row_threshold(row, mirror_bg, s3, dry_run, threshold):
    """Threshold method: make all white pixels transparent."""
    result = dict(row)
    mosaic_url = row.get("mosaic_url", "").strip()
    user_id = row.get("user_id", "").strip()
    result["parsed_name"] = parse_name_from_email(row.get("email", ""))

    if not mosaic_url:
        log.warning(f"Skipping {user_id}: missing mosaic_url")
        result["mockup_url"] = ""
        return result

    s3_key = f"mockups/{user_id}/mockup.jpg"
    if not dry_run and s3_object_exists(s3, MOCKUP_S3_BUCKET, s3_key):
        log.info(f"Skipping {user_id}: already exists")
        result["mockup_url"] = f"{S3_BASE_URL}/{MOCKUP_S3_BUCKET}/{s3_key}"
        return result

    try:
        mosaic = download_image(mosaic_url)
        mosaic_rgba = remove_white_background_threshold(mosaic, threshold)

        bg = mirror_bg.resize(mosaic_rgba.size, Image.LANCZOS).convert("RGBA")
        composite = Image.alpha_composite(bg, mosaic_rgba).convert("RGB")

        result["mockup_url"] = _save_or_upload(composite, user_id, s3_key, s3, dry_run)
    except Exception as e:
        log.error(f"Failed {user_id}: {e}")
        result["mockup_url"] = ""
    return result


def process_row_flood(row, mirror_bg, s3, dry_run, threshold):
    """Flood-fill method: remove connected white background from mosaic."""
    result = dict(row)
    mosaic_url = row.get("mosaic_url", "").strip()
    user_id = row.get("user_id", "").strip()
    result["parsed_name"] = parse_name_from_email(row.get("email", ""))

    if not mosaic_url:
        log.warning(f"Skipping {user_id}: missing mosaic_url")
        result["mockup_url"] = ""
        return result

    s3_key = f"mockups/{user_id}/mockup.jpg"
    if not dry_run and s3_object_exists(s3, MOCKUP_S3_BUCKET, s3_key):
        log.info(f"Skipping {user_id}: already exists")
        result["mockup_url"] = f"{S3_BASE_URL}/{MOCKUP_S3_BUCKET}/{s3_key}"
        return result

    try:
        mosaic = download_image(mosaic_url)
        mosaic_rgba = remove_white_background_flood(mosaic, threshold)

        bg = mirror_bg.resize(mosaic_rgba.size, Image.LANCZOS).convert("RGBA")
        composite = Image.alpha_composite(bg, mosaic_rgba).convert("RGB")

        result["mockup_url"] = _save_or_upload(composite, user_id, s3_key, s3, dry_run)
    except Exception as e:
        log.error(f"Failed {user_id}: {e}")
        result["mockup_url"] = ""
    return result


def process_row_regenerate(row, mirror_bg, s3, dry_run, engine):
    """Regenerate method: re-create mosaic from scratch with mirror background."""
    result = dict(row)
    source_url = row.get("source_image_url", "").strip()
    user_id = row.get("user_id", "").strip()
    result["parsed_name"] = parse_name_from_email(row.get("email", ""))

    if not source_url:
        log.warning(f"Skipping {user_id}: missing source_image_url")
        result["mockup_url"] = ""
        return result

    s3_key = f"mockups/{user_id}/mockup.jpg"
    if not dry_run and s3_object_exists(s3, MOCKUP_S3_BUCKET, s3_key):
        log.info(f"Skipping {user_id}: already exists")
        result["mockup_url"] = f"{S3_BASE_URL}/{MOCKUP_S3_BUCKET}/{s3_key}"
        return result

    try:
        source_img = download_image(source_url).convert("RGB")
        mosaic = engine.generate(source_img, mirror_bg)

        result["mockup_url"] = _save_or_upload(mosaic, user_id, s3_key, s3, dry_run)
    except Exception as e:
        log.error(f"Failed {user_id}: {e}")
        result["mockup_url"] = ""
    return result


def _save_or_upload(image, user_id, s3_key, s3, dry_run):
    """Save locally (dry run) or upload to S3. Returns the URL/path."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    jpeg_bytes = buf.getvalue()

    if dry_run:
        dry_dir = Path(__file__).resolve().parent / "dry_run_output"
        dry_dir.mkdir(exist_ok=True)
        path = dry_dir / f"{user_id}_mockup.jpg"
        path.write_bytes(jpeg_bytes)
        log.info(f"Dry run: saved {path} ({len(jpeg_bytes)} bytes)")
        return str(path)
    else:
        upload_to_s3_with_retry(s3, MOCKUP_S3_BUCKET, s3_key, jpeg_bytes)
        log.info(f"Uploaded mockup for {user_id}")
        return f"{S3_BASE_URL}/{MOCKUP_S3_BUCKET}/{s3_key}"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate mockup images with mirror background")
    parser.add_argument("csv_path", help="Path to participant CSV")
    parser.add_argument("--method", choices=["rembg", "flood-fill", "threshold", "regenerate"], default="rembg",
                        help="Background replacement method (default: rembg)")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool workers (default: 8)")
    parser.add_argument("--threshold", type=int, default=240,
                        help="White background threshold for flood-fill method (default: 240)")
    parser.add_argument("--thumbnail-bucket", type=str, default="pistoletto.moe",
                        help="S3 bucket to load thumbnails from for regenerate method")
    parser.add_argument("--thumbnail-limit", type=int, default=4000,
                        help="Max thumbnails to load for regenerate method (default: 4000)")
    parser.add_argument("--dry-run", action="store_true", help="Save locally instead of uploading to S3")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        log.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    log.info(f"Loading mirror background from {MIRROR_BG_PATH}")
    mirror_bg = Image.open(MIRROR_BG_PATH).convert("RGBA")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info(f"Loaded {len(rows)} rows from {csv_path}")

    s3 = None if args.dry_run else get_s3_client()

    # For regenerate method, load the mosaic engine once
    engine = None
    if args.method == "regenerate":
        from mosaic_engine import MosaicEngine
        log.info(f"Loading mosaic engine (thumbnails from {args.thumbnail_bucket})...")
        engine = MosaicEngine(
            aws_region=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            s3_bucket=args.thumbnail_bucket,
            thumbnail_limit=args.thumbnail_limit,
        )

    log.info(f"Processing {len(rows)} rows with method={args.method}")

    # Build the per-row processor
    def process(row):
        if args.method == "rembg":
            return process_row_rembg(row, mirror_bg, s3, args.dry_run)
        elif args.method == "threshold":
            return process_row_threshold(row, mirror_bg, s3, args.dry_run, args.threshold)
        elif args.method == "flood-fill":
            return process_row_flood(row, mirror_bg, s3, args.dry_run, args.threshold)
        elif args.method == "regenerate":
            return process_row_regenerate(row, mirror_bg, s3, args.dry_run, engine)

    # Sequential for small batches or regenerate (not thread-safe due to shared engine state)
    results = []
    if args.workers == 1 or len(rows) <= 2 or args.method == "regenerate":
        for row in tqdm(rows, desc=f"Processing ({args.method})"):
            results.append(process(row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process, row): i for i, row in enumerate(rows)}
            results = [None] * len(rows)
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing ({args.method})"):
                idx = futures[future]
                results[idx] = future.result()

    # Write output CSV
    output_path = csv_path.parent / f"{csv_path.stem}_with_mockups.csv"
    fieldnames = list(rows[0].keys()) + ["parsed_name", "mockup_url"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info(f"Output written to {output_path}")
    success = sum(1 for r in results if r.get("mockup_url"))
    log.info(f"Done: {success} successful, {len(results) - success} failed/skipped")


if __name__ == "__main__":
    main()
