"""
Read a manifest produced by split.py and process the selected images:

  1. Download from S3
  2. Remove background via HuggingFace API (not-lain/background-removal)
  3. Composite onto a black canvas
  4. Upload the result to resized_black_bg/ in S3
  5. Delete the original from its source location in S3

Uses a ThreadPoolExecutor for concurrent processing.  Control how many
images to process with --limit (start with --limit 1 for testing).

Usage
-----
  # Test on a single image
  python process.py --manifest manifest_20260304_150000.json --limit 1

  # Dry run (no uploads / deletes)
  python process.py --manifest manifest_20260304_150000.json --limit 5 --dry-run

  # Process everything with 6 threads
  python process.py --manifest manifest_20260304_150000.json --workers 6
"""

import os
import json
import argparse
import logging
import time
import sys
import traceback
from pathlib import Path
from io import BytesIO
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from botocore.exceptions import ClientError
from PIL import Image
from gradio_client import Client, handle_file
from dotenv import load_dotenv

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("GRADIO_API_KEY"))

JPEG_QUALITY = 95
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

SCRIPT_DIR = Path(__file__).resolve().parent
TEMP_DIR = SCRIPT_DIR / "tmp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("process")

# ── Globals ─────────────────────────────────────────────────────────────────

bg_client: Optional[Client] = None


def get_s3_client():
    """Each thread should call this to get its own S3 client (thread-safe)."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def list_existing_dest_keys(bucket: str, dest_prefix: str) -> set[str]:
    """
    List all keys already present in destination prefix.
    Used to avoid sending already-processed assets to HF API.
    """
    s3 = get_s3_client()
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=dest_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and not key.endswith("/"):
                keys.add(key)
    return keys


def init_bg_client():
    """Initialize the shared HuggingFace background-removal Gradio client."""
    global bg_client
    try:
        if HF_TOKEN:
            logger.info("Initializing HF background-removal client (authenticated) ...")
            bg_client = Client("not-lain/background-removal", hf_token=HF_TOKEN)
        else:
            logger.warning("No HF_TOKEN found – connecting without authentication.")
            bg_client = Client("not-lain/background-removal")
        logger.info("Background-removal client ready.")
    except Exception as exc:
        logger.error(f"Failed to init background-removal client: {exc}")
        bg_client = None


# ── Image helpers ───────────────────────────────────────────────────────────

def _collect_file_refs(payload) -> list[str]:
    """
    Collect possible file references from a Gradio response payload.
    Handles strings, lists/tuples, and nested dict/list structures.
    """
    refs: list[str] = []

    def walk(node):
        if isinstance(node, str):
            if node.startswith(("http://", "https://", "/")):
                refs.append(node)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    walk(payload)
    return refs


def _open_image_from_ref(ref: str) -> Image.Image:
    if ref.startswith(("http://", "https://")):
        resp = requests.get(ref)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    return Image.open(ref)


def _has_transparency(img: Image.Image) -> bool:
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    min_a, _ = alpha.getextrema()
    return min_a < 255

def remove_background(local_path: str) -> Optional[Image.Image]:
    """Call the HF background-removal API with retry logic. Returns RGBA PIL Image or None."""
    if bg_client is None:
        logger.error("Background-removal client not initialised.")
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = bg_client.predict(
                image=handle_file(local_path), api_name="/image"
            )
            file_refs = _collect_file_refs(result)
            if file_refs:
                # Newer API responses may include multiple files. We prefer an image
                # that actually contains transparency (true bg-removed output).
                opened: list[tuple[str, Image.Image]] = []
                for ref in file_refs:
                    try:
                        opened.append((ref, _open_image_from_ref(ref)))
                    except Exception:
                        continue

                if opened:
                    selected_img: Optional[Image.Image] = None
                    selected_ref: Optional[str] = None

                    for ref, img in opened:
                        try:
                            if _has_transparency(img):
                                selected_img = img.convert("RGBA")
                                selected_ref = ref
                                break
                        except Exception:
                            continue

                    # Fallback: use last decodable image (often the final output)
                    if selected_img is None:
                        selected_ref, fallback_img = opened[-1]
                        selected_img = fallback_img.convert("RGBA")

                    # Clean up local temp files returned by gradio.
                    for ref, _img in opened:
                        if ref.startswith(("http://", "https://")):
                            continue
                        if ref == local_path:
                            continue
                        if os.path.exists(ref):
                            try:
                                os.remove(ref)
                            except OSError:
                                pass

                    logger.debug(f"Selected bg-removed asset: {selected_ref}")
                    return selected_img

            logger.error(f"Unexpected API result for {local_path}: {result}")
            return None

        except Exception as exc:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} for "
                f"{os.path.basename(local_path)} failed: {exc}"
            )
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.info(f"  retrying in {delay}s ...")
                time.sleep(delay)
            else:
                logger.error(
                    f"All {MAX_RETRIES} attempts failed for "
                    f"{os.path.basename(local_path)}"
                )
                logger.debug(traceback.format_exc())
    return None


def composite_on_black(img: Image.Image) -> Image.Image:
    """Paste an RGBA/LA image onto a solid black background, return RGB."""
    rgba = img.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(black, rgba).convert("RGB")


def save_jpeg_bytes(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


# ── Single-image pipeline (runs inside a worker thread) ─────────────────────

def process_one(entry: dict, bucket: str, dry_run: bool = False) -> dict:
    """
    Full pipeline for one image. Returns a result dict for the log.
    """
    src_key = entry["src_key"]
    dest_key = entry["dest_key"]
    basename = os.path.basename(src_key)
    result = {
        "src_key": src_key,
        "dest_key": dest_key,
        "status": "error",
        "error": None,
    }

    s3 = get_s3_client()
    local_path = str(TEMP_DIR / basename)

    try:
        # Skip if destination already exists (idempotent re-runs)
        try:
            s3.head_object(Bucket=bucket, Key=dest_key)
            logger.info(f"[skip]      {dest_key} already exists")
            result["status"] = "skipped"
            return result
        except ClientError:
            pass  # 404 = not found → proceed

        # 1. Download
        logger.info(f"[download]  {src_key}")
        s3.download_file(bucket, src_key, local_path)

        # 2. Remove background
        logger.info(f"[bg-remove] {basename}")
        fg_img = remove_background(local_path)
        if fg_img is None:
            result["error"] = "background removal failed"
            return result

        # 3. Composite on black canvas
        final_img = composite_on_black(fg_img)
        jpeg_data = save_jpeg_bytes(final_img)

        if dry_run:
            logger.info(f"[dry-run]   would upload → {dest_key} and delete {src_key}")
            result["status"] = "dry-run"
            return result

        # 4. Upload to resized_black_bg/
        logger.info(f"[upload]    {dest_key}")
        s3.put_object(
            Bucket=bucket, Key=dest_key, Body=jpeg_data, ContentType="image/jpeg"
        )

        # 5. Delete original from its source location
        logger.info(f"[delete]    {src_key}")
        s3.delete_object(Bucket=bucket, Key=src_key)

        result["status"] = "ok"

    except ClientError as exc:
        result["error"] = f"S3 error: {exc}"
        logger.error(f"[s3-error]  {basename}: {exc}")
    except Exception as exc:
        result["error"] = str(exc)
        logger.error(f"[error]     {basename}: {exc}")
        logger.debug(traceback.format_exc())
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process images from a split manifest: "
            "bg-remove → black canvas → upload → delete original."
        )
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to the JSON manifest from split.py",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max images to process (default: all). Use --limit 1 for testing.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel worker threads (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full pipeline but skip upload and delete steps",
    )
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    bucket = manifest["bucket"]
    images = manifest["images"]
    dest_prefix = manifest.get("dest_prefix", "selected-images/resized_black_bg/")

    # Pre-filter manifest entries whose destination already exists in S3.
    # This lets `--limit` mean "N images still needing processing".
    logger.info(f"Checking existing destination assets under {dest_prefix} ...")
    existing_dest_keys = list_existing_dest_keys(bucket, dest_prefix)
    images = [entry for entry in images if entry["dest_key"] not in existing_dest_keys]

    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        logger.info("No images to process (all selected manifest entries already exist).")
        return

    logger.info(f"Bucket:  {bucket}")
    logger.info(
        f"Images:  {len(images)} pending "
        f"(manifest: {manifest['sample_size']}, existing in dest: {len(existing_dest_keys)})"
    )
    logger.info(f"Workers: {args.workers}")
    if args.dry_run:
        logger.info("*** DRY RUN – no uploads or deletions ***")

    # Initialise background-removal client (shared across threads)
    init_bg_client()
    if bg_client is None:
        logger.error("Cannot proceed without background-removal client. Exiting.")
        return

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Dispatch work to the thread pool
    results = []
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one, entry, bucket, args.dry_run): entry
                for entry in images
            }

            iterator = as_completed(futures)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(futures), desc="Processing")

            for future in iterator:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    entry = futures[future]
                    logger.error(f"Worker exception for {entry['src_key']}: {exc}")
                    results.append({
                        "src_key": entry["src_key"],
                        "status": "error",
                        "error": str(exc),
                    })
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Exiting cleanly...")
    finally:
        # Gradio client has background heartbeat resources; close explicitly so
        # interpreter shutdown does not hang waiting on threads.
        if bg_client is not None:
            try:
                bg_client.close()
            except Exception:
                pass

    elapsed = time.time() - start

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    dry = sum(1 for r in results if r["status"] == "dry-run")

    logger.info(
        f"\nDone in {elapsed:.1f}s  |  ok={ok}  skipped={skipped}  "
        f"errors={errors}  dry-run={dry}"
    )

    # Persist a log for auditing
    log_path = SCRIPT_DIR / f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "manifest": str(args.manifest),
        "bucket": bucket,
        "limit": args.limit,
        "workers": args.workers,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(elapsed, 2),
        "summary": {"ok": ok, "skipped": skipped, "errors": errors, "dry_run": dry},
        "results": results,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Log saved to {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        if bg_client is not None:
            try:
                bg_client.close()
            except Exception:
                pass
        sys.exit(130)
