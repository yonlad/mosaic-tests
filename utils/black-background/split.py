"""
Scan the pistoletto.sanitized S3 bucket, identify images that are NOT already
in the resized_black_bg/ directory, randomly sample a configurable percentage,
and persist the selection as a JSON manifest for process.py to consume.

Usage
-----
  # Default: 30% random sample
  python split.py

  # Custom sample percentage, fixed seed for reproducibility
  python split.py --sample-pct 10 --seed 42

  # Write manifest to a specific path
  python split.py -o my_manifest.json
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

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

BUCKET = "pistoletto.sanitized"
IMAGE_PREFIX = "selected-images/"
DEST_PREFIX = "selected-images/resized_black_bg/"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".heic", ".heif"}

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def list_s3_keys(s3, bucket, prefix):
    """Yield every object key under *prefix*, skipping directory markers."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                yield key


def is_image(key):
    return os.path.splitext(key)[1].lower() in IMAGE_EXTENSIONS


def dest_key_for(src_key):
    """Map any source key → its would-be destination in resized_black_bg/."""
    stem = os.path.splitext(os.path.basename(src_key))[0]
    return f"{DEST_PREFIX}{stem}.jpg"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select a random subset of images for black-background migration."
    )
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument(
        "--prefix", default=IMAGE_PREFIX,
        help="Top-level S3 prefix to scan (default: selected-images/)",
    )
    parser.add_argument(
        "--dest-prefix", default=DEST_PREFIX,
        help="Destination prefix – images here are excluded from selection",
    )
    parser.add_argument(
        "--sample-pct", type=float, default=30.0,
        help="Percentage of eligible images to select (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output manifest path (default: auto-generated in script dir)",
    )
    args = parser.parse_args()

    s3 = get_s3_client()

    # 1. List all images under the top-level prefix, excluding the destination
    print(f"Scanning s3://{args.bucket}/{args.prefix} ...")
    all_keys = []
    for key in list_s3_keys(s3, args.bucket, args.prefix):
        if key.startswith(args.dest_prefix):
            continue
        if is_image(key):
            all_keys.append(key)
    print(f"  Found {len(all_keys)} images outside {args.dest_prefix}")

    # 2. Collect basenames already present in the destination
    print(f"Scanning s3://{args.bucket}/{args.dest_prefix} for already-processed images ...")
    existing_basenames = set()
    for key in list_s3_keys(s3, args.bucket, args.dest_prefix):
        existing_basenames.add(os.path.splitext(os.path.basename(key))[0])
    print(f"  Found {len(existing_basenames)} images already in {args.dest_prefix}")

    # 3. Keep only images whose basename is not yet in the destination
    eligible = [
        key for key in all_keys
        if os.path.splitext(os.path.basename(key))[0] not in existing_basenames
    ]
    print(f"  {len(eligible)} images are eligible (not yet processed)")

    if not eligible:
        print("Nothing to do – all images are already processed or none found.")
        return

    # 4. Random sample
    if args.seed is not None:
        random.seed(args.seed)
    sample_size = max(1, int(len(eligible) * args.sample_pct / 100.0))
    selected = random.sample(eligible, min(sample_size, len(eligible)))
    print(f"  Selected {len(selected)} images ({args.sample_pct}% of {len(eligible)} eligible)")

    # 5. Build manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "source_prefix": args.prefix,
        "dest_prefix": args.dest_prefix,
        "total_in_bucket": len(all_keys),
        "already_processed": len(existing_basenames),
        "total_eligible": len(eligible),
        "sample_percentage": args.sample_pct,
        "sample_size": len(selected),
        "images": [
            {"src_key": k, "dest_key": dest_key_for(k)} for k in selected
        ],
    }

    # 6. Write manifest
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SCRIPT_DIR / f"manifest_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {out_path}")
    print(f"  Total selected: {len(selected)}")
    print(f"\nNext step:")
    print(f"  # Test on 1 image first:")
    print(f"  python process.py --manifest {out_path} --limit 1")
    print(f"  # Then process all:")
    print(f"  python process.py --manifest {out_path}")


if __name__ == "__main__":
    main()
