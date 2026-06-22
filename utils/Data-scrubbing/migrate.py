#!/usr/bin/env python3
"""
Asset migration pipeline for verified good assets.

Subcommands:
  migrate        Copy verified good assets from systems 1-3 to sanitized bucket
  dedup          Scan sanitized bucket for duplicate assets
  audit-system4  Diagnostic report of system 4 coverage

Usage:
    python migrate.py migrate --dry-run
    python migrate.py migrate --system 1 --start-date 2026-02-07 --end-date 2026-06-22
    python migrate.py dedup
    python migrate.py dedup --clean --dry-run
    python migrate.py audit-system4
"""

import argparse
import json
import re
import sys
from datetime import datetime, date, timezone
from pathlib import Path

from config import get_s3_client, BLENDS, IMAGE_PREFIX, IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SANITIZED_BUCKET = BLENDS[5]["bucket"]  # pistoletto.sanitized
REVIEWED_PREFIX = "selected-images/reviewed-images/"
MANIFEST_DIR = Path(__file__).resolve().parent / "reviewed_assets"

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
CAPTURE_DATE_RE = re.compile(r"capture_(\d{8})_\d{6}")

DEFAULT_RANGES = {
    1: (date(2026, 2, 7), date(2026, 6, 22)),
    2: (date(2026, 2, 21), date(2026, 6, 22)),
    3: (date(2026, 1, 4), date(2026, 6, 22)),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def is_participant_session(folder_name: str) -> bool:
    """Return True if folder_name is a valid lowercase UUID (participant session)."""
    return bool(UUID_RE.match(folder_name))


def parse_capture_date(filename: str) -> date | None:
    """Extract the date from a capture_YYYYMMDD_HHMMSS filename."""
    m = CAPTURE_DATE_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def extract_dedup_key(s3_key: str) -> str | None:
    """
    Extract <uuid>/<filename> from an S3 key, regardless of directory depth.

    Works for paths like:
      selected-images/<uuid>/file.jpg
      selected-images/system-1/<uuid>/file.jpg
      selected-images/reviewed-images/system-1/<uuid>/file.jpg

    Returns None if no UUID folder is found.
    """
    parts = s3_key.split("/")
    for i, part in enumerate(parts):
        if UUID_RE.match(part) and i + 1 < len(parts):
            return f"{part}/{parts[-1]}"
    return None


def load_flagged_keys(manifest_dir: Path) -> dict[str, set[str]]:
    """
    Load all deletion manifests from manifest_dir.
    Returns {bucket_name: set of flagged s3_keys}.
    """
    flagged: dict[str, set[str]] = {}
    if not manifest_dir.is_dir():
        return flagged
    for f in sorted(manifest_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        bucket = data.get("bucket", "")
        if not bucket:
            continue
        keys = flagged.setdefault(bucket, set())
        for item in data.get("items", []):
            s3_key = item.get("s3_key", "")
            if s3_key:
                keys.add(s3_key)
        print(f"  Loaded {len(data.get('items', []))} flagged keys from {f.name} (bucket: {bucket})")
    return flagged


def list_s3_images(s3, bucket: str, prefix: str = IMAGE_PREFIX) -> list[str]:
    """List all image keys under prefix in bucket using paginated listing."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
            if ext in IMAGE_EXTENSIONS:
                keys.append(key)
    return keys
