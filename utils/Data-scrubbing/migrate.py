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


# ---------------------------------------------------------------------------
# Subcommand: migrate
# ---------------------------------------------------------------------------

def run_migrate(args):
    """Copy verified good assets from source buckets to the sanitized bucket."""
    s3 = get_s3_client()
    dry_run = args.dry_run
    mode = "[DRY-RUN] " if dry_run else ""

    # Validate args
    if (args.start_date or args.end_date) and not args.system:
        print("Error: --start-date and --end-date require --system")
        sys.exit(1)

    # Determine which systems to process
    systems = [args.system] if args.system else [1, 2, 3]

    # Load deletion manifests
    print("Loading deletion manifests...")
    flagged = load_flagged_keys(MANIFEST_DIR)
    total_flagged = sum(len(v) for v in flagged.values())
    print(f"  Total flagged keys across all buckets: {total_flagged}")

    # List existing assets in sanitized bucket for dedup check
    print(f"\nListing existing assets in {SANITIZED_BUCKET} under {REVIEWED_PREFIX}...")
    existing_keys = set(list_s3_images(s3, SANITIZED_BUCKET, REVIEWED_PREFIX))
    print(f"  Found {len(existing_keys)} existing reviewed assets")

    log = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "systems": {},
    }

    for sys_num in systems:
        bucket = BLENDS[sys_num]["bucket"]

        # Determine date range
        if args.system and args.start_date and args.end_date:
            start = args.start_date
            end = args.end_date
        else:
            start, end = DEFAULT_RANGES[sys_num]

        print(f"\n{'='*60}")
        print(f"  {mode}System {sys_num}: {bucket}")
        print(f"  Date range: {start} to {end}")
        print(f"{'='*60}")

        # List all images in source bucket
        print(f"\nListing images in {bucket}...")
        all_keys = list_s3_images(s3, bucket)
        print(f"  Found {len(all_keys)} total images")

        # Get flagged keys for this bucket
        bucket_flagged = flagged.get(bucket, set())
        print(f"  {len(bucket_flagged)} flagged keys for this bucket")

        # Filter
        to_copy: list[tuple[str, str]] = []
        skipped = {
            "not_participant": 0,
            "out_of_range": 0,
            "flagged": 0,
            "already_exists": 0,
            "no_date": 0,
        }

        for key in all_keys:
            parts = key.split("/")
            # Expected: selected-images/<folder>/<filename>
            if len(parts) < 3:
                continue

            folder = parts[1]  # folder after selected-images/
            filename = parts[-1]

            # Must be a participant session (UUID folder)
            if not is_participant_session(folder):
                skipped["not_participant"] += 1
                continue

            # Must have a parseable capture date
            cap_date = parse_capture_date(filename)
            if cap_date is None:
                skipped["no_date"] += 1
                continue

            # Must be within the reviewed date range
            if cap_date < start or cap_date > end:
                skipped["out_of_range"] += 1
                continue

            # Must not be flagged by the reviewer
            if key in bucket_flagged:
                skipped["flagged"] += 1
                continue

            # Build destination key
            dest_key = f"selected-images/reviewed-images/system-{sys_num}/{folder}/{filename}"

            # Must not already exist in the destination
            if dest_key in existing_keys:
                skipped["already_exists"] += 1
                continue

            to_copy.append((key, dest_key))

        print(f"\n  Filtering results:")
        print(f"    To copy: {len(to_copy)}")
        print(f"    Flagged (excluded): {skipped['flagged']}")
        print(f"    Already in destination: {skipped['already_exists']}")
        print(f"    Non-participant: {skipped['not_participant']}")
        print(f"    Out of date range: {skipped['out_of_range']}")
        print(f"    No parseable date: {skipped['no_date']}")

        # Copy
        copied: list[dict] = []
        failed: list[dict] = []
        for i, (src_key, dst_key) in enumerate(to_copy):
            if (i + 1) % 50 == 0 or (i + 1) == len(to_copy):
                print(f"  {'Copying' if not dry_run else 'Would copy'} {i + 1}/{len(to_copy)}...")

            if dry_run:
                copied.append({"source": src_key, "destination": dst_key})
            else:
                try:
                    s3.copy_object(
                        Bucket=SANITIZED_BUCKET,
                        Key=dst_key,
                        CopySource={"Bucket": bucket, "Key": src_key},
                    )
                    copied.append({"source": src_key, "destination": dst_key})
                except Exception as e:
                    print(f"  FAILED: {src_key} -> {e}")
                    failed.append({"source": src_key, "destination": dst_key, "error": str(e)})

        log["systems"][str(sys_num)] = {
            "bucket": bucket,
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "total_listed": len(all_keys),
            "skipped": skipped,
            "copied": len(copied),
            "failed": len(failed),
            "copied_items": copied,
            "failed_items": failed,
        }

        print(f"\n  {mode}System {sys_num}: {len(copied)} copied, {len(failed)} failed")

    # Write log
    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    log_path = Path(__file__).resolve().parent / f"migrate_log_{int(datetime.now().timestamp())}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\nLog saved to: {log_path}")


# ---------------------------------------------------------------------------
# Subcommand: dedup
# ---------------------------------------------------------------------------

def run_dedup(args):
    """Scan the sanitized bucket for duplicate assets and optionally clean them."""
    s3 = get_s3_client()
    dry_run = args.dry_run
    clean = args.clean
    mode = "[DRY-RUN] " if dry_run else ""

    print(f"Scanning {SANITIZED_BUCKET} for duplicate assets...")
    all_keys = list_s3_images(s3, SANITIZED_BUCKET)
    print(f"  Found {len(all_keys)} total assets")

    # Group by dedup key
    groups: dict[str, list[str]] = {}
    skipped = 0
    for key in all_keys:
        dk = extract_dedup_key(key)
        if dk is None:
            skipped += 1
            continue
        groups.setdefault(dk, []).append(key)

    duplicates = {dk: paths for dk, paths in groups.items() if len(paths) > 1}
    total_extra = sum(len(paths) - 1 for paths in duplicates.values())

    # Report
    print(f"\n{'='*60}")
    print(f"  Dedup Report — {SANITIZED_BUCKET}")
    print(f"{'='*60}")
    print(f"  Total assets scanned: {len(all_keys)}")
    print(f"  Skipped (no UUID): {skipped}")
    print(f"  Unique assets: {len(groups)}")
    print(f"  Duplicated assets: {len(duplicates)}")
    print(f"  Extra copies: {total_extra}")

    if duplicates:
        print(f"\n  Duplicate groups:")
        for dk, paths in sorted(duplicates.items()):
            print(f"\n    {dk}:")
            for p in paths:
                print(f"      - {p}")

    # Write report JSON
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": SANITIZED_BUCKET,
        "total_scanned": len(all_keys),
        "skipped_no_uuid": skipped,
        "unique_assets": len(groups),
        "duplicate_groups": len(duplicates),
        "extra_copies": total_extra,
        "duplicates": {dk: paths for dk, paths in sorted(duplicates.items())},
    }
    report_path = Path(__file__).resolve().parent / f"dedup_report_{int(datetime.now().timestamp())}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")

    # Clean duplicates
    if clean and duplicates:
        print(f"\n{mode}Cleaning duplicates...")
        deleted: list[str] = []
        failed: list[dict] = []

        for dk, paths in duplicates.items():
            # Prefer the reviewed-images copy; fall back to first found
            reviewed = [p for p in paths if "/reviewed-images/" in p]
            keep = reviewed[0] if reviewed else paths[0]
            to_delete = [p for p in paths if p != keep]

            for key in to_delete:
                if dry_run:
                    print(f"  [DRY-RUN] Would delete {key} (keeping {keep})")
                    deleted.append(key)
                else:
                    try:
                        s3.delete_object(Bucket=SANITIZED_BUCKET, Key=key)
                        print(f"  Deleted: {key} (keeping {keep})")
                        deleted.append(key)
                    except Exception as e:
                        print(f"  FAILED: {key} — {e}")
                        failed.append({"key": key, "error": str(e)})

        print(f"\n  {mode}Clean summary: {len(deleted)} deleted, {len(failed)} failed")
    elif clean and not duplicates:
        print("\n  No duplicates found — nothing to clean.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Asset migration pipeline for verified good assets"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # migrate
    mig = sub.add_parser("migrate", help="Copy verified good assets to sanitized bucket")
    mig.add_argument("--dry-run", action="store_true", help="Preview without copying")
    mig.add_argument("--system", type=int, choices=[1, 2, 3],
                     help="Process a single system (default: all three)")
    mig.add_argument("--start-date", type=date.fromisoformat,
                     help="Override start date (YYYY-MM-DD), requires --system")
    mig.add_argument("--end-date", type=date.fromisoformat,
                     help="Override end date (YYYY-MM-DD), requires --system")

    # dedup
    ded = sub.add_parser("dedup", help="Scan sanitized bucket for duplicate assets")
    ded.add_argument("--clean", action="store_true", help="Delete duplicate copies (keeps reviewed-images)")
    ded.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    # audit-system4 (placeholder -- implemented in Task 4)
    sub.add_parser("audit-system4", help="Diagnostic report of system 4 coverage")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "migrate":
        run_migrate(args)
    elif args.command == "dedup":
        run_dedup(args)
    elif args.command == "audit-system4":
        print("audit-system4: not yet implemented")


if __name__ == "__main__":
    main()
