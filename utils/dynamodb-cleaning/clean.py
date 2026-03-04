#!/usr/bin/env python3
"""
DynamoDB orphan-row cleaner.

Scans every row in an eternity-mirror-blends table and verifies that the
S3 objects referenced by source_image_key, random_image_key, and s3_key
actually exist in the corresponding S3 bucket.  Rows whose referenced
assets are missing are flagged and optionally deleted.

Usage:
    # Dry-run (preview only, no changes):
    python clean.py --blend 1 --dry-run

    # Delete rows where ANY referenced asset is missing:
    python clean.py --blend 1

    # Delete rows only when ALL referenced assets are missing:
    python clean.py --blend 1 --mode all

    # Process all blends at once:
    python clean.py --blend all --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from config import (
    get_s3_client,
    get_dynamodb_resource,
    get_blend_config,
    S3_KEY_FIELDS,
    BLENDS,
)


# ---------------------------------------------------------------------------
# S3 inventory
# ---------------------------------------------------------------------------

def build_s3_inventory(s3, bucket: str) -> set[str]:
    """List every key in *bucket* and return them as a set for O(1) lookups."""
    keys: set[str] = set()
    kwargs = {"Bucket": bucket, "MaxKeys": 1000}
    page = 0

    print(f"  Building S3 inventory for s3://{bucket}/ ...")

    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.add(obj["Key"])
        page += 1

        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    print(f"    {len(keys):,} objects across {page} page(s)")
    return keys


# ---------------------------------------------------------------------------
# DynamoDB scan
# ---------------------------------------------------------------------------

def scan_table(dynamodb, table_name: str) -> list[dict]:
    """Full scan of the DynamoDB table. Returns raw items."""
    print(f"  Scanning DynamoDB table: {table_name} ...")
    table = dynamodb.Table(table_name)
    items: list[dict] = []
    scan_kwargs: dict = {}

    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    print(f"    {len(items):,} rows")
    return items


def _get_table_key_names(dynamodb, table_name: str) -> list[str]:
    """Return the attribute names that form the table's primary key."""
    table = dynamodb.Table(table_name)
    table.load()
    return [k["AttributeName"] for k in table.key_schema]


# ---------------------------------------------------------------------------
# Key normalisation
# ---------------------------------------------------------------------------

def _normalize_key(raw_key: str, bucket: str) -> str:
    """
    Strip the bucket-name prefix that DynamoDB records sometimes include.

    DynamoDB may store keys like "pistoletto.moe/selected-images/…" while
    S3 list_objects_v2 returns bare keys like "selected-images/…".
    """
    prefix = bucket + "/"
    if raw_key.startswith(prefix):
        return raw_key[len(prefix):]
    return raw_key


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------

def find_orphan_rows(
    items: list[dict],
    s3_keys: set[str],
    bucket: str,
    mode: str = "any",
) -> list[dict]:
    """
    Return rows whose referenced S3 assets are missing.

    mode="any"  -> flag the row if ANY of its asset keys is missing
    mode="all"  -> flag the row only when ALL of its asset keys are missing
    """
    orphans: list[dict] = []

    for item in items:
        refs: dict[str, str] = {}
        for field in S3_KEY_FIELDS:
            val = item.get(field, "")
            if val:
                refs[field] = val

        if not refs:
            orphans.append({
                "item": item,
                "missing_fields": list(S3_KEY_FIELDS),
                "present_fields": [],
            })
            continue

        missing = [f for f, k in refs.items()
                    if _normalize_key(k, bucket) not in s3_keys]
        present = [f for f, k in refs.items()
                    if _normalize_key(k, bucket) in s3_keys]

        is_orphan = (
            len(missing) > 0 if mode == "any" else len(missing) == len(refs)
        )

        if is_orphan:
            orphans.append({
                "item": item,
                "missing_fields": missing,
                "present_fields": present,
            })

    return orphans


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_rows(dynamodb, table_name: str, items: list[dict],
                *, dry_run: bool = False) -> dict:
    """Delete the given items from the DynamoDB table."""
    key_names = _get_table_key_names(dynamodb, table_name)
    table = dynamodb.Table(table_name)

    results = {"deleted": 0, "failed": 0, "errors": []}

    for entry in items:
        item = entry["item"]
        key = {k: item[k] for k in key_names if k in item}
        label = "  " + " / ".join(f"{k}={v}" for k, v in key.items())

        if dry_run:
            print(f"  [DRY-RUN] Would delete {label}")
            results["deleted"] += 1
            continue

        try:
            table.delete_item(Key=key)
            print(f"  Deleted {label}")
            results["deleted"] += 1
        except ClientError as e:
            msg = e.response["Error"]["Message"]
            print(f"  FAILED  {label} – {msg}")
            results["failed"] += 1
            results["errors"].append({"key": key, "error": msg})

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(orphans: list[dict], total_rows: int, total_s3: int,
                 mode: str, blend_num, bucket: str, table: str):
    """Print a human-readable summary of the scan."""
    clean = total_rows - len(orphans)

    print(f"\n{'='*60}")
    print(f"  Scan Results – Blend {blend_num}")
    print(f"  Bucket         : {bucket}")
    print(f"  Table          : {table}")
    print(f"  S3 objects     : {total_s3:,}")
    print(f"  DynamoDB rows  : {total_rows:,}")
    print(f"  Clean rows     : {clean:,}")
    print(f"  Orphan rows    : {len(orphans):,}")
    print(f"  Detection mode : {mode} (flag when {mode.upper()} asset keys are missing)")
    print(f"{'='*60}")

    if not orphans:
        print("\n  All rows have valid S3 references. Nothing to clean.")
        return

    missing_counts: dict[str, int] = {f: 0 for f in S3_KEY_FIELDS}
    for o in orphans:
        for f in o["missing_fields"]:
            if f in missing_counts:
                missing_counts[f] += 1

    print("\n  Missing-asset breakdown:")
    for field, count in missing_counts.items():
        print(f"    {field:25s} : {count:,} rows")

    print(f"\n  First 10 orphan rows:")
    for o in orphans[:10]:
        item = o["item"]
        blend_id = item.get("blend_id", "?")
        missing = ", ".join(o["missing_fields"])
        print(f"    blend_id={blend_id}  missing=[{missing}]")
    if len(orphans) > 10:
        print(f"    ... and {len(orphans) - 10} more")


# ---------------------------------------------------------------------------
# Per-blend processing
# ---------------------------------------------------------------------------

def process_blend(blend_num, s3, dynamodb, *, mode: str, dry_run: bool,
                  log_dir: Path) -> dict:
    """Run the full scan-and-clean pipeline for a single blend."""
    cfg = get_blend_config(blend_num)
    bucket = cfg["bucket"]
    table = cfg["table"]

    print(f"\n{'='*60}")
    print(f"  Blend {blend_num}:  bucket={bucket}  table={table}")
    print(f"{'='*60}\n")

    s3_keys = build_s3_inventory(s3, bucket)
    items = scan_table(dynamodb, table)

    orphans = find_orphan_rows(items, s3_keys, bucket, mode=mode)

    print_report(orphans, len(items), len(s3_keys), mode, blend_num, bucket, table)

    log: dict = {
        "blend_number": blend_num,
        "bucket": bucket,
        "table": table,
        "mode": mode,
        "dry_run": dry_run,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_s3_objects": len(s3_keys),
        "total_dynamo_rows": len(items),
        "orphan_count": len(orphans),
    }

    if orphans:
        mode_label = "[DRY-RUN] " if dry_run else ""
        print(f"\n{mode_label}Deleting {len(orphans):,} orphan row(s) from {table} ...")
        result = delete_rows(dynamodb, table, orphans, dry_run=dry_run)
        log["deletion"] = result
    else:
        log["deletion"] = {"deleted": 0, "failed": 0, "errors": []}

    log["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Serialise orphan details (without the full item to keep logs manageable)
    log["orphan_details"] = [
        {
            "blend_id": o["item"].get("blend_id", "unknown"),
            "missing_fields": o["missing_fields"],
            "present_fields": o["present_fields"],
            **{f: o["item"].get(f, "") for f in S3_KEY_FIELDS},
        }
        for o in orphans
    ]

    log_path = log_dir / f"clean_log_blend{blend_num}_{int(datetime.now().timestamp())}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\nLog saved to: {log_path.resolve()}")

    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Scan DynamoDB blend tables and remove rows whose S3 assets are missing"
    )
    p.add_argument(
        "--blend", required=True,
        help="Blend number (1-4) or 'all' to process every blend",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be deleted without making changes",
    )
    p.add_argument(
        "--mode", choices=["any", "all"], default="any",
        help=(
            "When to flag a row: "
            "'any' = flag if ANY asset key is missing (default), "
            "'all' = flag only when ALL asset keys are missing"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.blend.lower() == "all":
        blend_nums = sorted(BLENDS.keys())
    else:
        try:
            n = int(args.blend)
        except ValueError:
            print(f"Error: --blend must be 1-4 or 'all', got '{args.blend}'")
            sys.exit(1)
        if n not in BLENDS:
            print(f"Error: invalid blend number {n}. Must be 1-4.")
            sys.exit(1)
        blend_nums = [n]

    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()
    log_dir = Path(__file__).parent

    mode_label = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{mode_label}DynamoDB Orphan-Row Cleaner")
    print(f"Blends to process: {blend_nums}")
    print(f"Detection mode   : {args.mode}")
    print(f"Dry run          : {args.dry_run}")

    summaries: list[dict] = []
    for blend_num in blend_nums:
        summary = process_blend(
            blend_num, s3, dynamodb,
            mode=args.mode, dry_run=args.dry_run, log_dir=log_dir,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        total_orphans = sum(s["orphan_count"] for s in summaries)
        total_deleted = sum(s["deletion"]["deleted"] for s in summaries)
        total_failed = sum(s["deletion"]["failed"] for s in summaries)

        print(f"\n{'='*60}")
        print(f"  {mode_label}Grand Summary ({len(summaries)} blends)")
        print(f"  Total orphan rows : {total_orphans:,}")
        print(f"  Deleted           : {total_deleted:,}")
        print(f"  Failed            : {total_failed:,}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
