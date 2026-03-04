#!/usr/bin/env python3
"""
Deletion executor for the data-scrubbing workflow.

Reads a JSON manifest (exported from review.py's HTML gallery) and
removes the flagged assets from both S3 and DynamoDB.

Usage:
    python delete.py --manifest deletion_manifest_blend1_*.json --dry-run
    python delete.py --manifest deletion_manifest_blend1_*.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError

from config import get_s3_client, get_dynamodb_resource, get_blend_config


# ---------------------------------------------------------------------------
# DynamoDB key-schema discovery
# ---------------------------------------------------------------------------

def _get_table_key_names(dynamodb, table_name: str) -> list[str]:
    """Return the attribute names that form the table's primary key."""
    table = dynamodb.Table(table_name)
    table.load()
    return [k["AttributeName"] for k in table.key_schema]


# ---------------------------------------------------------------------------
# Deletion logic
# ---------------------------------------------------------------------------

def delete_s3_objects(s3, bucket: str, keys: list[str], *,
                      dry_run: bool = False) -> dict:
    results = {"deleted": [], "failed": []}
    for key in keys:
        label = f"  s3://{bucket}/{key}"
        if dry_run:
            print(f"  [DRY-RUN] Would delete {label}")
            results["deleted"].append(key)
            continue
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"  Deleted {label}")
            results["deleted"].append(key)
        except ClientError as e:
            msg = e.response["Error"]["Message"]
            print(f"  FAILED  {label} – {msg}")
            results["failed"].append({"key": key, "error": msg})
    return results


def delete_dynamodb_items(dynamodb, table_name: str, blend_ids: list[str],
                          *, dry_run: bool = False) -> dict:
    """Delete blend records by blend_id."""
    if not blend_ids:
        return {"deleted": [], "failed": []}

    key_names = _get_table_key_names(dynamodb, table_name)
    pk_name = key_names[0]

    table = dynamodb.Table(table_name)
    results = {"deleted": [], "failed": []}

    for bid in blend_ids:
        label = f"  {table_name}[{pk_name}={bid}]"
        if dry_run:
            print(f"  [DRY-RUN] Would delete {label}")
            results["deleted"].append(bid)
            continue
        try:
            table.delete_item(Key={pk_name: bid})
            print(f"  Deleted {label}")
            results["deleted"].append(bid)
        except ClientError as e:
            msg = e.response["Error"]["Message"]
            print(f"  FAILED  {label} – {msg}")
            results["failed"].append({"blend_id": bid, "error": msg})
    return results


def find_related_s3_keys(dynamodb, table_name: str, blend_ids: list[str]) -> list[str]:
    """Look up blend records and collect any video / extra S3 keys to clean up."""
    if not blend_ids:
        return []

    key_names = _get_table_key_names(dynamodb, table_name)
    pk_name = key_names[0]
    table = dynamodb.Table(table_name)

    extra_keys: list[str] = []
    video_fields = ("s3_key", "s3_video_key")

    for bid in blend_ids:
        try:
            resp = table.get_item(Key={pk_name: bid})
            item = resp.get("Item")
            if not item:
                continue
            for f in video_fields:
                val = item.get(f, "")
                if val and val not in extra_keys:
                    extra_keys.append(val)
        except ClientError:
            pass

    return extra_keys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Delete flagged assets from S3 and DynamoDB"
    )
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to the JSON deletion manifest")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview what would be deleted without making changes")
    p.add_argument("--skip-dynamo", action="store_true",
                   help="Only delete S3 objects; leave DynamoDB untouched")
    p.add_argument("--skip-videos", action="store_true",
                   help="Don't delete associated video-blend files from S3")
    p.add_argument("--log", default=None, type=Path,
                   help="Write a JSON deletion log to this path")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load manifest ---
    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}")
        sys.exit(1)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bucket = manifest["bucket"]
    table = manifest["table"]
    blend_num = manifest["blend_number"]
    items = manifest["items"]

    if not items:
        print("Manifest contains no items. Nothing to do.")
        sys.exit(0)

    s3_keys = [it["s3_key"] for it in items]
    blend_ids = sorted({bid for it in items for bid in it.get("blend_ids", [])})

    mode_label = "[DRY-RUN] " if args.dry_run else ""

    print(f"{'='*60}")
    print(f"  {mode_label}Deletion Plan – Blend {blend_num}")
    print(f"  Bucket : {bucket}")
    print(f"  Table  : {table}")
    print(f"  Images : {len(s3_keys)}")
    print(f"  Blends : {len(blend_ids)}")
    print(f"{'='*60}\n")

    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()

    log: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "blend_number": blend_num,
        "bucket": bucket,
        "table": table,
    }

    # --- Collect related video keys ---
    video_keys: list[str] = []
    if not args.skip_videos and blend_ids:
        print("Looking up associated video-blend S3 keys …")
        video_keys = find_related_s3_keys(dynamodb, table, blend_ids)
        already = set(s3_keys)
        video_keys = [k for k in video_keys if k not in already]
        print(f"  Found {len(video_keys)} extra video key(s) to clean up\n")

    # --- Delete images from S3 ---
    print(f"Deleting {len(s3_keys)} image(s) from S3 …")
    s3_img_result = delete_s3_objects(s3, bucket, s3_keys, dry_run=args.dry_run)
    log["s3_images"] = s3_img_result

    # --- Delete video blends from S3 ---
    if video_keys:
        print(f"\nDeleting {len(video_keys)} video-blend file(s) from S3 …")
        s3_vid_result = delete_s3_objects(s3, bucket, video_keys, dry_run=args.dry_run)
        log["s3_videos"] = s3_vid_result

    # --- Delete DynamoDB records ---
    if not args.skip_dynamo and blend_ids:
        print(f"\nDeleting {len(blend_ids)} blend record(s) from DynamoDB …")
        dynamo_result = delete_dynamodb_items(
            dynamodb, table, blend_ids, dry_run=args.dry_run
        )
        log["dynamodb"] = dynamo_result
    elif args.skip_dynamo:
        print("\nSkipping DynamoDB deletion (--skip-dynamo).")

    # --- Summary ---
    log["finished_at"] = datetime.now(timezone.utc).isoformat()

    s3_del = len(s3_img_result["deleted"]) + len((log.get("s3_videos") or {}).get("deleted", []))
    s3_fail = len(s3_img_result["failed"]) + len((log.get("s3_videos") or {}).get("failed", []))
    db_del = len((log.get("dynamodb") or {}).get("deleted", []))
    db_fail = len((log.get("dynamodb") or {}).get("failed", []))

    print(f"\n{'='*60}")
    print(f"  {mode_label}Summary")
    print(f"  S3 objects  : {s3_del} deleted, {s3_fail} failed")
    print(f"  DynamoDB    : {db_del} deleted, {db_fail} failed")
    print(f"{'='*60}")

    # --- Write log ---
    if args.log:
        args.log.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        print(f"\nDeletion log saved to: {args.log.resolve()}")
    else:
        default_log = Path(__file__).parent / f"delete_log_blend{blend_num}_{int(datetime.now().timestamp())}.json"
        default_log.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
        print(f"\nDeletion log saved to: {default_log.resolve()}")


if __name__ == "__main__":
    main()
