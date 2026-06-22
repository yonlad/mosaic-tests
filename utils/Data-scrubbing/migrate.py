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

from config import get_s3_client, get_dynamodb_resource, BLENDS, IMAGE_PREFIX, IMAGE_EXTENSIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SANITIZED_BUCKET = BLENDS[5]["bucket"]  # pistoletto.sanitized
REVIEWED_PREFIX = "selected-images/reviewed-images/"
MANIFEST_DIR = Path(__file__).resolve().parent / "reviewed_assets"
CENTRAL_TABLE = "eternity-mirror-blends-central"
AWS_REGION = "us-east-2"
S3_URL_BASE = f"https://s3.{AWS_REGION}.amazonaws.com/{SANITIZED_BUCKET}/"

SYSTEM_MAP = {
    "pistoletto.moe": 1,
    "pistoletto.moe2": 2,
    "pistoletto.moe3": 3,
    "pistoletto.moe4": 4,
}

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


def strip_bucket_prefix(raw_key: str) -> str:
    """Strip any leading bucket-name prefix from an S3 key.

    DynamoDB sometimes stores keys like 'pistoletto.moe3/selected-images/...'
    while the actual S3 key is just 'selected-images/...'.
    """
    for bucket_name in SYSTEM_MAP:
        prefix = bucket_name + "/"
        if raw_key.startswith(prefix):
            return raw_key[len(prefix):]
    if raw_key.startswith(SANITIZED_BUCKET + "/"):
        return raw_key[len(SANITIZED_BUCKET) + 1:]
    return raw_key


def public_url(bucket: str, key: str) -> str:
    """Build a plain public S3 URL (no signature, no expiry)."""
    return f"https://s3.{AWS_REGION}.amazonaws.com/{bucket}/{key}"


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
        # Build redirect map: deleted_key -> kept_key
        redirect_map: dict[str, str] = {}
        for dk, paths in duplicates.items():
            reviewed = [p for p in paths if "/reviewed-images/" in p]
            keep = reviewed[0] if reviewed else paths[0]
            for p in paths:
                if p != keep:
                    redirect_map[p] = keep

        # Scan DynamoDB for records referencing keys we're about to delete
        print(f"\n{mode}Scanning DynamoDB table {CENTRAL_TABLE} for affected references...")
        dynamodb = get_dynamodb_resource()
        table = dynamodb.Table(CENTRAL_TABLE)
        db_items: list[dict] = []
        scan_kwargs: dict = {}
        while True:
            resp = table.scan(**scan_kwargs)
            db_items.extend(resp.get("Items", []))
            last = resp.get("LastEvaluatedKey")
            if not last:
                break
            scan_kwargs["ExclusiveStartKey"] = last

        # Find records that reference deleted keys (check both image fields)
        KEY_FIELDS = [
            ("source_image_key", "source_image_bucket", "source_image_url"),
            ("random_image_key", "random_image_bucket", "random_image_url"),
        ]
        redirects: list[dict] = []
        for item in db_items:
            for key_field, bucket_field, url_field in KEY_FIELDS:
                raw_key = item.get(key_field, "")
                bucket_val = item.get(bucket_field, "")
                # Handle both bare keys and bucket-prefixed keys
                if bucket_val == SANITIZED_BUCKET and raw_key in redirect_map:
                    redirects.append({
                        "blend_id": item["blend_id"],
                        "field": key_field,
                        "old_key": raw_key,
                        "new_key": redirect_map[raw_key],
                        "url_field": url_field,
                    })
                elif raw_key.startswith(SANITIZED_BUCKET + "/"):
                    bare = raw_key[len(SANITIZED_BUCKET) + 1:]
                    if bare in redirect_map:
                        redirects.append({
                            "blend_id": item["blend_id"],
                            "field": key_field,
                            "old_key": raw_key,
                            "new_key": SANITIZED_BUCKET + "/" + redirect_map[bare],
                            "url_field": url_field,
                        })

        print(f"  Found {len(redirects)} DynamoDB references to update")

        # Update DynamoDB references
        db_updated = 0
        db_failed: list[dict] = []
        for redir in redirects:
            new_url = S3_URL_BASE + redir["new_key"]
            if dry_run:
                print(f"  [DRY-RUN] Would update {CENTRAL_TABLE}[{redir['blend_id'][:12]}...].{redir['field']}: {redir['old_key'][-50:]} -> {redir['new_key'][-50:]}")
                db_updated += 1
            else:
                try:
                    table.update_item(
                        Key={"blend_id": redir["blend_id"]},
                        UpdateExpression=f"SET #kf = :nk, #uf = :nu",
                        ExpressionAttributeNames={
                            "#kf": redir["field"],
                            "#uf": redir["url_field"],
                        },
                        ExpressionAttributeValues={
                            ":nk": redir["new_key"],
                            ":nu": new_url,
                        },
                    )
                    print(f"  Updated {CENTRAL_TABLE}[{redir['blend_id'][:12]}...].{redir['field']}")
                    db_updated += 1
                except Exception as e:
                    print(f"  FAILED updating {redir['blend_id']}: {e}")
                    db_failed.append({"blend_id": redir["blend_id"], "error": str(e)})

        print(f"  {mode}DynamoDB: {db_updated} updated, {len(db_failed)} failed")

        if db_failed:
            print("  ERROR: DynamoDB updates failed — aborting S3 deletions to prevent broken references.")
        else:
            # Delete duplicate S3 objects
            print(f"\n{mode}Deleting duplicate S3 objects...")
            deleted: list[str] = []
            failed: list[dict] = []

            for key, keep in redirect_map.items():
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

            print(f"\n  {mode}Clean summary: {len(deleted)} S3 objects deleted, {len(failed)} failed")
    elif clean and not duplicates:
        print("\n  No duplicates found — nothing to clean.")


# ---------------------------------------------------------------------------
# Subcommand: audit-system4
# ---------------------------------------------------------------------------

def run_audit_system4(args):
    """Produce a diagnostic report of system 4 asset coverage in the sanitized bucket."""
    s3 = get_s3_client()
    source_bucket = BLENDS[4]["bucket"]

    print(f"Auditing system 4 ({source_bucket}) coverage in {SANITIZED_BUCKET}...\n")

    # List system 4 assets
    print(f"Listing assets in {source_bucket}...")
    sys4_keys = list_s3_images(s3, source_bucket)
    print(f"  Found {len(sys4_keys)} total images")

    # Filter to participant sessions only
    sys4_participant: list[str] = []
    for key in sys4_keys:
        parts = key.split("/")
        if len(parts) >= 3 and is_participant_session(parts[1]):
            sys4_participant.append(key)
    print(f"  {len(sys4_participant)} participant session images")

    # List sanitized assets and build dedup key lookup
    print(f"\nListing assets in {SANITIZED_BUCKET}...")
    sanitized_keys = list_s3_images(s3, SANITIZED_BUCKET)
    print(f"  Found {len(sanitized_keys)} total images")

    sanitized_dedup: dict[str, list[str]] = {}
    for key in sanitized_keys:
        dk = extract_dedup_key(key)
        if dk:
            sanitized_dedup.setdefault(dk, []).append(key)

    # Compare
    present: list[dict] = []
    missing: list[str] = []
    for key in sys4_participant:
        dk = extract_dedup_key(key)
        if dk and dk in sanitized_dedup:
            present.append({
                "source_key": key,
                "dedup_key": dk,
                "found_at": sanitized_dedup[dk],
            })
        else:
            missing.append(key)

    # Report
    print(f"\n{'='*60}")
    print(f"  System 4 Audit — {source_bucket}")
    print(f"{'='*60}")
    print(f"  Total participant assets: {len(sys4_participant)}")
    print(f"  Present in sanitized: {len(present)}")
    print(f"  Missing from sanitized: {len(missing)}")

    if missing:
        show = missing[:20]
        print(f"\n  Missing assets (showing {len(show)} of {len(missing)}):")
        for key in show:
            print(f"    - {key}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more (see JSON report)")

    if present:
        print(f"\n  Sample present assets (showing up to 5):")
        for item in present[:5]:
            print(f"    - {item['dedup_key']} -> {item['found_at']}")

    # Write report JSON
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket": source_bucket,
        "sanitized_bucket": SANITIZED_BUCKET,
        "total_participant_assets": len(sys4_participant),
        "present_in_sanitized": len(present),
        "missing_from_sanitized": len(missing),
        "present_details": present,
        "missing_keys": missing,
    }
    report_path = Path(__file__).resolve().parent / f"audit_system4_{int(datetime.now().timestamp())}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Subcommand: migrate-sources
# ---------------------------------------------------------------------------

def run_migrate_sources(args):
    """Migrate source images to sanitized bucket and convert all URLs to public format."""
    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(CENTRAL_TABLE)
    dry_run = args.dry_run
    mode = "[DRY-RUN] " if dry_run else ""

    # Scan central table
    print(f"Scanning {CENTRAL_TABLE}...")
    db_items: list[dict] = []
    scan_kwargs: dict = {}
    while True:
        resp = table.scan(**scan_kwargs)
        db_items.extend(resp.get("Items", []))
        last = resp.get("LastEvaluatedKey")
        if not last:
            break
        scan_kwargs["ExclusiveStartKey"] = last
    print(f"  Found {len(db_items)} records")

    # List existing assets in sanitized bucket
    print(f"\nListing existing assets in {SANITIZED_BUCKET}...")
    existing_keys = set(list_s3_images(s3, SANITIZED_BUCKET))
    print(f"  Found {len(existing_keys)} existing assets")

    # Phase 1: Copy missing source images to sanitized bucket
    print(f"\n{'='*60}")
    print(f"  {mode}Phase 1: Copy source images to sanitized bucket")
    print(f"{'='*60}")

    copied = 0
    already_exists = 0
    copy_failed: list[dict] = []
    skipped_no_uuid = 0

    for item in db_items:
        sib = item.get("source_image_bucket", "")
        sik = item.get("source_image_key", "")
        if not sib or not sik:
            continue

        sys_num = SYSTEM_MAP.get(sib)
        if not sys_num:
            continue

        bare_key = strip_bucket_prefix(sik)
        parts = bare_key.split("/")

        # Find UUID and filename
        uuid_part = None
        for p in parts:
            if UUID_RE.match(p):
                uuid_part = p
                break
        if not uuid_part:
            skipped_no_uuid += 1
            continue

        filename = parts[-1]
        dest_key = f"selected-images/system-{sys_num}/{uuid_part}/{filename}"

        if dest_key in existing_keys:
            already_exists += 1
            continue

        if dry_run:
            copied += 1
        else:
            try:
                s3.copy_object(
                    Bucket=SANITIZED_BUCKET,
                    Key=dest_key,
                    CopySource={"Bucket": sib, "Key": bare_key},
                )
                existing_keys.add(dest_key)
                copied += 1
            except Exception as e:
                print(f"  FAILED: {sib}/{bare_key} -> {e}")
                copy_failed.append({"source": f"{sib}/{bare_key}", "dest": dest_key, "error": str(e)})

        if copied % 100 == 0 and copied > 0:
            print(f"  {'Would copy' if dry_run else 'Copied'} {copied}...")

    print(f"\n  {mode}Phase 1 summary:")
    print(f"    Already in sanitized: {already_exists}")
    print(f"    Copied: {copied}")
    print(f"    Failed: {len(copy_failed)}")
    print(f"    Skipped (no UUID): {skipped_no_uuid}")

    if copy_failed:
        print(f"\n  WARNING: {len(copy_failed)} copies failed — skipping DynamoDB updates for those records.")

    # Phase 2: Update all DynamoDB records — redirect source refs + convert all URLs to public
    print(f"\n{'='*60}")
    print(f"  {mode}Phase 2: Update DynamoDB records (redirect + public URLs)")
    print(f"{'='*60}")

    updated = 0
    update_failed: list[dict] = []
    failed_sources = {f["source"] for f in copy_failed}

    for item in db_items:
        blend_id = item["blend_id"]
        sib = item.get("source_image_bucket", "")
        sik = item.get("source_image_key", "")
        rib = item.get("random_image_bucket", "")
        rik = item.get("random_image_key", "")

        # Build the new source key in sanitized
        new_source_key = None
        sys_num = SYSTEM_MAP.get(sib)
        if sys_num and sik:
            bare_key = strip_bucket_prefix(sik)
            parts = bare_key.split("/")
            uuid_part = None
            for p in parts:
                if UUID_RE.match(p):
                    uuid_part = p
                    break
            if uuid_part:
                filename = parts[-1]
                new_source_key = f"selected-images/system-{sys_num}/{uuid_part}/{filename}"

                # Skip if the copy failed for this asset
                if f"{sib}/{bare_key}" in failed_sources:
                    continue

        # Build the bare random key
        new_random_key = strip_bucket_prefix(rik) if rik else rik

        # Build update expression
        updates = {}
        names = {}
        values = {}
        idx = 0

        # Source image: redirect to sanitized
        if new_source_key:
            updates[f"#f{idx}"] = f":v{idx}"
            names[f"#f{idx}"] = "source_image_key"
            values[f":v{idx}"] = new_source_key
            idx += 1
            updates[f"#f{idx}"] = f":v{idx}"
            names[f"#f{idx}"] = "source_image_bucket"
            values[f":v{idx}"] = SANITIZED_BUCKET
            idx += 1
            updates[f"#f{idx}"] = f":v{idx}"
            names[f"#f{idx}"] = "source_image_url"
            values[f":v{idx}"] = public_url(SANITIZED_BUCKET, new_source_key)
            idx += 1

        # Random image URL: convert to public
        if rik and rib:
            target_bucket = SANITIZED_BUCKET if rib == SANITIZED_BUCKET else rib
            updates[f"#f{idx}"] = f":v{idx}"
            names[f"#f{idx}"] = "random_image_url"
            values[f":v{idx}"] = public_url(target_bucket, new_random_key)
            idx += 1
            # Also normalize the key (strip bucket prefix if present)
            if rik != new_random_key:
                updates[f"#f{idx}"] = f":v{idx}"
                names[f"#f{idx}"] = "random_image_key"
                values[f":v{idx}"] = new_random_key
                idx += 1

        # Video URLs: convert to public
        for video_field, url_field in [("s3_key", "blend_url"), ("s3_key", "video_url"), ("s3_key", "videoUrl")]:
            video_key = item.get("s3_video_key", "") or item.get("s3_key", "")
            old_url = item.get(url_field, "")
            if old_url and video_key:
                blend_bucket = item.get("blend_bucket", sib)
                bare_video = strip_bucket_prefix(video_key)
                updates[f"#f{idx}"] = f":v{idx}"
                names[f"#f{idx}"] = url_field
                values[f":v{idx}"] = public_url(blend_bucket, bare_video)
                idx += 1

        if not updates:
            continue

        expr = "SET " + ", ".join(f"{k} = {v}" for k, v in updates.items())

        if dry_run:
            updated += 1
        else:
            try:
                table.update_item(
                    Key={"blend_id": blend_id},
                    UpdateExpression=expr,
                    ExpressionAttributeNames=names,
                    ExpressionAttributeValues=values,
                )
                updated += 1
            except Exception as e:
                print(f"  FAILED updating {blend_id}: {e}")
                update_failed.append({"blend_id": blend_id, "error": str(e)})

        if updated % 200 == 0 and updated > 0:
            print(f"  {'Would update' if dry_run else 'Updated'} {updated}...")

    print(f"\n  {mode}Phase 2 summary:")
    print(f"    Records updated: {updated}")
    print(f"    Update failures: {len(update_failed)}")

    # Write log
    log = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "phase1_copied": copied,
        "phase1_already_exists": already_exists,
        "phase1_failed": len(copy_failed),
        "phase1_failed_items": copy_failed,
        "phase2_updated": updated,
        "phase2_failed": len(update_failed),
        "phase2_failed_items": update_failed,
    }
    log_path = Path(__file__).resolve().parent / f"migrate_sources_log_{int(datetime.now().timestamp())}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\nLog saved to: {log_path}")


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

    # migrate-sources
    ms = sub.add_parser("migrate-sources",
                        help="Migrate source images to sanitized bucket + convert URLs to public")
    ms.add_argument("--dry-run", action="store_true", help="Preview without changes")

    # audit-system4
    sub.add_parser("audit-system4", help="Diagnostic report of system 4 coverage")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "migrate":
        run_migrate(args)
    elif args.command == "dedup":
        run_dedup(args)
    elif args.command == "migrate-sources":
        run_migrate_sources(args)
    elif args.command == "audit-system4":
        run_audit_system4(args)


if __name__ == "__main__":
    main()
