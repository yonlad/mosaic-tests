"""
Generate per-system CSV files mapping each participant's email to their mosaic URL in S3.

Cross-references:
  - System CSVs (dynamodb-tables/system{1-4}.csv) for email + user_id
  - S3 mosaics folders for actual mosaic image files
  - eternity-mirror-users DynamoDB table for fallback email + mosaic_key
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import boto3
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

SYSTEMS = {
    1: {"bucket": "pistoletto.moe", "csv": "system1.csv", "users_table": "eternity-mirror-users"},
    2: {"bucket": "pistoletto.moe2", "csv": "system2.csv", "users_table": "eternity-mirror-users-2"},
    3: {"bucket": "pistoletto.moe3", "csv": "system3.csv", "users_table": "eternity-mirror-users-3"},
    4: {"bucket": "pistoletto.moe4", "csv": "system4.csv", "users_table": "eternity-mirror-users-4"},
}

DYNAMO_TABLES_DIR = Path(__file__).resolve().parent / "dynamodb-tables"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

S3_BASE_URL = f"https://s3.{AWS_REGION}.amazonaws.com"


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_dynamodb_resource():
    return boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# ── Step 1: Load system CSVs ────────────────────────────────────────────────

def load_system_csv(system_num: int) -> dict[str, dict]:
    """Return {user_id: {email, source_image_url, created_at}} from a system CSV."""
    csv_path = DYNAMO_TABLES_DIR / SYSTEMS[system_num]["csv"]
    users = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id", "").strip()
            if not uid:
                continue
            # Keep first occurrence (or overwrite — last email wins for dupes)
            users[uid] = {
                "email": row.get("email", "").strip(),
                "source_image_url": row.get("source_image_url", "").strip(),
                "created_at": row.get("created_at", "").strip(),
            }
    return users


# ── Step 2: List S3 mosaic objects ───────────────────────────────────────────

def list_s3_mosaics(s3, bucket: str) -> dict[str, dict]:
    """Return {user_id: {mosaic_key, thumbnail_key}} from the mosaics/ prefix."""
    user_files: dict[str, dict] = defaultdict(lambda: {"mosaic_key": None, "thumbnail_key": None})
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix="mosaics/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            # Expected: mosaics/<user_id>/<filename>
            if len(parts) < 3:
                continue
            uid = parts[1]
            filename = parts[2]

            if "thumbnail" in filename:
                # Prefer .webp thumbnail
                if filename.endswith(".webp"):
                    user_files[uid]["thumbnail_key"] = key
                elif not user_files[uid]["thumbnail_key"]:
                    user_files[uid]["thumbnail_key"] = key
            elif filename.endswith(".jpg"):
                # Full-res mosaic — keep the latest (last alphabetically = latest timestamp)
                existing = user_files[uid]["mosaic_key"]
                if not existing or key > existing:
                    user_files[uid]["mosaic_key"] = key

    return dict(user_files)


# ── Step 3: Query eternity-mirror-users DynamoDB table ───────────────────────

def scan_users_table(dynamodb, table_name: str) -> dict[str, dict]:
    """Return {user_id: {email, mosaic_key, selected_image_key, created_at}}."""
    table = dynamodb.Table(table_name)
    users = {}
    scan_kwargs = {}

    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp.get("Items", []):
            uid = item.get("user_id", "")
            if not uid:
                continue
            users[uid] = {
                "email": item.get("email", ""),
                "mosaic_key": item.get("mosaic_key", ""),
                "selected_image_key": item.get("selected_image_key", ""),
                "created_at": item.get("created_at", ""),
            }
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return users


# ── Step 4 & 5: Cross-reference and write output ────────────────────────────

def strip_query_params(url: str) -> str:
    """Remove presigned URL query parameters — objects are publicly readable now."""
    if not url:
        return ""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query=""))


def list_s3_source_images(s3, bucket: str) -> set[str]:
    """Return set of user_ids that have a source image folder in S3."""
    user_ids = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="selected-images/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            uid = cp["Prefix"].replace("selected-images/", "").rstrip("/")
            if uid:
                user_ids.add(uid)
    return user_ids


def build_url(bucket: str, key: str) -> str:
    if not key:
        return ""
    return f"{S3_BASE_URL}/{bucket}/{key}"


def process_system(system_num: int, s3, users_table: dict[str, dict]):
    bucket = SYSTEMS[system_num]["bucket"]
    print(f"\n{'='*60}")
    print(f"System {system_num} — bucket: {bucket}")
    print(f"{'='*60}")

    # Step 1: Load CSV
    csv_users = load_system_csv(system_num)
    print(f"  CSV users: {len(csv_users)}")

    # Step 2: List S3 mosaics
    s3_mosaics = list_s3_mosaics(s3, bucket)
    print(f"  S3 mosaic folders: {len(s3_mosaics)}")

    # Step 2b: List S3 source images to filter out deleted ones
    s3_source_images = list_s3_source_images(s3, bucket)
    print(f"  S3 source image folders: {len(s3_source_images)}")

    # Step 4: Cross-reference
    # Start with all user_ids from both CSV and S3
    all_user_ids = set(csv_users.keys()) | set(s3_mosaics.keys())
    print(f"  Combined unique user_ids: {len(all_user_ids)}")

    rows = []
    missing_email = 0
    missing_mosaic = 0
    skipped_no_source = 0

    for uid in sorted(all_user_ids):
        # Skip users whose source image no longer exists in S3
        if uid not in s3_source_images:
            skipped_no_source += 1
            continue

        csv_data = csv_users.get(uid, {})
        s3_data = s3_mosaics.get(uid, {})
        dynamo_data = users_table.get(uid, {})

        # Resolve email: prefer CSV, fallback to DynamoDB
        email = csv_data.get("email") or dynamo_data.get("email", "")

        # Resolve mosaic files from S3
        mosaic_key = s3_data.get("mosaic_key", "") if s3_data else ""
        thumbnail_key = s3_data.get("thumbnail_key", "") if s3_data else ""

        # Resolve source image URL: prefer CSV, fallback to DynamoDB selected_image_key
        source_image_url = strip_query_params(csv_data.get("source_image_url", ""))
        if not source_image_url and dynamo_data.get("selected_image_key"):
            source_image_url = build_url(bucket, dynamo_data["selected_image_key"])

        # Resolve created_at: prefer CSV, fallback to DynamoDB
        created_at = csv_data.get("created_at") or dynamo_data.get("created_at", "")

        if not email:
            missing_email += 1
        if not mosaic_key:
            missing_mosaic += 1

        rows.append({
            "email": email,
            "user_id": uid,
            "mosaic_url": build_url(bucket, mosaic_key),
            "mosaic_thumbnail_url": build_url(bucket, thumbnail_key),
            "source_image_url": source_image_url,
            "created_at": created_at,
        })

    print(f"  Skipped (no source image): {skipped_no_source}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Missing email: {missing_email}")
    print(f"  Missing mosaic: {missing_mosaic}")

    # Step 5: Write output CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"system{system_num}_mosaics.csv"
    fieldnames = ["email", "user_id", "mosaic_url", "mosaic_thumbnail_url", "source_image_url", "created_at"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written to: {output_path}")
    return len(rows)


def main():
    print("Initializing AWS clients...")
    s3 = get_s3_client()
    dynamodb = get_dynamodb_resource()

    total = 0
    for sys_num in range(1, 5):
        table_name = SYSTEMS[sys_num]["users_table"]
        print(f"\nScanning {table_name}...")
        users_table = scan_users_table(dynamodb, table_name)
        print(f"  Users table entries: {len(users_table)}")
        total += process_system(sys_num, s3, users_table)

    print(f"\nDone! Total rows across all systems: {total}")


if __name__ == "__main__":
    main()
