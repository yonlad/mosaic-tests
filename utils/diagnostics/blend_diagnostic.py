#!/usr/bin/env python3
"""
Diagnostic report for Mirror of Eternity blend & mosaic production.

Scans eternity-mirror-blends, eternity-mirror-blends-2, eternity-mirror-users,
and eternity-mirror-users-2 for the period 2026-04-15 to 2026-06-15.

Produces a markdown report with:
  - Overall counts and success/failure percentages
  - Failure analysis (error types, timing patterns)
  - Mosaic production analysis (users with/without mosaics)
  - Cross-system comparison
  - Weekly breakdown
"""

import boto3
import os
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

DATE_START = "2026-04-15"
DATE_END = "2026-06-15"

SYSTEMS = {
    1: {
        "blend_table": "eternity-mirror-blends",
        "user_table": "eternity-mirror-users",
        "bucket": "pistoletto.moe",
    },
    2: {
        "blend_table": "eternity-mirror-blends-2",
        "user_table": "eternity-mirror-users-2",
        "bucket": "pistoletto.moe2",
    },
}

# Systems 3 & 4 have user/mosaic data but are not included in the
# blend diagnostic date-range analysis — only in the all-time S3 inventory.
MOSAIC_ONLY_SYSTEMS = {
    3: {
        "user_table": "eternity-mirror-users-3",
        "bucket": "pistoletto.moe3",
    },
    4: {
        "user_table": "eternity-mirror-users-4",
        "bucket": "pistoletto.moe4",
    },
}


# ── AWS clients ───────────────────────────────────────────────────────────
def get_dynamodb():
    return boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# ── Scanning ──────────────────────────────────────────────────────────────
def full_scan(dynamodb, table_name):
    table = dynamodb.Table(table_name)
    items = []
    kwargs = {}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        last = resp.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    return items


def filter_by_date(items, date_field="created_at", start=DATE_START, end=DATE_END):
    filtered = []
    for item in items:
        val = item.get(date_field, "")
        if val and start <= val[:10] <= end:
            filtered.append(item)
    return filtered


# ── S3 asset verification (spot-check) ───────────────────────────────────
def check_s3_keys_exist(s3, bucket, keys_to_check):
    """Check a list of S3 keys and return set of missing ones."""
    missing = set()
    for key in keys_to_check:
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except Exception:
            missing.add(key)
    return missing


# ── Analysis helpers ──────────────────────────────────────────────────────
def week_label(date_str):
    """Return ISO week label like '2026-W16'."""
    dt = datetime.fromisoformat(date_str[:19])
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def analyze_blends(items, system_num):
    """Analyze blend items for a single system. Returns a dict of stats."""
    total = len(items)
    if total == 0:
        return {"total": 0}

    statuses = Counter(i.get("status", "unknown") for i in items)
    completed = statuses.get("completed", 0)
    failed = total - completed

    # Error breakdown
    errors = []
    for i in items:
        if i.get("status") != "completed":
            errors.append({
                "blend_id": i.get("blend_id"),
                "status": i.get("status"),
                "error_message": i.get("error_message", "N/A"),
                "created_at": i.get("created_at", ""),
                "email": i.get("email", ""),
                "user_id": i.get("user_id", ""),
            })

    # Error type grouping
    error_types = Counter()
    for e in errors:
        msg = e["error_message"]
        if "timed out" in msg.lower() or "timeout" in msg.lower():
            error_types["API Timeout"] += 1
        elif "polling" in msg.lower() or "max polling" in msg.lower():
            error_types["Polling Timeout"] += 1
        elif "background removal" in msg.lower():
            error_types["Background Removal Failure"] += 1
        elif msg == "N/A":
            error_types["Unknown (no error message)"] += 1
        else:
            error_types[msg[:80]] += 1

    # Weekly breakdown
    weekly = defaultdict(lambda: {"total": 0, "completed": 0, "failed": 0})
    for i in items:
        w = week_label(i.get("created_at", DATE_START))
        weekly[w]["total"] += 1
        if i.get("status") == "completed":
            weekly[w]["completed"] += 1
        else:
            weekly[w]["failed"] += 1

    # Check for missing critical fields on "completed" items
    completed_items = [i for i in items if i.get("status") == "completed"]
    missing_blend_url = sum(1 for i in completed_items if not i.get("blend_url"))
    missing_s3_key = sum(1 for i in completed_items if not i.get("s3_key"))
    missing_video = sum(1 for i in completed_items if not i.get("blend_url") and not i.get("video_url"))

    # Time-of-day distribution for failures
    failure_hours = Counter()
    for e in errors:
        try:
            dt = datetime.fromisoformat(e["created_at"][:19])
            failure_hours[dt.hour] += 1
        except Exception:
            pass

    return {
        "total": total,
        "statuses": dict(statuses),
        "completed": completed,
        "failed": failed,
        "success_rate": round(completed / total * 100, 1) if total else 0,
        "failure_rate": round(failed / total * 100, 1) if total else 0,
        "errors": errors,
        "error_types": dict(error_types),
        "weekly": dict(weekly),
        "missing_blend_url": missing_blend_url,
        "missing_s3_key": missing_s3_key,
        "missing_video": missing_video,
        "failure_hours": dict(failure_hours),
    }


def analyze_users(items, system_num):
    """Analyze user items for mosaic production."""
    total = len(items)
    if total == 0:
        return {"total": 0}

    has_mosaic = sum(1 for i in items if i.get("mosaic_key"))
    no_mosaic = total - has_mosaic

    # Users with selected image but no mosaic
    has_image_no_mosaic = sum(
        1 for i in items
        if i.get("selected_image_key") and not i.get("mosaic_key")
    )

    # Users with neither
    no_image_no_mosaic = sum(
        1 for i in items
        if not i.get("selected_image_key") and not i.get("mosaic_key")
    )

    # Users without mosaic detail
    users_no_mosaic = []
    for i in items:
        if not i.get("mosaic_key"):
            users_no_mosaic.append({
                "user_id": i.get("user_id", "?"),
                "email": i.get("email", "?"),
                "created_at": i.get("created_at", "?"),
                "has_selected_image": bool(i.get("selected_image_key")),
            })

    return {
        "total": total,
        "has_mosaic": has_mosaic,
        "no_mosaic": no_mosaic,
        "mosaic_rate": round(has_mosaic / total * 100, 1) if total else 0,
        "has_image_no_mosaic": has_image_no_mosaic,
        "no_image_no_mosaic": no_image_no_mosaic,
        "users_no_mosaic": users_no_mosaic,
    }


# ── S3 mosaic folder inventory ────────────────────────────────────────────
def inventory_mosaic_folders(s3, dynamodb, system_cfg):
    """
    Count all mosaic folders in S3 and cross-reference with DynamoDB users.
    Returns a dict with all-time inventory stats.
    """
    bucket = system_cfg["bucket"]
    user_table = system_cfg["user_table"]

    # Paginate to get all sub-folders under mosaics/
    s3_user_ids = set()
    kwargs = {"Bucket": bucket, "Prefix": "mosaics/", "Delimiter": "/"}
    total_objects = 0
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for p in resp.get("CommonPrefixes", []):
            uid = p["Prefix"].split("/")[1]
            s3_user_ids.add(uid)
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    # Count total objects under mosaics/
    obj_kwargs = {"Bucket": bucket, "Prefix": "mosaics/"}
    while True:
        resp = s3.list_objects_v2(**obj_kwargs)
        total_objects += resp.get("KeyCount", 0)
        if resp.get("IsTruncated"):
            obj_kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break

    # Get all users from DynamoDB
    all_users = full_scan(dynamodb, user_table)
    db_user_ids = {i["user_id"] for i in all_users}
    db_with_mosaic = {i["user_id"] for i in all_users if i.get("mosaic_key")}
    db_without_mosaic = db_user_ids - db_with_mosaic

    matched = db_with_mosaic & s3_user_ids
    phantom = db_with_mosaic - s3_user_ids  # DB says yes, S3 says no
    orphan = s3_user_ids - db_user_ids       # S3 has it, DB doesn't know

    return {
        "s3_mosaic_folders": len(s3_user_ids),
        "s3_total_objects": total_objects,
        "db_total_users": len(db_user_ids),
        "db_with_mosaic_key": len(db_with_mosaic),
        "db_without_mosaic_key": len(db_without_mosaic),
        "verified_match": len(matched),
        "phantom_mosaics": len(phantom),
        "phantom_user_ids": sorted(phantom),
        "orphan_mosaics": len(orphan),
        "orphan_user_ids": sorted(orphan),
    }


# ── S3 spot-check on failed blends ───────────────────────────────────────
def spot_check_s3_assets(s3, blend_items, system_cfg, sample_size=20):
    """Spot-check S3 assets for a sample of completed blends."""
    completed = [i for i in blend_items if i.get("status") == "completed"]
    sample = completed[:sample_size]
    bucket = system_cfg["bucket"]

    missing_source = 0
    missing_blend = 0
    checked = 0

    for item in sample:
        checked += 1
        src_key = item.get("source_image_key", "")
        blend_key = item.get("s3_key", "")

        if src_key:
            try:
                s3.head_object(Bucket=bucket, Key=src_key)
            except Exception:
                missing_source += 1

        if blend_key:
            try:
                s3.head_object(Bucket=bucket, Key=blend_key)
            except Exception:
                missing_blend += 1

    return {
        "checked": checked,
        "missing_source_images": missing_source,
        "missing_blend_videos": missing_blend,
    }


# ── Report generation ─────────────────────────────────────────────────────
def generate_report(sys_blend_stats, sys_user_stats, s3_checks, s3_inventories):
    lines = []
    lines.append("# Mirror of Eternity — Diagnostic Report")
    lines.append(f"**Period:** {DATE_START} to {DATE_END}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    total_blends = sum(s["total"] for s in sys_blend_stats.values())
    total_completed = sum(s.get("completed", 0) for s in sys_blend_stats.values())
    total_failed = sum(s.get("failed", 0) for s in sys_blend_stats.values())
    total_users = sum(s["total"] for s in sys_user_stats.values())
    total_mosaics = sum(s.get("has_mosaic", 0) for s in sys_user_stats.values())

    overall_blend_rate = round(total_completed / total_blends * 100, 1) if total_blends else 0
    overall_mosaic_rate = round(total_mosaics / total_users * 100, 1) if total_users else 0

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total blend attempts (both systems) | {total_blends} |")
    lines.append(f"| Successful blends | {total_completed} |")
    lines.append(f"| Failed blends | {total_failed} |")
    lines.append(f"| **Overall blend success rate** | **{overall_blend_rate}%** |")
    lines.append(f"| Total users registered (both systems) | {total_users} |")
    lines.append(f"| Users with mosaic produced | {total_mosaics} |")
    lines.append(f"| Users without mosaic | {total_users - total_mosaics} |")
    lines.append(f"| **Overall mosaic production rate** | **{overall_mosaic_rate}%** |")
    lines.append("")

    # ── Per-System Blend Analysis
    for sys_num in sorted(sys_blend_stats.keys()):
        stats = sys_blend_stats[sys_num]
        if stats["total"] == 0:
            continue

        lines.append(f"## System {sys_num} — Blend Analysis")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total blend attempts | {stats['total']} |")
        lines.append(f"| Completed | {stats['completed']} ({stats['success_rate']}%) |")
        lines.append(f"| Failed | {stats['failed']} ({stats['failure_rate']}%) |")
        lines.append(f"| Completed items missing blend_url | {stats['missing_blend_url']} |")
        lines.append(f"| Completed items missing s3_key | {stats['missing_s3_key']} |")
        lines.append("")

        # Status breakdown
        lines.append(f"### Status Distribution")
        lines.append("")
        lines.append(f"| Status | Count | Percentage |")
        lines.append(f"|--------|-------|------------|")
        for status, count in sorted(stats["statuses"].items(), key=lambda x: -x[1]):
            pct = round(count / stats["total"] * 100, 1)
            lines.append(f"| {status} | {count} | {pct}% |")
        lines.append("")

        # Error analysis
        if stats["errors"]:
            lines.append(f"### Failure Analysis")
            lines.append("")
            lines.append(f"**Error type breakdown:**")
            lines.append("")
            lines.append(f"| Error Type | Count |")
            lines.append(f"|-----------|-------|")
            for etype, count in sorted(stats["error_types"].items(), key=lambda x: -x[1]):
                lines.append(f"| {etype} | {count} |")
            lines.append("")

            lines.append(f"**Failed blend details:**")
            lines.append("")
            lines.append(f"| Blend ID | Status | Created | Error |")
            lines.append(f"|----------|--------|---------|-------|")
            for e in stats["errors"]:
                err_short = e["error_message"][:60]
                lines.append(f"| `{e['blend_id'][:12]}...` | {e['status']} | {e['created_at'][:16]} | {err_short} |")
            lines.append("")

            # Time of day for failures
            if stats["failure_hours"]:
                lines.append(f"**Failure time-of-day (UTC):**")
                lines.append("")
                for h in sorted(stats["failure_hours"].keys()):
                    lines.append(f"- {h:02d}:00 UTC — {stats['failure_hours'][h]} failure(s)")
                lines.append("")

        # Weekly breakdown
        if stats["weekly"]:
            lines.append(f"### Weekly Breakdown")
            lines.append("")
            lines.append(f"| Week | Total | Completed | Failed | Success Rate |")
            lines.append(f"|------|-------|-----------|--------|-------------|")
            for w in sorted(stats["weekly"].keys()):
                wd = stats["weekly"][w]
                rate = round(wd["completed"] / wd["total"] * 100, 1) if wd["total"] else 0
                lines.append(f"| {w} | {wd['total']} | {wd['completed']} | {wd['failed']} | {rate}% |")
            lines.append("")

        # S3 spot check
        if sys_num in s3_checks:
            sc = s3_checks[sys_num]
            lines.append(f"### S3 Asset Spot-Check ({sc['checked']} completed blends sampled)")
            lines.append("")
            lines.append(f"- Missing source images in bucket: **{sc['missing_source_images']}**")
            lines.append(f"- Missing blend videos in bucket: **{sc['missing_blend_videos']}**")
            lines.append("")

    # ── Per-System User/Mosaic Analysis
    for sys_num in sorted(sys_user_stats.keys()):
        stats = sys_user_stats[sys_num]
        if stats["total"] == 0:
            continue

        lines.append(f"## System {sys_num} — Mosaic Production Analysis")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total users registered | {stats['total']} |")
        lines.append(f"| Users with mosaic produced | {stats['has_mosaic']} ({stats['mosaic_rate']}%) |")
        lines.append(f"| Users without mosaic | {stats['no_mosaic']} ({round(100 - stats['mosaic_rate'], 1)}%) |")
        lines.append(f"| Users with photo selected but no mosaic | {stats['has_image_no_mosaic']} |")
        lines.append(f"| Users with no photo and no mosaic | {stats['no_image_no_mosaic']} |")
        lines.append("")

        # List users without mosaic (limit to first 15)
        no_mosaic_list = stats.get("users_no_mosaic", [])
        if no_mosaic_list:
            lines.append(f"### Users Without Mosaic (showing up to 15)")
            lines.append("")
            lines.append(f"| User ID | Email | Created | Has Selected Image |")
            lines.append(f"|---------|-------|---------|-------------------|")
            for u in no_mosaic_list[:15]:
                uid = u["user_id"][:12] + "..." if len(u["user_id"]) > 12 else u["user_id"]
                lines.append(f"| `{uid}` | {u['email']} | {u['created_at'][:16]} | {'Yes' if u['has_selected_image'] else 'No'} |")
            if len(no_mosaic_list) > 15:
                lines.append(f"| ... | *{len(no_mosaic_list) - 15} more* | | |")
            lines.append("")

    # ── S3 Mosaic Folder Inventory (all-time)
    lines.append("## S3 Mosaic Folder Inventory (All-Time)")
    lines.append("")
    lines.append("This section counts the actual mosaic folders in each system's S3 bucket and cross-references")
    lines.append("with the DynamoDB user tables to verify data consistency.")
    lines.append("")

    all_system_cfgs = {**SYSTEMS, **MOSAIC_ONLY_SYSTEMS}
    for sys_num in sorted(s3_inventories.keys()):
        inv = s3_inventories[sys_num]
        bucket = all_system_cfgs[sys_num]["bucket"]
        avg_files = round(inv["s3_total_objects"] / inv["s3_mosaic_folders"], 1) if inv["s3_mosaic_folders"] else 0

        lines.append(f"### System {sys_num} — `{bucket}`")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| S3 mosaic folders (unique users) | {inv['s3_mosaic_folders']:,} |")
        lines.append(f"| S3 total mosaic objects | {inv['s3_total_objects']:,} |")
        lines.append(f"| Avg files per user folder | {avg_files} |")
        lines.append(f"| DynamoDB total users | {inv['db_total_users']:,} |")
        lines.append(f"| DynamoDB users with `mosaic_key` | {inv['db_with_mosaic_key']:,} |")
        lines.append(f"| DynamoDB users without `mosaic_key` | {inv['db_without_mosaic_key']:,} |")
        lines.append(f"| **DB `mosaic_key` exists AND S3 folder exists** | **{inv['verified_match']:,}** |")
        lines.append(f"| DB `mosaic_key` exists but S3 folder missing | {inv['phantom_mosaics']:,} |")
        lines.append(f"| S3 folder exists but not in DB | {inv['orphan_mosaics']:,} |")
        lines.append("")

        if inv["phantom_mosaics"] > 0:
            lines.append(f"**{inv['phantom_mosaics']} phantom mosaic(s)** — these users have a `mosaic_key` recorded in DynamoDB but the actual")
            lines.append(f"files are missing from S3.")
            lines.append("")

    # Combined totals
    lines.append("### Combined All-Time Totals")
    lines.append("")
    lines.append("| Metric | " + " | ".join(f"System {n}" for n in sorted(s3_inventories)) + " | Total |")
    lines.append("|--------|" + "|".join("----------|" for _ in s3_inventories) + "-------|")
    for label, key in [
        ("S3 mosaic folders", "s3_mosaic_folders"),
        ("DynamoDB users", "db_total_users"),
        ("DB users with mosaic_key", "db_with_mosaic_key"),
        ("Verified mosaics (DB + S3)", "verified_match"),
        ("Phantom mosaics (DB yes, S3 no)", "phantom_mosaics"),
        ("Orphan mosaics (S3 yes, DB no)", "orphan_mosaics"),
    ]:
        vals = [s3_inventories[n][key] for n in sorted(s3_inventories)]
        total = sum(vals)
        cells = " | ".join(f"{v:,}" for v in vals)
        lines.append(f"| {label} | {cells} | **{total:,}** |")
    lines.append("")

    # ── Cross-system comparison
    lines.append("## Cross-System Comparison")
    lines.append("")
    lines.append("| Metric | System 1 | System 2 |")
    lines.append("|--------|----------|----------|")

    for metric, key in [
        ("Total blends", "total"),
        ("Completed", "completed"),
        ("Failed", "failed"),
        ("Success rate", "success_rate"),
    ]:
        v1 = sys_blend_stats.get(1, {}).get(key, "N/A")
        v2 = sys_blend_stats.get(2, {}).get(key, "N/A")
        if key == "success_rate":
            v1 = f"{v1}%" if v1 != "N/A" else "N/A"
            v2 = f"{v2}%" if v2 != "N/A" else "N/A"
        lines.append(f"| {metric} | {v1} | {v2} |")

    for metric, key in [
        ("Registered users (in period)", "total"),
        ("Mosaics produced (in period)", "has_mosaic"),
        ("Mosaic rate (in period)", "mosaic_rate"),
    ]:
        v1 = sys_user_stats.get(1, {}).get(key, "N/A")
        v2 = sys_user_stats.get(2, {}).get(key, "N/A")
        if key == "mosaic_rate":
            v1 = f"{v1}%" if v1 != "N/A" else "N/A"
            v2 = f"{v2}%" if v2 != "N/A" else "N/A"
        lines.append(f"| {metric} | {v1} | {v2} |")

    for metric, key in [
        ("S3 mosaic folders (all-time)", "s3_mosaic_folders"),
        ("DynamoDB users (all-time)", "db_total_users"),
        ("Verified mosaics (all-time)", "verified_match"),
    ]:
        v1 = s3_inventories.get(1, {}).get(key, "N/A")
        v2 = s3_inventories.get(2, {}).get(key, "N/A")
        if isinstance(v1, int):
            v1 = f"{v1:,}"
        if isinstance(v2, int):
            v2 = f"{v2:,}"
        lines.append(f"| {metric} | {v1} | {v2} |")
    lines.append("")

    # ── Key Findings
    lines.append("## Key Findings & Recommendations")
    lines.append("")
    findings = []

    if total_failed > 0:
        all_error_types = Counter()
        for s in sys_blend_stats.values():
            for et, c in s.get("error_types", {}).items():
                all_error_types[et] += c
        top_error = all_error_types.most_common(1)
        if top_error:
            findings.append(
                f"**Top failure cause:** {top_error[0][0]} ({top_error[0][1]} occurrence(s)). "
                f"This accounts for {round(top_error[0][1] / total_failed * 100)}% of all failures."
            )

    if total_failed == 0:
        findings.append("No blend failures were recorded in this period — excellent reliability.")
    elif total_failed <= 5:
        findings.append(
            f"Only {total_failed} blend failure(s) out of {total_blends} total — "
            f"the system is highly reliable ({overall_blend_rate}% success rate)."
        )

    for sys_num, us in sys_user_stats.items():
        nim = us.get("has_image_no_mosaic", 0)
        if nim > 0:
            findings.append(
                f"**System {sys_num}:** {nim} user(s) selected a photo but never received a mosaic. "
                f"This may indicate mosaic generation failures or abandoned sessions."
            )

    for sys_num, sc in s3_checks.items():
        if sc["missing_blend_videos"] > 0:
            findings.append(
                f"**System {sys_num} S3 spot-check:** {sc['missing_blend_videos']} of "
                f"{sc['checked']} sampled blend videos are missing from the S3 bucket. "
                f"This suggests potential S3 cleanup or data loss."
            )
        if sc["missing_source_images"] > 0:
            findings.append(
                f"**System {sys_num} S3 spot-check:** {sc['missing_source_images']} of "
                f"{sc['checked']} sampled source images are missing from the S3 bucket."
            )

    # Phantom mosaic findings
    total_phantoms = sum(inv.get("phantom_mosaics", 0) for inv in s3_inventories.values())
    total_verified = sum(inv.get("verified_match", 0) for inv in s3_inventories.values())
    if total_phantoms > 0:
        per_sys = ", ".join(
            f"{inv['phantom_mosaics']} in System {n}"
            for n, inv in sorted(s3_inventories.items())
            if inv["phantom_mosaics"] > 0
        )
        findings.append(
            f"**{total_phantoms} phantom mosaic(s) detected (all-time):** {per_sys} have a "
            f"`mosaic_key` in DynamoDB but the corresponding S3 files no longer exist. "
            f"These DB records should be audited."
        )
    findings.append(
        f"**All-time mosaic production:** {total_verified:,} verified mosaics across all {len(s3_inventories)} systems, "
        f"confirmed present in both DynamoDB and S3."
    )
    total_orphans = sum(inv.get("orphan_mosaics", 0) for inv in s3_inventories.values())
    if total_orphans == 0:
        findings.append(
            "**Zero orphan S3 mosaics:** Every mosaic folder in S3 corresponds to a known user "
            "in DynamoDB — no wasted storage."
        )

    if not findings:
        findings.append("All systems appear healthy within the analysis period.")

    for f in findings:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Connecting to AWS...")
    dynamodb = get_dynamodb()
    s3 = get_s3()

    sys_blend_stats = {}
    sys_user_stats = {}
    s3_checks = {}
    s3_inventories = {}

    for sys_num, cfg in SYSTEMS.items():
        print(f"\n--- System {sys_num} ---")

        # Blend analysis
        print(f"  Scanning {cfg['blend_table']}...")
        all_blends = full_scan(dynamodb, cfg["blend_table"])
        period_blends = filter_by_date(all_blends)
        print(f"  {len(all_blends)} total blends, {len(period_blends)} in date range")
        sys_blend_stats[sys_num] = analyze_blends(period_blends, sys_num)

        # User / mosaic analysis
        print(f"  Scanning {cfg['user_table']}...")
        all_users = full_scan(dynamodb, cfg["user_table"])
        period_users = filter_by_date(all_users)
        print(f"  {len(all_users)} total users, {len(period_users)} in date range")
        sys_user_stats[sys_num] = analyze_users(period_users, sys_num)

        # S3 spot-check
        print(f"  Running S3 spot-check on {cfg['bucket']}...")
        s3_checks[sys_num] = spot_check_s3_assets(s3, period_blends, cfg, sample_size=20)
        print(f"  Spot-check done: {s3_checks[sys_num]}")

        # S3 mosaic folder inventory (all-time)
        print(f"  Running S3 mosaic folder inventory on {cfg['bucket']}...")
        s3_inventories[sys_num] = inventory_mosaic_folders(s3, dynamodb, cfg)
        print(f"  Inventory done: {s3_inventories[sys_num]['s3_mosaic_folders']} S3 folders, "
              f"{s3_inventories[sys_num]['verified_match']} verified, "
              f"{s3_inventories[sys_num]['phantom_mosaics']} phantom")

    # S3 mosaic inventory for systems 3 & 4 (mosaic-only, no blend analysis)
    for sys_num, cfg in MOSAIC_ONLY_SYSTEMS.items():
        print(f"\n--- System {sys_num} (mosaic inventory only) ---")
        print(f"  Running S3 mosaic folder inventory on {cfg['bucket']}...")
        s3_inventories[sys_num] = inventory_mosaic_folders(s3, dynamodb, cfg)
        print(f"  Inventory done: {s3_inventories[sys_num]['s3_mosaic_folders']} S3 folders, "
              f"{s3_inventories[sys_num]['verified_match']} verified, "
              f"{s3_inventories[sys_num]['phantom_mosaics']} phantom")

    # Generate report
    print("\nGenerating report...")
    report = generate_report(sys_blend_stats, sys_user_stats, s3_checks, s3_inventories)

    output_path = PROJECT_ROOT / "output" / "diagnostic_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
