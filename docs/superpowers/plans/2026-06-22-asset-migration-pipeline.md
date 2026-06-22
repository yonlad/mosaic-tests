# Asset Migration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `migrate.py` in `utils/data-scrubbing/` with three subcommands (migrate, dedup, audit-system4) to copy verified good assets to the centralized sanitized bucket, detect duplicates, and audit system 4 coverage.

**Architecture:** Single Python script with argparse subcommands, reusing the shared `config.py` for AWS clients and bucket mappings. Pure helper functions handle filtering logic (UUID validation, date parsing, manifest loading). Each subcommand produces JSON logs consistent with the existing `delete.py` patterns.

**Tech Stack:** Python 3.10+, boto3 (via existing config.py), pytest for unit tests

## Global Constraints

- Reuse `config.py` for all AWS access — do not duplicate credentials or client factories
- Image extensions: `{"jpg", "jpeg", "png", "webp", "tiff", "tif"}` (from `config.IMAGE_EXTENSIONS`)
- S3 prefix: `selected-images/` (from `config.IMAGE_PREFIX`)
- Destination bucket: `pistoletto.sanitized` (BLENDS[5])
- Reviewed assets destination prefix: `selected-images/reviewed-images/system-{1,2,3}/<uuid>/<filename>`
- Manifest directory: `utils/data-scrubbing/reviewed_assets/`
- All scripts run from `utils/data-scrubbing/` directory
- JSON logs written to `utils/data-scrubbing/` directory

---

### Task 1: Shared Helper Functions + Unit Tests

**Files:**
- Create: `utils/data-scrubbing/migrate.py` (helpers only, no subcommands yet)
- Create: `utils/data-scrubbing/test_migrate.py`

**Interfaces:**
- Produces:
  - `is_participant_session(folder_name: str) -> bool`
  - `parse_capture_date(filename: str) -> date | None`
  - `extract_dedup_key(s3_key: str) -> str | None`
  - `load_flagged_keys(manifest_dir: Path) -> dict[str, set[str]]`
  - `list_s3_images(s3, bucket: str, prefix: str) -> list[str]`
  - Constants: `SANITIZED_BUCKET`, `REVIEWED_PREFIX`, `UUID_RE`, `CAPTURE_DATE_RE`, `DEFAULT_RANGES`, `MANIFEST_DIR`

- [ ] **Step 1: Write the failing tests**

Create `utils/data-scrubbing/test_migrate.py`:

```python
#!/usr/bin/env python3
"""Unit tests for migrate.py helper functions."""

import json
import tempfile
from datetime import date
from pathlib import Path

from migrate import (
    is_participant_session,
    parse_capture_date,
    extract_dedup_key,
    load_flagged_keys,
)


class TestIsParticipantSession:
    def test_valid_uuid(self):
        assert is_participant_session("31dc9a6b-04c0-4a95-8450-56db0cd90d34") is True

    def test_zero_uuid(self):
        assert is_participant_session("00000000-0000-0000-0000-000000000000") is True

    def test_resized_white_bg(self):
        assert is_participant_session("resized_white_bg") is False

    def test_resized_black_bg(self):
        assert is_participant_session("resized_black_bg") is False

    def test_flora_screenshot(self):
        assert is_participant_session("flora-screenshot") is False

    def test_empty_string(self):
        assert is_participant_session("") is False

    def test_uppercase_uuid_rejected(self):
        assert is_participant_session("31DC9A6B-04C0-4A95-8450-56DB0CD90D34") is False


class TestParseCaptureDate:
    def test_standard_filename(self):
        assert parse_capture_date("capture_20260607_071909.jpg") == date(2026, 6, 7)

    def test_new_year(self):
        assert parse_capture_date("capture_20260101_000000.jpg") == date(2026, 1, 1)

    def test_no_capture_prefix(self):
        assert parse_capture_date("random_file.jpg") is None

    def test_invalid_date_digits(self):
        assert parse_capture_date("capture_99991399_000000.jpg") is None

    def test_bg_removed_suffix(self):
        assert parse_capture_date("capture_20250628_133311_bg_removed_fit.jpg") == date(2025, 6, 28)


class TestExtractDedupKey:
    def test_standard_path(self):
        key = "selected-images/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_system_path(self):
        key = "selected-images/system-1/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_reviewed_path(self):
        key = "selected-images/reviewed-images/system-1/31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"
        assert extract_dedup_key(key) == "31dc9a6b-04c0-4a95-8450-56db0cd90d34/capture_20260606_071909.jpg"

    def test_no_uuid_returns_none(self):
        key = "selected-images/resized_white_bg/capture_20250628_133311_bg_removed_fit.jpg"
        assert extract_dedup_key(key) is None


class TestLoadFlaggedKeys:
    def test_loads_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "bucket": "pistoletto.moe",
                "items": [
                    {"s3_key": "selected-images/abc/capture_1.jpg", "blend_ids": []},
                    {"s3_key": "selected-images/def/capture_2.jpg", "blend_ids": ["b1"]},
                ],
            }
            Path(tmpdir, "manifest1.json").write_text(json.dumps(manifest))
            result = load_flagged_keys(Path(tmpdir))
            assert "pistoletto.moe" in result
            assert len(result["pistoletto.moe"]) == 2
            assert "selected-images/abc/capture_1.jpg" in result["pistoletto.moe"]

    def test_multiple_buckets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "a.jpg", "blend_ids": []}]}
            m2 = {"bucket": "pistoletto.moe2", "items": [{"s3_key": "b.jpg", "blend_ids": []}]}
            Path(tmpdir, "m1.json").write_text(json.dumps(m1))
            Path(tmpdir, "m2.json").write_text(json.dumps(m2))
            result = load_flagged_keys(Path(tmpdir))
            assert len(result) == 2
            assert "a.jpg" in result["pistoletto.moe"]
            assert "b.jpg" in result["pistoletto.moe2"]

    def test_same_bucket_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "a.jpg", "blend_ids": []}]}
            m2 = {"bucket": "pistoletto.moe", "items": [{"s3_key": "b.jpg", "blend_ids": []}]}
            Path(tmpdir, "m1.json").write_text(json.dumps(m1))
            Path(tmpdir, "m2.json").write_text(json.dumps(m2))
            result = load_flagged_keys(Path(tmpdir))
            assert len(result["pistoletto.moe"]) == 2

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_flagged_keys(Path(tmpdir)) == {}

    def test_nonexistent_dir(self):
        assert load_flagged_keys(Path("/nonexistent/path")) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python -m pytest test_migrate.py -v`

Expected: ImportError — `cannot import name 'is_participant_session' from 'migrate'` (module doesn't exist yet)

- [ ] **Step 3: Write the implementation**

Create `utils/data-scrubbing/migrate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python -m pytest test_migrate.py -v`

Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yonatan/Desktop/mosaic-tests
git add utils/data-scrubbing/migrate.py utils/data-scrubbing/test_migrate.py
git commit -m "feat: add shared helper functions for asset migration pipeline

Adds UUID validation, date parsing, dedup key extraction, manifest
loading, and S3 listing helpers. Includes unit tests for all pure
functions."
```

---

### Task 2: `migrate` Subcommand

**Files:**
- Modify: `utils/data-scrubbing/migrate.py`

**Interfaces:**
- Consumes: `is_participant_session`, `parse_capture_date`, `load_flagged_keys`, `list_s3_images`, `SANITIZED_BUCKET`, `REVIEWED_PREFIX`, `DEFAULT_RANGES`, `MANIFEST_DIR`, `BLENDS`, `get_s3_client`
- Produces: `run_migrate(args) -> None` — callable from CLI, writes JSON log to disk

- [ ] **Step 1: Add run_migrate function and CLI scaffold**

Append to `utils/data-scrubbing/migrate.py` (after the helper functions):

```python
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

    # dedup (placeholder — implemented in Task 3)
    sub.add_parser("dedup", help="Scan sanitized bucket for duplicate assets")

    # audit-system4 (placeholder — implemented in Task 4)
    sub.add_parser("audit-system4", help="Diagnostic report of system 4 coverage")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "migrate":
        run_migrate(args)
    elif args.command == "dedup":
        print("dedup: not yet implemented")
    elif args.command == "audit-system4":
        print("audit-system4: not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI scaffold works**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python migrate.py --help`

Expected: Help text showing the three subcommands

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python migrate.py migrate --help`

Expected: Help text showing `--dry-run`, `--system`, `--start-date`, `--end-date`

- [ ] **Step 3: Verify migrate dry-run against real S3**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python migrate.py migrate --system 1 --dry-run`

Expected: Output showing listing from `pistoletto.moe`, filtering results, and "Would copy N/N..." lines. JSON log written. Verify:
- Non-participant count > 0 (resized_white_bg etc. are filtered)
- Flagged count > 0 (manifest keys are excluded)
- No actual copies made (dry-run)

- [ ] **Step 4: Run existing tests still pass**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python -m pytest test_migrate.py -v`

Expected: All 16 tests still PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yonatan/Desktop/mosaic-tests
git add utils/data-scrubbing/migrate.py
git commit -m "feat: add migrate subcommand to copy verified assets to sanitized bucket

Lists source bucket assets, filters by UUID session, date range, and
deletion manifests, then copies good assets to reviewed-images/ in the
sanitized bucket. Supports --dry-run, --system, --start-date, --end-date."
```

---

### Task 3: `dedup` Subcommand

**Files:**
- Modify: `utils/data-scrubbing/migrate.py`

**Interfaces:**
- Consumes: `extract_dedup_key`, `list_s3_images`, `SANITIZED_BUCKET`, `get_s3_client`
- Produces: `run_dedup(args) -> None` — scans sanitized bucket, reports duplicates, optionally cleans

- [ ] **Step 1: Add run_dedup function**

Insert in `utils/data-scrubbing/migrate.py` after `run_migrate` and before `parse_args`:

```python
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
```

- [ ] **Step 2: Update CLI to wire dedup subcommand**

In `parse_args`, replace the dedup placeholder with:

```python
    # dedup
    ded = sub.add_parser("dedup", help="Scan sanitized bucket for duplicate assets")
    ded.add_argument("--clean", action="store_true", help="Delete duplicate copies (keeps reviewed-images)")
    ded.add_argument("--dry-run", action="store_true", help="Preview without deleting")
```

In `main`, replace the dedup placeholder with:

```python
    elif args.command == "dedup":
        run_dedup(args)
```

- [ ] **Step 3: Verify dedup report-only mode against real S3**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python migrate.py dedup`

Expected: Output showing total assets scanned, unique count, and any duplicate groups found. JSON report written. No deletions.

- [ ] **Step 4: Run all tests still pass**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python -m pytest test_migrate.py -v`

Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yonatan/Desktop/mosaic-tests
git add utils/data-scrubbing/migrate.py
git commit -m "feat: add dedup subcommand to detect and clean duplicate assets

Scans all selected-images/ in the sanitized bucket, groups by
uuid/filename dedup key, reports duplicates. Supports --clean to
remove extras (prefers reviewed-images copies) and --dry-run."
```

---

### Task 4: `audit-system4` Subcommand

**Files:**
- Modify: `utils/data-scrubbing/migrate.py`

**Interfaces:**
- Consumes: `is_participant_session`, `extract_dedup_key`, `list_s3_images`, `SANITIZED_BUCKET`, `BLENDS`, `get_s3_client`
- Produces: `run_audit_system4(args) -> None` — diagnostic report comparing system 4 vs sanitized

- [ ] **Step 1: Add run_audit_system4 function**

Insert in `utils/data-scrubbing/migrate.py` after `run_dedup` and before `parse_args`:

```python
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
```

- [ ] **Step 2: Update CLI to wire audit-system4**

In `parse_args`, the audit-system4 parser is already added (no args needed). In `main`, replace the placeholder with:

```python
    elif args.command == "audit-system4":
        run_audit_system4(args)
```

- [ ] **Step 3: Verify audit against real S3**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python migrate.py audit-system4`

Expected: Output showing system 4 participant asset count, how many are present/missing in sanitized bucket. JSON report written.

- [ ] **Step 4: Run all tests still pass**

Run: `cd /Users/yonatan/Desktop/mosaic-tests/utils/data-scrubbing && python -m pytest test_migrate.py -v`

Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/yonatan/Desktop/mosaic-tests
git add utils/data-scrubbing/migrate.py
git commit -m "feat: add audit-system4 subcommand for system 4 coverage diagnostic

Compares participant assets in pistoletto.moe4 against the sanitized
bucket using uuid/filename dedup keys. Reports present and missing
asset counts with full JSON report."
```
