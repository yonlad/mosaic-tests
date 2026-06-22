# Sanitized Bucket Migration — June 22, 2026

## Goal

Establish `pistoletto.sanitized` as the single source of truth for all production-quality assets, backed by the `eternity-mirror-blends-central` DynamoDB table that controls the blend screen slideshow. Ensure all asset references are consistent, deduplicated, and use stable public URLs.

## Background

The system operates across four local installations (systems 1-4), each with its own S3 bucket and DynamoDB table:

| System | S3 Bucket | DynamoDB Table |
|--------|-----------|----------------|
| 1 | `pistoletto.moe` | `eternity-mirror-blends` |
| 2 | `pistoletto.moe2` | `eternity-mirror-blends-2` |
| 3 | `pistoletto.moe3` | `eternity-mirror-blends-3` |
| 4 | `pistoletto.moe4` | `eternity-mirror-blends-4` |
| Central | `pistoletto.sanitized` | `eternity-mirror-blends-central` |

A reviewer had flagged bad assets across systems 1-3 between specific date ranges. The deletion manifests (in `utils/data-scrubbing/reviewed_assets/`) list these flagged assets. Everything the reviewer did NOT flag within those date ranges is considered verified good.

## What Was Done

### Step 1: Migrate Verified Good Assets

**Tool:** `python migrate.py migrate`

For each source system, listed all images in `selected-images/`, then filtered:
1. **Participant sessions only** — parent folder must be a valid UUID (excludes `resized_white_bg`, `resized_black_bg`, `flora-screenshot`, and any non-UUID folder)
2. **Within reviewed date range** — date extracted from `capture_YYYYMMDD_HHMMSS` filenames
3. **Not flagged** — excluded any key in the reviewer's deletion manifests
4. **Not already in destination** — idempotent for re-runs

Per-system date ranges reviewed:
- System 1: Feb 7 – Jun 22, 2026
- System 2: Feb 21 – Jun 22, 2026
- System 3: Jan 4 – Jun 22, 2026

**Result:** 979 verified good participant images copied to `pistoletto.sanitized/selected-images/reviewed-images/system-{1,2,3}/<uuid>/<filename>`.

### Step 2: Deduplication

**Tool:** `python migrate.py dedup --clean`

Scanned the entire sanitized bucket, grouped assets by dedup key (`<uuid>/<capture_filename>`), and found 168 duplicate groups:
- Assets existing in both `reviewed-images/system-N/` and `system-N/` (same asset, two copies)
- Assets duplicated across `system-1/` and `system-2/` (pre-existing cross-system duplicates)

**DynamoDB safety:** Before deleting any S3 object, the script scans `eternity-mirror-blends-central` for records referencing the key being deleted. Found 72 DynamoDB records that would break. Updated all 72 `random_image_key` and `random_image_url` references to point to the kept copy BEFORE deleting the duplicate S3 objects. If any DynamoDB update fails, S3 deletions are aborted.

**Dedup preference:** When a duplicate exists in `reviewed-images/` (verified) and `system-N/`, keep the `reviewed-images/` copy. Otherwise keep the first found.

**Result:** 72 DynamoDB records redirected, 168 duplicate S3 objects removed. Post-clean verification: 0 duplicates remaining.

### Step 3: Fix Broken References

Found 5 pre-existing broken `random_image_key` references in the central table — assets that were referenced but missing from the sanitized bucket. All 5 existed on the original system buckets. Copied them to the sanitized bucket.

Later found 8 more records with random images not on the sanitized bucket:
- 5 recoverable (asset exists on source bucket) — copied and updated
- 3 unrecoverable (asset deleted from all buckets) — left as-is

### Step 4: Source Image Migration

**Tool:** `python migrate.py migrate-sources`

The central DynamoDB table had `source_image_key` / `source_image_bucket` pointing to the original system buckets. Migrated all source images to the sanitized bucket:

**Phase 1 — S3 copy:** 910 source images copied from original buckets to `pistoletto.sanitized/selected-images/system-N/<uuid>/<filename>`. 337 were already present.

**Phase 2 — DynamoDB update:** All 1,247 records updated:
- `source_image_bucket` → `pistoletto.sanitized`
- `source_image_key` → bare key in sanitized bucket
- `source_image_url` → public URL

### Step 5: Public URL Conversion

Converted all stored URLs from presigned (expiring) to plain public format across ALL 5 DynamoDB tables:

**Public URL format:** `https://s3.us-east-2.amazonaws.com/<bucket>/<key>` (no signature, no expiry)

This matches the transition already made in the application code (`blend_processor.py` and `blend_routes.py`), where the `_sign()` function builds public URLs. The buckets have public-read ACLs, so public URLs work directly.

| Table | Records Updated |
|-------|----------------|
| `eternity-mirror-blends-central` | 1,247 |
| `eternity-mirror-blends` | 495 |
| `eternity-mirror-blends-2` | 239 |
| `eternity-mirror-blends-3` | 995 |
| `eternity-mirror-blends-4` | 90 |

### System 4 Audit

**Tool:** `python migrate.py audit-system4`

Diagnostic report comparing `pistoletto.moe4` against the sanitized bucket:
- 160 participant assets in system 4
- 113 present in sanitized bucket
- 47 missing (not yet reviewed/migrated)

## Architectural Decisions

### Destination Structure

```
pistoletto.sanitized/selected-images/
  system-1/<uuid>/capture_*.jpg          # existing + migrated source images
  system-2/<uuid>/capture_*.jpg
  system-3/<uuid>/capture_*.jpg
  system-4/<uuid>/capture_*.jpg
  reviewed-images/                        # verified good assets from review process
    system-1/<uuid>/capture_*.jpg
    system-2/<uuid>/capture_*.jpg
    system-3/<uuid>/capture_*.jpg
  resized_white_bg/capture_*_bg_removed_fit.jpg   # AI assets used by blends
  flora-screenshots/blend_*.*                      # AI assets used by blends
```

The `reviewed-images/` directory contains assets explicitly verified through the review process (not flagged by the reviewer). The `system-N/` layer inside preserves provenance for dedup comparison.

### Dedup Key

`<uuid>/<capture_filename>` — extracted by finding the UUID-format folder in the path and taking the filename. Works regardless of directory depth, so the same asset at `system-1/<uuid>/file.jpg` and `reviewed-images/system-1/<uuid>/file.jpg` produces the same dedup key.

### DynamoDB Safety During Dedup

Never delete an S3 object without first checking and updating all DynamoDB references. The dedup clean:
1. Builds redirect map (deleted_key → kept_key)
2. Scans DynamoDB for references to any key being deleted
3. Updates all references to point to the kept copy
4. Only then deletes the S3 objects
5. If any DynamoDB update fails, aborts all S3 deletions

### No Deletions on Source Buckets

All operations are copy-only from the source buckets. Nothing is deleted from `pistoletto.moe`, `moe2`, `moe3`, or `moe4`. This ensures full rollback capability — if anything breaks, the DynamoDB CSV backup can be restored and all original assets are still in place.

### Video Assets Stay on System Buckets

Blend videos (`video-blends/blend_output_*.mp4`) remain on their original system buckets. Only images (source and random) were migrated to the sanitized bucket. Videos are not part of the normalization pipeline.

### URL Strategy

All stored URLs converted to plain public format: `https://s3.<region>.amazonaws.com/<bucket>/<key>`. This matches the application code which already builds public URLs in `_sign()` and `blend_processor.py`. Benefits:
- URLs never expire (unlike presigned URLs)
- Enables browser caching across kiosk restarts
- Buckets already have public-read ACLs

## Tools Built

**`utils/data-scrubbing/migrate.py`** — single script with subcommands:

| Command | Purpose |
|---------|---------|
| `migrate` | Copy verified good assets from systems 1-3 to sanitized bucket |
| `migrate --system N --start-date X --end-date Y` | Process a single system with custom date range |
| `dedup` | Report duplicates in sanitized bucket |
| `dedup --clean` | Remove duplicates (with DynamoDB redirect safety) |
| `migrate-sources` | Copy source images from central table to sanitized + convert URLs |
| `audit-system4` | Diagnostic report of system 4 coverage |

All commands support `--dry-run`. All produce JSON logs.

## Recurring Workflow

For future asset reviews:
1. Reviewer flags bad assets for a date range → exports deletion manifests
2. Place manifests in `utils/data-scrubbing/reviewed_assets/`
3. `python migrate.py migrate --system N --start-date YYYY-MM-DD --end-date YYYY-MM-DD --dry-run`
4. Review the log, then run without `--dry-run`
5. `python migrate.py dedup` to check for duplicates
6. `python migrate.py dedup --clean --dry-run` then `--clean` if needed

## Final State

- **Sanitized bucket:** ~3,800 assets, 0 duplicates, single source of truth
- **Central DynamoDB table:** 1,247 records, all source images on sanitized bucket, all URLs public
- **System tables (1-4):** 1,819 records, all URLs public
- **3 unrecoverable records:** Random images deleted from all buckets, cannot be restored
- **47 system 4 assets:** Not yet in sanitized bucket (pending review)
