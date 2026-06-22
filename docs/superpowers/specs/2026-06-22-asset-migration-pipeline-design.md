# Asset Migration Pipeline Design

## Overview

A reusable pipeline (`migrate.py`) in `utils/data-scrubbing/` that migrates verified good assets from source S3 buckets (systems 1-3) to the centralized `pistoletto.sanitized` bucket, detects duplicates, and audits system 4 coverage.

## Context

The data-scrubbing workflow uses `review.py` to generate visual review galleries and exports deletion manifests listing flagged bad assets. This pipeline does the inverse: it takes everything the reviewer did NOT flag within a date range and copies those verified-good participant assets to the centralized bucket.

## Architecture

Single script with three subcommands, reusing the shared `config.py` for AWS clients and bucket/table mappings.

### Bucket Mapping

| System | Bucket | Role |
|--------|--------|------|
| 1 | `pistoletto.moe` | Source |
| 2 | `pistoletto.moe2` | Source |
| 3 | `pistoletto.moe3` | Source |
| 4 | `pistoletto.moe4` | Audit target |
| 5 | `pistoletto.sanitized` | Destination (centralized) |

### Destination Structure

```
pistoletto.sanitized/
  selected-images/
    system-1/...          # existing assets
    system-2/...
    system-3/...
    reviewed-images/      # new — verified good assets
      system-1/<uuid>/capture_*.jpg
      system-2/<uuid>/capture_*.jpg
      system-3/<uuid>/capture_*.jpg
```

The `system-*` layer inside `reviewed-images/` preserves provenance for dedup comparison without affecting downstream consumption (which can glob across all of `reviewed-images/`).

## Subcommand 1: `migrate`

### Purpose

Copy verified good assets from systems 1-3 to the sanitized bucket.

### CLI Interface

```
python migrate.py migrate [--dry-run]
python migrate.py migrate --system 1 --start-date 2026-02-07 --end-date 2026-06-22 --dry-run
```

### Default Date Ranges

| System | Start | End |
|--------|-------|-----|
| 1 | 2026-02-07 | 2026-06-22 |
| 2 | 2026-02-21 | 2026-06-22 |
| 3 | 2026-01-04 | 2026-06-22 |

When `--system` is not specified, all three systems are processed with their default ranges. When `--start-date`/`--end-date` are provided, they apply to the specified `--system`.

### Processing Flow

1. **Load deletion manifests**: Read all JSON files in `reviewed_assets/` directory. Group flagged `s3_key` values by bucket.
2. **List source assets**: For each source bucket, list all objects under `selected-images/`.
3. **Filter**:
   - **Participant only**: Parent folder under `selected-images/` must be a valid UUID. Skip known non-participant paths: `resized_white_bg`, `resized_black_bg`, `flora-screenshot`, and any other non-UUID folder.
   - **Date range**: Parse date from filename (`capture_YYYYMMDD_HHMMSS.jpg`). Keep only assets within the system's reviewed date range (inclusive).
   - **Not flagged**: Exclude any `s3_key` present in the deletion manifests for that bucket.
4. **Dedup against destination**: Check if `reviewed-images/system-N/<uuid>/<filename>` already exists in the sanitized bucket. Skip if already present (idempotent re-runs).
5. **Copy**: Use S3 copy_object to copy each asset from source to `pistoletto.sanitized/selected-images/reviewed-images/system-N/<uuid>/<filename>`.
6. **Log**: Write a JSON log with timestamp, counts (total listed, filtered, skipped as duplicate, copied, failed), and per-item details.

### Filtering Details

UUID validation regex: `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`

Date extraction regex from filename: `capture_(\d{8})_\d{6}` → parse `YYYYMMDD`.

## Subcommand 2: `dedup`

### Purpose

Scan the entire sanitized bucket for duplicate assets across all directories under `selected-images/`.

### CLI Interface

```
python migrate.py dedup [--dry-run]
python migrate.py dedup --clean [--dry-run]
```

### Processing Flow

1. **List all objects** under `selected-images/` in `pistoletto.sanitized`.
2. **Extract dedup key**: For each object, extract `<uuid>/<capture_filename>` from the full S3 key.
3. **Group by dedup key**: Any key with multiple full paths is a duplicate.
4. **Report** (JSON + printed summary):
   - Total assets scanned
   - Number of unique assets
   - Number of duplicated dedup keys
   - Total duplicate copies (extra copies beyond the first)
   - For each duplicate group: all full paths where the asset exists
5. **Clean** (when `--clean` is passed): For each duplicate group, keep the copy in `reviewed-images/` (verified) and delete the others. If no `reviewed-images/` copy exists, keep the first found. Supports `--dry-run`.

### Dedup Key

The dedup key is `<uuid>/<capture_filename>` — extracted by finding the UUID-format folder and the filename from the full S3 key. This works regardless of where in the directory tree the asset lives.

## Subcommand 3: `audit-system4`

### Purpose

Diagnostic report showing how many system 4 assets exist in the sanitized bucket.

### CLI Interface

```
python migrate.py audit-system4
```

### Processing Flow

1. **List source assets**: List all objects under `selected-images/` in `pistoletto.moe4`. Apply same UUID participant filter.
2. **List sanitized assets**: List all objects under `selected-images/` in `pistoletto.sanitized`. Build a set of dedup keys (`<uuid>/<capture_filename>`).
3. **Compare**: For each system 4 asset, check if its dedup key exists in the sanitized set.
4. **Report** (JSON + printed summary):
   - Total system 4 participant assets
   - How many are present in the sanitized bucket (and in which directory)
   - How many are missing
   - List of missing asset keys

## Shared Infrastructure

- **`config.py`**: AWS clients, bucket/table mappings (already exists)
- **S3 listing helper**: Paginated `list_objects_v2` with image extension filter (similar to `review.py`)
- **UUID validation**: Shared regex function
- **Date parsing**: Shared function to extract date from capture filename
- **JSON logging**: Consistent with `delete.py` log format (timestamps, counts, per-item details)

## Error Handling

- S3 copy failures are logged per-item and don't halt the batch. Summary reports failures.
- Missing or malformed deletion manifests produce a clear error and abort.
- Filenames that don't match the expected `capture_YYYYMMDD_HHMMSS` pattern are skipped with a warning.

## Future Use

For recurring runs:
1. Reviewer flags bad assets for a new date range → exports new deletion manifests
2. Place manifests in `reviewed_assets/`
3. Run `python migrate.py migrate --system N --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
4. Run `python migrate.py dedup` to check for new duplicates
