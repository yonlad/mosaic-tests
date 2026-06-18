# Mirror of Eternity — Diagnostic Report
**Period:** 2026-04-15 to 2026-06-15
**Generated:** 2026-06-16 15:38:14

## Executive Summary

| Metric | Value |
|--------|-------|
| Total blend attempts (both systems) | 434 |
| Successful blends | 430 |
| Failed blends | 4 |
| **Overall blend success rate** | **99.1%** |
| Total users registered (both systems) | 424 |
| Users with mosaic produced | 424 |
| Users without mosaic | 0 |
| **Overall mosaic production rate** | **100.0%** |

## System 1 — Blend Analysis

| Metric | Value |
|--------|-------|
| Total blend attempts | 290 |
| Completed | 288 (99.3%) |
| Failed | 2 (0.7%) |
| Completed items missing blend_url | 0 |
| Completed items missing s3_key | 0 |

### Status Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| completed | 288 | 99.3% |
| error | 2 | 0.7% |

### Failure Analysis

**Error type breakdown:**

| Error Type | Count |
|-----------|-------|
| API Timeout | 1 |
| Background Removal Failure | 1 |

**Failed blend details:**

| Blend ID | Status | Created | Error |
|----------|--------|---------|-------|
| `c4a929ba-699...` | error | 2026-05-26T07:24 | Replicate API failed: The read operation timed out |
| `7e54bdb3-6c5...` | error | 2026-05-16T02:51 | Image background removal failed: Background removal API retu |

**Failure time-of-day (UTC):**

- 02:00 UTC — 1 failure(s)
- 07:00 UTC — 1 failure(s)

### Weekly Breakdown

| Week | Total | Completed | Failed | Success Rate |
|------|-------|-----------|--------|-------------|
| 2026-W16 | 32 | 32 | 0 | 100.0% |
| 2026-W17 | 31 | 31 | 0 | 100.0% |
| 2026-W18 | 46 | 46 | 0 | 100.0% |
| 2026-W19 | 35 | 35 | 0 | 100.0% |
| 2026-W20 | 31 | 30 | 1 | 96.8% |
| 2026-W21 | 45 | 45 | 0 | 100.0% |
| 2026-W22 | 24 | 23 | 1 | 95.8% |
| 2026-W23 | 32 | 32 | 0 | 100.0% |
| 2026-W24 | 14 | 14 | 0 | 100.0% |

### S3 Asset Spot-Check (20 completed blends sampled)

- Missing source images in bucket: **0**
- Missing blend videos in bucket: **0**

## System 2 — Blend Analysis

| Metric | Value |
|--------|-------|
| Total blend attempts | 144 |
| Completed | 142 (98.6%) |
| Failed | 2 (1.4%) |
| Completed items missing blend_url | 0 |
| Completed items missing s3_key | 0 |

### Status Distribution

| Status | Count | Percentage |
|--------|-------|------------|
| completed | 142 | 98.6% |
| timeout | 2 | 1.4% |

### Failure Analysis

**Error type breakdown:**

| Error Type | Count |
|-----------|-------|
| Polling Timeout | 2 |

**Failed blend details:**

| Blend ID | Status | Created | Error |
|----------|--------|---------|-------|
| `d48f0156-442...` | timeout | 2026-05-01T17:23 | Max polling attempts reached. |
| `2ff1963f-221...` | timeout | 2026-05-01T11:40 | Max polling attempts reached. |

**Failure time-of-day (UTC):**

- 11:00 UTC — 1 failure(s)
- 17:00 UTC — 1 failure(s)

### Weekly Breakdown

| Week | Total | Completed | Failed | Success Rate |
|------|-------|-----------|--------|-------------|
| 2026-W16 | 6 | 6 | 0 | 100.0% |
| 2026-W17 | 4 | 4 | 0 | 100.0% |
| 2026-W18 | 4 | 2 | 2 | 50.0% |
| 2026-W19 | 5 | 5 | 0 | 100.0% |
| 2026-W20 | 102 | 102 | 0 | 100.0% |
| 2026-W21 | 6 | 6 | 0 | 100.0% |
| 2026-W22 | 4 | 4 | 0 | 100.0% |
| 2026-W23 | 9 | 9 | 0 | 100.0% |
| 2026-W24 | 1 | 1 | 0 | 100.0% |
| 2026-W25 | 3 | 3 | 0 | 100.0% |

### S3 Asset Spot-Check (20 completed blends sampled)

- Missing source images in bucket: **0**
- Missing blend videos in bucket: **0**

## System 1 — Mosaic Production Analysis

| Metric | Value |
|--------|-------|
| Total users registered | 294 |
| Users with mosaic produced | 294 (100.0%) |
| Users without mosaic | 0 (0.0%) |
| Users with photo selected but no mosaic | 0 |
| Users with no photo and no mosaic | 0 |

## System 2 — Mosaic Production Analysis

| Metric | Value |
|--------|-------|
| Total users registered | 130 |
| Users with mosaic produced | 130 (100.0%) |
| Users without mosaic | 0 (0.0%) |
| Users with photo selected but no mosaic | 0 |
| Users with no photo and no mosaic | 0 |

## S3 Mosaic Folder Inventory (All-Time)

This section counts the actual mosaic folders in each system's S3 bucket and cross-references
with the DynamoDB user tables to verify data consistency.

### System 1 — `pistoletto.moe`

| Metric | Value |
|--------|-------|
| S3 mosaic folders (unique users) | 1,041 |
| S3 total mosaic objects | 3,798 |
| Avg files per user folder | 3.6 |
| DynamoDB total users | 1,065 |
| DynamoDB users with `mosaic_key` | 1,062 |
| DynamoDB users without `mosaic_key` | 3 |
| **DB `mosaic_key` exists AND S3 folder exists** | **1,041** |
| DB `mosaic_key` exists but S3 folder missing | 21 |
| S3 folder exists but not in DB | 0 |

**21 phantom mosaic(s)** — these users have a `mosaic_key` recorded in DynamoDB but the actual
files are missing from S3.

### System 2 — `pistoletto.moe2`

| Metric | Value |
|--------|-------|
| S3 mosaic folders (unique users) | 224 |
| S3 total mosaic objects | 855 |
| Avg files per user folder | 3.8 |
| DynamoDB total users | 225 |
| DynamoDB users with `mosaic_key` | 225 |
| DynamoDB users without `mosaic_key` | 0 |
| **DB `mosaic_key` exists AND S3 folder exists** | **224** |
| DB `mosaic_key` exists but S3 folder missing | 1 |
| S3 folder exists but not in DB | 0 |

**1 phantom mosaic(s)** — these users have a `mosaic_key` recorded in DynamoDB but the actual
files are missing from S3.

### System 3 — `pistoletto.moe3`

| Metric | Value |
|--------|-------|
| S3 mosaic folders (unique users) | 1,128 |
| S3 total mosaic objects | 4,302 |
| Avg files per user folder | 3.8 |
| DynamoDB total users | 1,133 |
| DynamoDB users with `mosaic_key` | 1,128 |
| DynamoDB users without `mosaic_key` | 5 |
| **DB `mosaic_key` exists AND S3 folder exists** | **1,127** |
| DB `mosaic_key` exists but S3 folder missing | 1 |
| S3 folder exists but not in DB | 0 |

**1 phantom mosaic(s)** — these users have a `mosaic_key` recorded in DynamoDB but the actual
files are missing from S3.

### System 4 — `pistoletto.moe4`

| Metric | Value |
|--------|-------|
| S3 mosaic folders (unique users) | 125 |
| S3 total mosaic objects | 569 |
| Avg files per user folder | 4.6 |
| DynamoDB total users | 125 |
| DynamoDB users with `mosaic_key` | 125 |
| DynamoDB users without `mosaic_key` | 0 |
| **DB `mosaic_key` exists AND S3 folder exists** | **125** |
| DB `mosaic_key` exists but S3 folder missing | 0 |
| S3 folder exists but not in DB | 0 |

### Combined All-Time Totals

| Metric | System 1 | System 2 | System 3 | System 4 | Total |
|--------|----------||----------||----------||----------|-------|
| S3 mosaic folders | 1,041 | 224 | 1,128 | 125 | **2,518** |
| DynamoDB users | 1,065 | 225 | 1,133 | 125 | **2,548** |
| DB users with mosaic_key | 1,062 | 225 | 1,128 | 125 | **2,540** |
| Verified mosaics (DB + S3) | 1,041 | 224 | 1,127 | 125 | **2,517** |
| Phantom mosaics (DB yes, S3 no) | 21 | 1 | 1 | 0 | **23** |
| Orphan mosaics (S3 yes, DB no) | 0 | 0 | 0 | 0 | **0** |

## Cross-System Comparison

| Metric | System 1 | System 2 |
|--------|----------|----------|
| Total blends | 290 | 144 |
| Completed | 288 | 142 |
| Failed | 2 | 2 |
| Success rate | 99.3% | 98.6% |
| Registered users (in period) | 294 | 130 |
| Mosaics produced (in period) | 294 | 130 |
| Mosaic rate (in period) | 100.0% | 100.0% |
| S3 mosaic folders (all-time) | 1,041 | 224 |
| DynamoDB users (all-time) | 1,065 | 225 |
| Verified mosaics (all-time) | 1,041 | 224 |

## Key Findings & Recommendations

- **Top failure cause:** Polling Timeout (2 occurrence(s)). This accounts for 50% of all failures.
- Only 4 blend failure(s) out of 434 total — the system is highly reliable (99.1% success rate).
- **23 phantom mosaic(s) detected (all-time):** 21 in System 1, 1 in System 2, 1 in System 3 have a `mosaic_key` in DynamoDB but the corresponding S3 files no longer exist. These DB records should be audited.
- **All-time mosaic production:** 2,517 verified mosaics across all 4 systems, confirmed present in both DynamoDB and S3.
- **Zero orphan S3 mosaics:** Every mosaic folder in S3 corresponds to a known user in DynamoDB — no wasted storage.
