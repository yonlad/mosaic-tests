# Mockup Generation & Email Dispatch Pipeline

## Context

Participants in Michelangelo Pistoletto's "Mirror of Eternity" installation have mosaic portraits stored in S3 with white backgrounds. The goal is to:
1. Replace the white background with a mirror/reflective gradient texture
2. Email each participant their personalized mockup with an offer to acquire a print

The pipeline is split into two independent scripts for modularity and debuggability.

---

## File Structure

```
utils/mockup-pipeline/
  generate_mockups.py    # Phase 1: background swap + S3 upload
  send_emails.py         # Phase 2: Gmail SMTP email dispatch
  email_template.py      # HTML email body as a Python constant
```

Follows existing convention: each utility in its own subdirectory under `utils/`.

---

## Phase 1: `generate_mockups.py` — Mockup Generator

### Input/Output
- **Input**: Single CSV file (CLI arg) with columns: `email, user_id, mosaic_url, mosaic_thumbnail_url, source_image_url, created_at`
- **Output**: New CSV (`{input_stem}_with_mockups.csv`) with all original columns + `parsed_name` + `mockup_url`
- The `parsed_name` column lets you review and manually correct names before running Phase 2

### CLI
```
python generate_mockups.py path/to/participants.csv [--workers 8] [--threshold 240] [--dry-run]
```

### Flow
1. Load `mirror-background.png` once into memory
2. Read input CSV with `csv.DictReader`
3. For each row (via `ThreadPoolExecutor`):
   a. Download mosaic JPEG from `mosaic_url` using `requests`
   b. Remove white background using Pillow + numpy:
      - Convert to RGBA
      - Create mask where R, G, B are all > threshold (default 240)
      - Set alpha=0 for masked pixels
      - Optional: 1-2px Gaussian blur on alpha channel for edge feathering
   c. Resize mirror background to match mosaic dimensions
   d. `Image.alpha_composite(mirror_bg, mosaic_rgba)` — mosaic on top of mirror
   e. Convert to RGB, save as JPEG bytes
   f. Upload to S3: `mockups/{user_id}/mockup.jpg` in the mockup bucket
   g. Parse first name from email address (see Name Parsing below)
   h. Return row with `parsed_name` and `mockup_url` appended
4. Write output CSV — **user reviews `parsed_name` column and corrects any mistakes before Phase 2**

### Name Parsing (done in Phase 1)
- Extract local part before `@`, split on `.`, `_`, `-`
- Take first token, title-case it
- Default to `"there"` if empty, all digits, or single character
- Result stored in `parsed_name` column for review before emailing

### Key Patterns to Reuse
- `PROJECT_ROOT` / `.env` loading from `generate_mosaic_csvs.py:20-21`
- `get_s3_client()` pattern from `generate_mosaic_csvs.py:40-46`
- S3 URL construction: `https://s3.{region}.amazonaws.com/{bucket}/{key}`

### Error Handling
- Retry S3 uploads 3x with exponential backoff
- Skip rows with empty `mosaic_url` (log warning)
- On processing failure: log error, set `mockup_url` to empty string, continue
- Idempotent: check `head_object` before uploading; skip if exists

---

## Phase 2: `send_emails.py` — Email Dispatcher

### Input
- CSV from Phase 1 (with `parsed_name` and `mockup_url` columns)
- **Important**: Phase 2 uses the `parsed_name` column as-is — no re-parsing. Edit the CSV to fix any names before running.

### CLI
```
python send_emails.py path/to/participants_with_mockups.csv [--batch-size 480] [--dry-run]
```

### Gmail SMTP Setup
- Uses `smtplib.SMTP_SSL("smtp.gmail.com", 465)` with App Password
- Team member needs: 2FA enabled on Google account, then generate App Password
- Credentials from `.env`: `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`

### Batching & Resume (500/day Gmail limit)
- **Progress file**: `{csv_stem}_email_progress.json` alongside the CSV
  ```json
  {
    "sent": ["email1@example.com", ...],
    "failed": {"email3@example.com": "SMTP error: ..."},
    "last_run": "2026-06-10T14:30:00",
    "total_sent_this_run": 245
  }
  ```
- On each run: load progress, skip `sent` emails, stop at batch limit (default 480)
- Save progress every 10 emails (crash resilience)
- Sleep 1-2s between emails to avoid rate triggers
- Single SMTP connection reused for the batch

### Email Construction
- `MIMEMultipart("related")` with:
  - `MIMEText(html, "html")` — main body
  - `MIMEImage(mockup_bytes)` — inline attachment with `Content-ID` header
- Mockup referenced in HTML as `<img src="cid:mosaic-portrait">`
- Subject: `"Your Mirror of Eternity Mosaic Portrait"`

### Email Template (`email_template.py`)
- Customer-facing copy starts at "Dear {name}," through artwork details
- "here" links to: `https://thebass.org/art/michelangelo-pistoletto-mirror-of-eternity-2025/`
- `jacqueline@dminti.com` as `mailto:` link
- Pistoletto's quote styled as blockquote
- Mockup image at the very bottom, below signature and artwork details
- Inline CSS only (email client compatibility)

---

## .env Additions

```
# Mockup Pipeline
MOCKUP_S3_BUCKET=pistoletto.moe.mockups

# Gmail SMTP
GMAIL_ADDRESS=team-member@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

Reuses existing `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.

---

## Dependencies

**No new pip packages needed.** Everything is in stdlib or already in `requirements.txt`:
- `csv`, `smtplib`, `email.mime.*`, `json`, `argparse`, `logging` — stdlib
- `boto3`, `Pillow`, `requests`, `python-dotenv`, `numpy`, `tqdm` — already installed

---

## Critical Files

| File | Role |
|------|------|
| `utils/backgrounds/mirror-background.png` | Background texture for compositing |
| `utils/find-mosaics/generate_mosaic_csvs.py` | Reference patterns (PROJECT_ROOT, S3 client, CSV I/O) |
| `.env` | Needs MOCKUP_S3_BUCKET + Gmail credential additions |
| `utils/find-mosaics/output/system*_mosaics.csv` | Example input CSVs for testing |

---

## Verification

### Phase 1 Testing
1. Create a 2-3 row test CSV from existing system CSVs
2. Run `python generate_mockups.py test.csv --workers 1`
3. Verify output CSV has populated `mockup_url` column
4. Open a mockup URL in browser — confirm mosaic content intact, mirror background replaced white
5. Re-run same CSV — verify idempotent skip behavior

### Phase 2 Testing
1. Create a 1-2 row test CSV with your own email address
2. Run `python send_emails.py test_with_mockups.csv --batch-size 2`
3. Check inbox: correct name, working links, inline mockup image visible
4. Verify progress JSON was created
5. Re-run — confirm no duplicate sends

### White Background Threshold
- Default 240 should work for project-generated mosaics (known white backgrounds)
- If edges look rough: lower to 230. If mosaic content is clipped: raise to 245
- Use `--threshold` CLI arg to tune

---

## Agent Implementation Prompt

Use the following prompt to have an AI agent implement this plan:

```
You are implementing a two-phase mockup and email pipeline for the mosaic-tests project at /Users/yonatan/Desktop/mosaic-tests.

READ the full plan at utils/backgrounds/plan.md before writing any code.

Key references to read first:
- utils/find-mosaics/generate_mosaic_csvs.py (patterns for PROJECT_ROOT, .env, S3 client, CSV I/O)
- utils/backgrounds/mirror-background.png (the background texture)
- .env (existing AWS credentials)

Create these files under utils/mockup-pipeline/:
1. generate_mockups.py - Phase 1: CSV in → download mosaics → swap white BG for mirror → upload to S3 → CSV out with parsed_name + mockup_url
2. email_template.py - HTML email template constant
3. send_emails.py - Phase 2: CSV in → Gmail SMTP with batching/resume → send personalized emails with inline mockup

Follow the plan exactly for: CLI args, name parsing logic, batching/resume mechanism, error handling, and email template content.

No new pip packages — use only what's in requirements.txt + stdlib.

After implementation, test Phase 1 with a 2-3 row test CSV, then test Phase 2 with --dry-run.
```
