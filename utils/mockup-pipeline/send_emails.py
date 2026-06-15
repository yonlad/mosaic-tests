"""
Phase 2: Send personalized emails with inline mockup images via Gmail SMTP.

Reads the CSV output from Phase 1 (with parsed_name and mockup_url columns),
downloads each mockup, and sends an email with the mockup as an inline image.

Includes batching (default 480 to stay under Gmail's 500/day limit) and
crash-resilient resume via a progress JSON file.
"""

import argparse
import csv
import json
import logging
import os
import random
import smtplib
import sys
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from email_template import SUBJECT, get_email_html

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_progress(progress_path: Path) -> dict:
    """Load or initialize the progress file."""
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            return json.load(f)
    return {"sent": [], "failed": {}, "last_run": None, "total_sent_this_run": 0}


def save_progress(progress_path: Path, progress: dict):
    """Save progress to disk."""
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def build_email(to_addr: str, name: str, mockup_bytes: bytes) -> MIMEMultipart:
    """Construct a MIME email with inline mockup image."""
    msg = MIMEMultipart("related")
    msg["Subject"] = SUBJECT
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_addr

    html = get_email_html(name)
    msg.attach(MIMEText(html, "html"))

    img = MIMEImage(mockup_bytes, _subtype="jpeg")
    img.add_header("Content-ID", "<mosaic-portrait>")
    img.add_header("Content-Disposition", "inline", filename="mosaic-portrait.jpg")
    msg.attach(img)

    return msg


def download_mockup(mockup_url: str) -> bytes:
    """Download mockup image bytes from URL or local path."""
    if mockup_url.startswith(("http://", "https://")):
        resp = requests.get(mockup_url, timeout=30)
        resp.raise_for_status()
        return resp.content
    else:
        return Path(mockup_url).read_bytes()


def main():
    parser = argparse.ArgumentParser(description="Send mockup emails via Gmail SMTP")
    parser.add_argument("csv_path", help="Path to CSV with mockup_url and parsed_name columns")
    parser.add_argument("--batch-size", type=int, default=480, help="Max emails per run (default: 480)")
    parser.add_argument("--dry-run", action="store_true", help="Build emails but don't send them")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        log.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    if not args.dry_run and (not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD):
        log.error("GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in .env")
        sys.exit(1)

    # Load CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    log.info(f"Loaded {len(rows)} rows from {csv_path}")

    # Load progress
    progress_path = csv_path.parent / f"{csv_path.stem}_email_progress.json"
    progress = load_progress(progress_path)
    sent_set = set(progress["sent"])
    progress["total_sent_this_run"] = 0
    progress["last_run"] = datetime.now().isoformat()

    # Filter to rows that need sending
    to_send = []
    for row in rows:
        email = row.get("email", "").strip()
        mockup_url = row.get("mockup_url", "").strip()
        if not email or not mockup_url:
            continue
        if email in sent_set:
            continue
        to_send.append(row)

    log.info(f"{len(to_send)} emails to send ({len(sent_set)} already sent)")

    if not to_send:
        log.info("Nothing to send.")
        return

    # Cap at batch size
    batch = to_send[:args.batch_size]
    log.info(f"Processing batch of {len(batch)} emails")

    smtp = None
    if not args.dry_run:
        log.info("Connecting to Gmail SMTP...")
        smtp = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)

    try:
        for i, row in enumerate(batch):
            email = row["email"].strip()
            name = row.get("parsed_name", "there").strip()
            mockup_url = row["mockup_url"].strip()

            try:
                mockup_bytes = download_mockup(mockup_url)
                msg = build_email(email, name, mockup_bytes)

                if args.dry_run:
                    log.info(f"[DRY RUN] Would send to {email} (name={name}, mockup={len(mockup_bytes)} bytes)")
                else:
                    smtp.sendmail(GMAIL_ADDRESS, email, msg.as_string())
                    log.info(f"Sent to {email}")

                progress["sent"].append(email)
                progress["total_sent_this_run"] += 1

            except Exception as e:
                log.error(f"Failed to send to {email}: {e}")
                progress["failed"][email] = str(e)

            # Save progress every 10 emails
            if (i + 1) % 10 == 0:
                save_progress(progress_path, progress)

            # Rate limiting (skip in dry run)
            if not args.dry_run and i < len(batch) - 1:
                time.sleep(1 + random.random())

    finally:
        if smtp:
            smtp.quit()
        save_progress(progress_path, progress)

    log.info(f"Done: {progress['total_sent_this_run']} sent this run, {len(progress['failed'])} failed total")


if __name__ == "__main__":
    main()
