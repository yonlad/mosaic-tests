# Blend Review Tool — Quick Start

## Setup (one time)

1. Unzip the folder — keep `Blend Review.app` and `.env` **side by side** in the same folder.
2. **First launch only:** right-click `Blend Review.app` -> choose **Open** -> click **Open** in the dialog.
   (This clears macOS Gatekeeper since the app is unsigned. After the first time it opens normally with a double-click.)

## Using the app

1. Double-click `Blend Review.app`.
2. Pick a blend from the dropdown (each one maps to a different S3 bucket / DynamoDB table).
3. Click **Review S3 Images** or **Review Blends** — the app will scan the database and open a gallery in your browser.
4. In the browser gallery:
   - Click images/cards to **flag** them (a red border appears).
   - Use the toolbar buttons to select all, invert, or bulk-flag small images.
   - Click **Export Deletion Manifest** when done — a JSON file downloads to your **Downloads** folder.
5. **Email** the downloaded JSON file to Yonatan. He will run the actual deletion.

## Troubleshooting

- **"Missing AWS Credentials" error:** Make sure the `.env` file is in the same folder as the app (not inside it).
- **App won't open at all:** Right-click -> Open (see step 2 above).
- **Gallery shows broken images:** The presigned URLs expire after 24 hours. Re-run the review to get fresh URLs.
