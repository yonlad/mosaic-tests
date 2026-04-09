cd utils/Data-scrubbing

# Step 1 – Generate the review gallery
python review.py --blend 1

# Step 2 – Browser opens → click bad images → "Export Deletion Manifest"

# Step 3 – Preview deletions
python delete.py --manifest deletion_manifest_blend1_*.json --dry-run

# Step 4 – Execute for real
python delete.py --manifest deletion_manifest_blend5_*.json

When you're ready to move on to blends 2–4, just change --blend 2, --blend 3, etc. Want to kick off Blend 1 now?


# Review Blends
cd utils/data-scrubbing
python review_blends.py --blend 1          # generates review_blends1.html, opens in browser
python review_blends.py --blend 2 --no-open

# Delete Blends

# Step 1 — Preview what would be deleted
python delete.py --manifest blend_manifest_blend1_*.json --dry-run

# Step 2 — Execute for real
python delete.py --manifest blend_manifest_blend1_*.json
