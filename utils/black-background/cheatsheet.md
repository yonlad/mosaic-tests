# Default 30% sample
python split.py

# Custom percentage with reproducible seed
python split.py --sample-pct 10 --seed 42





Recommended workflow:
# Step 1: Generate the manifest (picks 30% of eligible images)
python split.py

# Step 2: Test on a single image
python process.py --manifest manifest_20260304_151743.json --limit 1

# Step 3: Verify the result in S3, then scale up gradually
python process.py --manifest manifest_YYYYMMDD_HHMMSS.json --limit 10 --workers 4

# Step 4: Process everything
python process.py --manifest manifest_YYYYMMDD_HHMMSS.json --workers 6