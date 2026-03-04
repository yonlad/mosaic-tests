cd utils/dynamodb-cleaning

# Step 1 – Dry-run: preview orphan rows (no deletions)
python clean.py --blend 1 --dry-run

# Step 2 – Review the generated log file
#   clean_log_blend1_<timestamp>.json

# Step 3 – Execute for real (deletes orphan rows from DynamoDB)
python clean.py --blend 1

# Stricter mode: only delete rows where ALL asset keys are missing
python clean.py --blend 1 --mode all --dry-run

# Process all blends at once
python clean.py --blend all --dry-run
