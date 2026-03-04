#!/usr/bin/env python3
"""
DynamoDB CSV Importer

This script reads a CSV file and inserts the data into a DynamoDB table.
It replaces the source bucket name (pistoletto.moe4) with a configurable target bucket name.

Usage:
    python dynamodb-insert.py --table TABLE_NAME --bucket BUCKET_NAME [--csv CSV_PATH] [--region REGION] [--dry-run]

Example:
    python dynamodb-insert.py --table my-blends-table --bucket my-new-bucket
    python dynamodb-insert.py --table my-blends-table --bucket my-new-bucket --csv ./table.csv --region us-east-2 --dry-run
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Source bucket name to be replaced
SOURCE_BUCKET = "pistoletto.moe4"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Import CSV data into DynamoDB with configurable bucket name replacement"
    )
    parser.add_argument(
        "--table",
        required=True,
        help="DynamoDB table name to insert data into"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Target S3 bucket name to replace 'pistoletto.moe4' with"
    )
    parser.add_argument(
        "--csv",
        default=Path(__file__).parent / "table.csv",
        type=Path,
        help="Path to CSV file (default: ./table.csv)"
    )
    parser.add_argument(
        "--region",
        default="us-east-2",
        help="AWS region (default: us-east-2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without inserting into DynamoDB"
    )
    
    return parser.parse_args()


def replace_bucket_name(value: str, source_bucket: str, target_bucket: str) -> str:
    """Replace source bucket name with target bucket name in a string value."""
    if isinstance(value, str) and source_bucket in value:
        return value.replace(source_bucket, target_bucket)
    return value


def process_row(row: dict, source_bucket: str, target_bucket: str) -> dict:
    """Process a CSV row, replacing bucket names in all fields and cleaning up empty values."""
    processed = {}
    for key, value in row.items():
        # Skip empty values - DynamoDB doesn't allow empty strings
        if value == "" or value is None:
            continue
        processed[key] = replace_bucket_name(value, source_bucket, target_bucket)
    return processed


def main():
    args = parse_args()
    
    # Validate CSV file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Check for AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not aws_access_key or not aws_secret_key:
        print("Warning: AWS credentials not found in .env file")
        print("Falling back to default AWS credential chain (environment, ~/.aws/credentials, IAM role)")
    
    print(f"Configuration:")
    print(f"  CSV file: {csv_path}")
    print(f"  DynamoDB table: {args.table}")
    print(f"  Target bucket: {args.bucket}")
    print(f"  AWS region: {args.region}")
    print(f"  Dry run: {args.dry_run}")
    print()
    
    # Read and process CSV
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Replace bucket name in all fields and clean empty values
            processed_row = process_row(row, SOURCE_BUCKET, args.bucket)
            items.append(processed_row)
    
    print(f"Processed {len(items)} rows from CSV")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Preview of first 3 items (with bucket replacement):\n")
        
        for i, item in enumerate(items[:3]):
            print(f"Item {i + 1}:")
            for key, value in list(item.items())[:5]:  # Show first 5 fields
                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else value
                print(f"  {key}: {display_value}")
            print("  ...")
            print()
        
        print(f"Would insert {len(items)} items into table '{args.table}'")
        print("\nRun without --dry-run to perform the actual insert.")
        return
    
    # Initialize DynamoDB resource (higher-level API)
    print(f"\nConnecting to DynamoDB in {args.region}...")
    dynamodb = boto3.resource("dynamodb", region_name=args.region)
    table = dynamodb.Table(args.table)
    
    # Verify table exists
    try:
        table.load()
        print(f"Table '{args.table}' found")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"Error: Table '{args.table}' does not exist")
            sys.exit(1)
        raise
    
    # Insert items using batch_writer (handles batching automatically)
    print(f"\nInserting {len(items)} items into '{args.table}'...")
    
    success_count = 0
    failure_count = 0
    
    try:
        with table.batch_writer() as batch:
            for i, item in enumerate(items):
                try:
                    batch.put_item(Item=item)
                    success_count += 1
                    
                    # Progress update every 10 items
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{len(items)} items...")
                        
                except Exception as e:
                    print(f"  Error inserting item {i + 1}: {e}")
                    failure_count += 1
                    
    except ClientError as e:
        print(f"Error during batch write: {e.response['Error']['Message']}")
        sys.exit(1)
    
    print(f"\nComplete!")
    print(f"  Successfully inserted: {success_count}")
    print(f"  Failed: {failure_count}")
    
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
