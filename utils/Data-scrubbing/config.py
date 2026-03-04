"""
Shared configuration for the data-scrubbing utilities.

Loads AWS credentials from the project .env and provides client factories
plus the bucket ↔ DynamoDB table mapping for all four blends.
"""

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BLENDS = {
    1: {"bucket": "pistoletto.moe", "table": "eternity-mirror-blends"},
    2: {"bucket": "pistoletto.moe2", "table": "eternity-mirror-blends-2"},
    3: {"bucket": "pistoletto.moe3", "table": "eternity-mirror-blends-3"},
    4: {"bucket": "pistoletto.moe4", "table": "eternity-mirror-blends-4"},
    5: {"bucket": "pistoletto.sanitized", "table": "eternity-mirror-blends-sanitized"},
}

IMAGE_PREFIX = "selected-images/"
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "tiff", "tif"}


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_dynamodb_resource():
    return boto3.resource(
        "dynamodb",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def get_blend_config(blend_number: int) -> dict:
    if blend_number not in BLENDS:
        raise ValueError(f"Invalid blend number {blend_number}. Must be 1–4.")
    return BLENDS[blend_number]
