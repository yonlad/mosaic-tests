import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv


def _is_frozen():
    return getattr(sys, "frozen", False)


def resolve_env_path():
    candidates = []
    if _is_frozen():
        # Next to the .app bundle
        app_dir = Path(sys.executable).resolve().parent.parent.parent
        candidates.append(app_dir / ".env")
        # ~/Library/Application Support/BlendReviewTool/.env
        candidates.append(
            Path.home() / "Library" / "Application Support" / "BlendReviewTool" / ".env"
        )
        # Bundled inside the app (PyInstaller data dir)
        candidates.append(Path(sys._MEIPASS) / ".env")
    else:
        candidates.append(Path(__file__).resolve().parent / ".env")
        candidates.append(Path(__file__).resolve().parent.parent.parent / ".env")

    for p in candidates:
        if p.is_file():
            return p
    return None


def app_data_dir():
    if _is_frozen():
        d = Path.home() / "Library" / "Application Support" / "BlendReviewTool" / "galleries"
    else:
        d = Path(__file__).resolve().parent / "galleries"
    d.mkdir(parents=True, exist_ok=True)
    return d


env_path = resolve_env_path()
if env_path:
    load_dotenv(env_path)

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BLENDS = {
    1: {"bucket": "pistoletto.moe", "table": "eternity-mirror-blends"},
    2: {"bucket": "pistoletto.moe2", "table": "eternity-mirror-blends-2"},
    3: {"bucket": "pistoletto.moe3", "table": "eternity-mirror-blends-3"},
    4: {"bucket": "pistoletto.moe4", "table": "eternity-mirror-blends-4"},
    5: {"bucket": "pistoletto.sanitized", "table": "eternity-mirror-blends-central"},
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
        raise ValueError(f"Invalid blend number {blend_number}. Must be 1-5.")
    return BLENDS[blend_number]
