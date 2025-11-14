import os
import argparse
from io import BytesIO
from typing import Iterator, Optional

from PIL import Image

try:
    # Optional: support HEIC/HEIF if pillow-heif is installed
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
except Exception:
    pass

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

import boto3  # type: ignore
from botocore.exceptions import ClientError  # type: ignore


def _aws_config():
    region = os.getenv("REACT_APP_AWS_REGION", os.getenv("AWS_REGION", "us-east-2"))
    access_key = os.getenv("REACT_APP_AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
    secret_key = os.getenv(
        "REACT_APP_AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    return region, access_key, secret_key


def _s3_client():
    region, access_key, secret_key = _aws_config()
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def list_keys(bucket: str, prefix: str) -> Iterator[str]:
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj.get("Key")
            if not key:
                continue
            # skip "directories"
            if key.endswith("/"):
                continue
            yield key


def s3_key_exists(bucket: str, key: str) -> bool:
    s3 = _s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey") or code == "404":
            return False
        return False


def _dest_key_for(src_key: str, src_prefix: str, dst_prefix: str) -> str:
    # Preserve the leaf filename but convert extension to .jpg
    basename = os.path.basename(src_key)
    stem, _ = os.path.splitext(basename)
    return f"{dst_prefix.rstrip('/')}/{stem}.jpg"


def _open_image_from_s3(bucket: str, key: str) -> Optional[Image.Image]:
    s3 = _s3_client()
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        img = Image.open(BytesIO(body))
        return img
    except Exception:
        return None


def _composite_on_black(img: Image.Image) -> Image.Image:
    # Handle transparency; composite over black, return RGB
    if img.mode in ("RGBA", "LA") or ("A" in img.getbands()):
        rgba = img.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        composited = Image.alpha_composite(black, rgba).convert("RGB")
        return composited
    # No alpha; ensure RGB
    return img.convert("RGB")


def _save_jpeg_to_bytes(img: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    try:
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    except Exception:
        # Fallback without optimize/progressive if Pillow fails to optimize
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _upload_bytes_to_s3(bucket: str, key: str, data: bytes) -> None:
    s3 = _s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/jpeg")


def process_stream(
    bucket: str,
    src_prefix: str,
    dst_prefix: str,
    *,
    overwrite: bool = False,
    quality: int = 82,
    max_items: Optional[int] = None,
) -> None:
    processed = 0
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

    print(
        f"Starting processing: bucket={bucket}, src_prefix={src_prefix}, dst_prefix={dst_prefix}, "
        f"quality={quality}, overwrite={overwrite}"
    )

    for key in list_keys(bucket, src_prefix):
        if max_items is not None and processed >= max_items:
            break

        ext = os.path.splitext(key)[1].lower()
        if ext and ext not in allowed_ext:
            continue

        dst_key = _dest_key_for(key, src_prefix, dst_prefix)

        if not overwrite and s3_key_exists(bucket, dst_key):
            print(f"[skip] already exists: s3://{bucket}/{dst_key}")
            continue

        try:
            img = _open_image_from_s3(bucket, key)
            if img is None:
                print(f"[warn] failed to open: s3://{bucket}/{key}")
                continue

            out = _composite_on_black(img)
            data = _save_jpeg_to_bytes(out, quality=quality)
            _upload_bytes_to_s3(bucket, dst_key, data)
            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} images...")
        except Exception as e:
            print(f"[error] {key}: {e}")
            continue

    print(f"Done. Total processed: {processed}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stream images from S3, place on black background, save as JPEG, and upload back."
        )
    )
    p.add_argument("--bucket", type=str, default="pistoletto.moe3")
    p.add_argument(
        "--source-prefix",
        type=str,
        default="selected-images/resized_no_bg/",
        help="S3 prefix to read from",
    )
    p.add_argument(
        "--dest-prefix",
        type=str,
        default="selected-images/resized_black_bg/",
        help="S3 prefix to write processed JPEGs",
    )
    p.add_argument("--quality", type=int, default=82, help="JPEG quality (1-95)")
    p.add_argument("--overwrite", action="store_true", help="Recreate files even if they exist")
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of images to process",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_stream(
        bucket=args.bucket,
        src_prefix=args.source_prefix,
        dst_prefix=args.dest_prefix,
        overwrite=args.overwrite,
        quality=args.quality,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()


