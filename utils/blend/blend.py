"""
Utility for blending two local images via the Flora API.

Pipeline:
  1. Upload local images to S3 (temporary)
  2. Initiate blend via Flora API
  3. Poll until complete
  4. Download result video
  5. Extract 10 evenly-spaced frames as PNGs
  6. (Optional) Remove background via Gradio API, composite onto white
  7. Save final PNGs locally
  8. Clean up S3 temp files

Usage:
  from utils.blend import blend_images

  result = blend_images("photo_a.png", "photo_b.png", output_path="output.png")

CLI:
  python -m utils.blend <image1> <image2> [output.png]
"""

import os
import time
import random
import tempfile
import requests
import boto3
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root (one level up from utils/)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ---------------------------------------------------------------------------
# Configuration from .env
# ---------------------------------------------------------------------------
FLORA_API_URL = os.getenv("FLORA_FAUNA_API_URL")
FLORA_AUTH_TOKEN = os.getenv("FLORA_AUTH_TOKEN")
GRADIO_API_KEY = os.getenv("GRADIO_API_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _get_s3_client():
    """Create and return a boto3 S3 client."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def _upload_to_s3_temp(s3_client, local_path: str, session_id: str):
    """
    Upload a local file to S3 under temp-blends/<session_id>/.
    Returns (presigned_url, s3_key).
    """
    filename = os.path.basename(local_path)
    s3_key = f"temp-blends/{session_id}/{filename}"

    ext = local_path.lower().rsplit(".", 1)[-1] if "." in local_path else ""
    content_type_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    with open(local_path, "rb") as f:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=f.read(),
            ContentType=content_type,
        )

    presigned_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": s3_key},
        ExpiresIn=3600,
    )
    return presigned_url, s3_key


def _cleanup_s3_temp(s3_client, s3_keys: list[str]):
    """Delete temporary S3 objects."""
    for key in s3_keys:
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=key)
            print(f"  Cleaned up S3 temp file: {key}")
        except Exception as e:
            print(f"  Warning: failed to clean up {key}: {e}")


# ---------------------------------------------------------------------------
# Flora API helpers
# ---------------------------------------------------------------------------

def _initiate_blend(image_url_1: str, image_url_2: str, prompt: str,
                    strength_ab: float, seed: int) -> dict:
    """POST to Flora API to start the blend. Returns the JSON response."""
    payload = {
        "prompt": prompt,
        "input_image_url": [image_url_1, image_url_2],
        "strength_ab": strength_ab,
        "seed": seed,
    }
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": FLORA_AUTH_TOKEN,
    }

    response = requests.post(f"{FLORA_API_URL}/", json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Flora API initiation failed ({response.status_code}): {response.text[:500]}"
        )
    return response.json()


def _poll_for_result(request_id: str, progress_uuid: str,
                     poll_interval: int = 10, max_polls: int = 100) -> str:
    """
    Poll the Flora API until the blend is complete.
    Returns the video/image URL on success.
    """
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": FLORA_AUTH_TOKEN,
    }
    polling_url = (
        f"{FLORA_API_URL}/result"
        f"?request_id={request_id}&progress_uuid={progress_uuid}"
    )

    for i in range(max_polls):
        time.sleep(poll_interval)

        try:
            response = requests.get(polling_url, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"  Poll {i+1}/{max_polls}: HTTP {response.status_code}, retrying...")
                continue

            result = response.json()
            status = result.get("status", "unknown").lower()
            progress = result.get("progress", 0)
            print(f"  Poll {i+1}/{max_polls}: status={status}, progress={progress}%")

            if status == "success":
                video_url = result.get("image_url") or result.get("output_url")
                if not video_url:
                    raise RuntimeError("Flora returned success but no output URL in response")
                return video_url

            if status == "error":
                error_msg = result.get("error", "Unknown Flora error")
                raise RuntimeError(f"Flora blend failed: {error_msg}")

            # Otherwise still processing -- continue polling

        except requests.exceptions.Timeout:
            print(f"  Poll {i+1}/{max_polls}: request timed out, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"  Poll {i+1}/{max_polls}: network error ({e}), retrying...")

    raise TimeoutError(f"Blend did not complete after {max_polls} polls ({max_polls * poll_interval}s)")


# ---------------------------------------------------------------------------
# Video / frame helpers
# ---------------------------------------------------------------------------

def _extract_frames(video_bytes: bytes, num_frames: int = 10) -> list[Image.Image]:
    """Extract *num_frames* evenly-spaced frames from a video, returned as PIL Images."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        try:
            import cv2
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                raise RuntimeError("cv2 failed to open video")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise RuntimeError("cv2 reported 0 frames in the video")

            num_frames = min(num_frames, total_frames)
            indices = [
                int(round(i * (total_frames - 1) / (num_frames - 1)))
                if num_frames > 1 else 0
                for i in range(num_frames)
            ]

            frames: list[Image.Image] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    print(f"  Warning: could not read frame {idx}, skipping")
            cap.release()

            if not frames:
                raise RuntimeError("cv2 could not read any frames from the video")
            return frames

        except ImportError:
            pass

        # Fallback: imageio
        import imageio
        reader = imageio.get_reader(tmp.name)
        try:
            total_frames = reader.count_frames()
            num_frames = min(num_frames, total_frames)
            indices = [
                int(round(i * (total_frames - 1) / (num_frames - 1)))
                if num_frames > 1 else 0
                for i in range(num_frames)
            ]
            frames = [Image.fromarray(reader.get_data(idx)) for idx in indices]
        finally:
            reader.close()
        return frames


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def _remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background via the Gradio 'not-lain/background-removal' space
    using the /png endpoint, then composite onto a pure white canvas.
    Returns an RGB image ready to save as PNG.
    """
    from gradio_client import Client, handle_file

    tmp_path = None
    try:
        # Save image to a temp file for the Gradio API
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        print("  Connecting to Gradio background removal API...")
        client = Client("not-lain/background-removal", hf_token=GRADIO_API_KEY)

        # /png endpoint: takes a filepath, returns a single transparent PNG
        # (the old /image endpoint now returns an ImageSlider tuple where
        #  index 0 is the original and index 1 is the processed image)
        result = client.predict(
            handle_file(tmp_path),
            api_name="/png",
        )

        # Result is a filepath string to the downloaded transparent PNG
        if isinstance(result, (list, tuple)):
            processed_path = result[0]
        elif isinstance(result, dict):
            processed_path = result.get("path") or result.get("url")
        else:
            processed_path = result

        # Open the transparent PNG
        if isinstance(processed_path, str) and processed_path.startswith(("http://", "https://")):
            resp = requests.get(processed_path)
            resp.raise_for_status()
            processed_img = Image.open(BytesIO(resp.content))
        else:
            processed_img = Image.open(processed_path)

        # Composite onto a pure white canvas
        processed_img = processed_img.convert("RGBA")
        white_canvas = Image.new("RGBA", processed_img.size, (255, 255, 255, 255))
        white_canvas.paste(processed_img, (0, 0), mask=processed_img)

        return white_canvas.convert("RGB")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def blend_images(
    image_path_1: str,
    image_path_2: str,
    output_path: str = "utils/blended_output/frame.png",
    prompt: str = "The photo of a person with a white background",
    strength_ab: float | None = None,
    seed: int | None = None,
    remove_background: bool = True,
    poll_interval: int = 10,
    max_polls: int = 100,
    num_frames: int = 10,
) -> list[str]:
    """
    Blend two local images using the Flora API, extract evenly-spaced frames,
    optionally remove the background, and save each as a PNG.

    Args:
        image_path_1:       Path to the first local image.
        image_path_2:       Path to the second local image.
        output_path:        Base path for saved PNGs. Frame index is inserted
                            before the extension (e.g. out_01.png … out_10.png).
        prompt:             Text prompt sent to the Flora API.
        strength_ab:        Blend strength 0.0-1.0 (random if None).
        seed:               Random seed 0-1000 (random if None).
        remove_background:  Run Gradio background removal (default: True).
        poll_interval:      Seconds between Flora poll attempts (default: 10).
        max_polls:          Max Flora poll attempts (default: 100).
        num_frames:         Number of evenly-spaced frames to extract (default: 10).

    Returns:
        A list of absolute paths to the saved output PNGs.
    """
    # Validate inputs
    for p in (image_path_1, image_path_2):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    # Apply random defaults
    if strength_ab is None:
        strength_ab = round(random.uniform(0, 1), 4)
    if seed is None:
        seed = random.randint(0, 1000)

    session_id = f"blend_{int(time.time())}_{random.randint(1000, 9999)}"
    s3_client = _get_s3_client()
    temp_s3_keys: list[str] = []

    try:
        # -- Step 1: Upload to S3 ------------------------------------------
        print(f"[1/6] Uploading images to S3 (temp-blends/{session_id}/)...")
        url_1, key_1 = _upload_to_s3_temp(s3_client, image_path_1, session_id)
        temp_s3_keys.append(key_1)
        url_2, key_2 = _upload_to_s3_temp(s3_client, image_path_2, session_id)
        temp_s3_keys.append(key_2)
        print("  Both images uploaded.")

        # -- Step 2: Initiate blend ----------------------------------------
        print(f"[2/6] Initiating Flora blend (strength_ab={strength_ab}, seed={seed})...")
        init_result = _initiate_blend(url_1, url_2, prompt, strength_ab, seed)
        request_id = init_result.get("request_id")
        progress_uuid = init_result.get("progress_uuid")
        if not request_id or not progress_uuid:
            raise RuntimeError(
                f"Flora response missing request_id or progress_uuid: {init_result}"
            )
        print(f"  Blend initiated (request_id={request_id}).")

        # -- Step 3: Poll for result ---------------------------------------
        print(f"[3/6] Polling for result (every {poll_interval}s, up to {max_polls} attempts)...")
        video_url = _poll_for_result(request_id, progress_uuid, poll_interval, max_polls)
        print("  Blend complete! Video URL received.")

        # -- Step 4: Download video ----------------------------------------
        print("[4/6] Downloading result video...")
        video_response = requests.get(video_url, timeout=120)
        video_response.raise_for_status()
        video_bytes = video_response.content
        print(f"  Downloaded {len(video_bytes) / 1024:.1f} KB.")

        # -- Step 5: Extract frames ----------------------------------------
        print(f"[5/6] Extracting {num_frames} evenly-spaced frames from video...")
        frames = _extract_frames(video_bytes, num_frames=num_frames)
        print(f"  Extracted {len(frames)} frames ({frames[0].size[0]}x{frames[0].size[1]}).")

        # -- Step 6: Background removal (optional) + save ------------------
        base, ext = os.path.splitext(output_path)
        ext = ext or ".png"
        saved_paths: list[str] = []

        for i, frame in enumerate(frames, start=1):
            frame_label = f"frame {i}/{len(frames)}"

            if remove_background:
                print(f"[6/6] Running background removal on {frame_label}...")
                frame = _remove_background(frame)
            else:
                if frame.mode != "RGB":
                    frame = frame.convert("RGB")

            frame_path = f"{base}_{i:02d}{ext}"
            frame.save(frame_path, format="PNG")
            saved_paths.append(os.path.abspath(frame_path))
            print(f"  Saved {frame_label} -> {frame_path}")

        print(f"\nDone! Saved {len(saved_paths)} frames.")
        return saved_paths

    finally:
        if temp_s3_keys:
            print("\nCleaning up S3 temp files...")
            _cleanup_s3_temp(s3_client, temp_s3_keys)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blend two images via the Flora API.")
    parser.add_argument("-image1", default="utils/1.jpg", help="Path to the first image")
    parser.add_argument("-image2", default="utils/2.png", help="Path to the second image")
    parser.add_argument("-o", "--output", default="utils/blended_output/frame.png", help="Output PNG path")
    parser.add_argument("-p", "--prompt", default="A medium shot photo of a person with a white background")
    parser.add_argument("-s", "--strength", type=float, default=float(random.uniform(0, 1)), help="Blend strength 0.0-1.0")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000), help="Random seed 0-1000")
    parser.add_argument("--no-bg-removal", action="store_true", help="Skip background removal")
    parser.add_argument("--poll-interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--max-polls", type=int, default=100, help="Max poll attempts")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of evenly-spaced frames to extract")

    args = parser.parse_args()

    blend_images(
        image_path_1=args.image1,
        image_path_2=args.image2,
        output_path=args.output,
        prompt=args.prompt,
        strength_ab=args.strength,
        seed=args.seed,
        remove_background=not args.no_bg_removal,
        poll_interval=args.poll_interval,
        max_polls=args.max_polls,
        num_frames=args.num_frames,
    )
