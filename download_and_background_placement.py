"""
A script to download images from specific S3 folders, remove their backgrounds,
and place them on a randomly selected colored background.
This version processes images concurrently in a multi-stage pipeline.
"""
import os
import argparse
import random
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from gradio_client import Client, handle_file
from PIL import Image
import requests
from io import BytesIO
import logging
import traceback
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- Configuration ---
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('REACT_APP_AWS_REGION', 'us-east-2')
AWS_ACCESS_KEY_ID = os.getenv('REACT_APP_AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('REACT_APP_AWS_SECRET_ACCESS_KEY')


# Hugging Face Configuration - get from environment variable for authentication
HF_TOKEN = os.environ.get("GRADIO_API_KEY")

# Background Colors and their weights for randomization
# Format: { "color_name": ((R, G, B), weight) }
# The weights are relative and do not need to sum to 100.
# For example, two colors with weight 20 and 80 will have a 20% and 80% chance respectively.
BACKGROUND_COLORS = {
    "white": ((255, 255, 255), 70),
    "black": ((0, 0, 0), 30),
}

# Default local directories
DEFAULT_DOWNLOAD_DIR = "s3_downloads"
DEFAULT_BG_REMOVED_DIR = "bg_removed"
DEFAULT_FINAL_DIR = "final_images"
DEFAULT_WORKERS = 5

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gradio Client ---
background_removal_client = None

def init_background_removal_client():
    """
    Initialize the background removal Gradio client from Hugging Face.
    """
    global background_removal_client
    try:
        if HF_TOKEN:
            logger.info("Initializing background removal client (not-lain/background-removal) with HF Token...")
            background_removal_client = Client("not-lain/background-removal", hf_token=HF_TOKEN)
            logger.info("Background removal client initialized successfully.")
        else:
            logger.warning("No HF_TOKEN environment variable found, initializing client without authentication.")
            background_removal_client = Client("not-lain/background-removal")
            logger.info("Background removal client initialized successfully WITHOUT AUTHENTICATION.")
    except Exception as e:
        logger.error(f"Error initializing background removal client: {e}")
        logger.error(traceback.format_exc())
        background_removal_client = None

# --- S3 & Pipeline Functions ---

def get_all_s3_keys(bucket_name, s3_folders, s3_client):
    """
    Get a list of all object keys from the specified S3 folders.
    """
    all_keys = []
    logger.info(f"Fetching file list from S3 bucket '{bucket_name}'...")
    for folder in s3_folders:
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=folder)
            
            folder_keys = [obj['Key'] for page in pages if "Contents" in page for obj in page["Contents"] if not obj['Key'].endswith('/')]
            all_keys.extend(folder_keys)
            logger.info(f"Found {len(folder_keys)} files in s3://{bucket_name}/{folder}")
        except ClientError as e:
            logger.error(f"Could not access s3://{bucket_name}/{folder}. Error: {e}. Skipping.")
    
    logger.info(f"Total files to process: {len(all_keys)}")
    return all_keys

def process_image_pipeline(s3_key, bucket_name, download_dir, bg_removed_dir, final_dir):
    """
    The full processing pipeline for a single image, designed to be run in a thread.
    1. Downloads image from S3.
    2. Removes background.
    3. Places on a new colored background.
    """
    try:
        # Each thread needs its own S3 client
        s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

        base_filename = os.path.basename(s3_key)
        name, _ = os.path.splitext(base_filename)

        # --- Stage 1: Download ---
        download_path = os.path.join(download_dir, base_filename)
        try:
            s3_client.download_file(bucket_name, s3_key, download_path)
        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return # Stop processing this image if download fails

        # --- Stage 2: Remove Background ---
        bg_removed_pil = remove_background(download_path)
        if not bg_removed_pil:
            logger.warning(f"Skipping {base_filename} due to background removal failure.")
            return

        # Save intermediate background-removed image (as PNG to preserve transparency)
        bg_removed_path = os.path.join(bg_removed_dir, f"{name}_bg_removed.png")
        try:
            bg_removed_pil.save(bg_removed_path, "PNG")
        except Exception as e:
            logger.error(f"Could not save background-removed image {bg_removed_path}: {e}")
            return

        # --- Stage 3: Place on Colored Background ---
        color = choose_background_color()
        final_pil = place_on_background(bg_removed_pil, color)
        if not final_pil:
            logger.warning(f"Skipping {base_filename} due to background placement failure.")
            return
        
        # Save final image (as JPEG)
        final_path = os.path.join(final_dir, f"{name}_final.jpg")
        try:
            final_pil.save(final_path, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Could not save final image {final_path}: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {s3_key}: {e}")
        logger.error(traceback.format_exc())

# --- Image Processing Functions ---
def remove_background(image_path):
    """
    Remove the background from a single image using the Gradio client.
    Includes retry logic with exponential backoff for network resilience.
    Returns a PIL Image object with an alpha channel for transparency.
    """
    if not background_removal_client:
        logger.error("Background removal client is not initialized.")
        return None

    max_retries = 3
    base_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            logger.debug(f"Removing background for: {image_path} (Attempt {attempt + 1}/{max_retries})")
            prediction_result = background_removal_client.predict(image=handle_file(image_path), api_name="/image")

            if isinstance(prediction_result, (list, tuple)) and len(prediction_result) > 0:
                temp_file_path = prediction_result[0]
                logger.debug(f"Background removal API returned temp file: {temp_file_path}")

                if temp_file_path.startswith(('http://', 'https://')):
                    img_response = requests.get(temp_file_path)
                    img_response.raise_for_status()
                    processed_img = Image.open(BytesIO(img_response.content))
                else:
                    processed_img = Image.open(temp_file_path)

                if processed_img.mode not in ('RGBA', 'LA'):
                    logger.warning(f"Processed image for {image_path} does not have an alpha channel (mode: {processed_img.mode}). Converting to RGBA.")
                    processed_img = processed_img.convert('RGBA')

                if not temp_file_path.startswith(('http://', 'https://')) and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file_path}: {e}")
                
                return processed_img # Success
            else:
                logger.error(f"Unexpected result from background removal API for {image_path}: {prediction_result}")
                return None # Non-retryable error

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} for {os.path.basename(image_path)} failed: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for {os.path.basename(image_path)}.")
                # Log full traceback on the last attempt for debugging
                logger.error(traceback.format_exc())
                return None
    
    return None # Fallback

def place_on_background(image_with_alpha, color):
    """
    Places a PIL image with an alpha channel onto a solid colored background.
    """
    if image_with_alpha.mode not in ('RGBA', 'LA'):
        logger.error(f"Image must have an alpha channel to be placed on a background. Got {image_with_alpha.mode}.")
        return None
        
    background = Image.new("RGB", image_with_alpha.size, color)
    background.paste(image_with_alpha, (0, 0), image_with_alpha)
    return background

def choose_background_color():
    """
    Chooses a random background color based on the weights in BACKGROUND_COLORS.
    """
    if not BACKGROUND_COLORS:
        logger.warning("BACKGROUND_COLORS dictionary is empty. Defaulting to white background.")
        return (255, 255, 255)
        
    items = list(BACKGROUND_COLORS.values())
    colors = [item[0] for item in items]
    weights = [item[1] for item in items]
    
    return random.choices(colors, weights=weights, k=1)[0]

# --- Main Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, process, and place images from S3 concurrently.")
    parser.add_argument("--bucket", default="mosaic.tests", help="The S3 bucket name.")
    parser.add_argument("--folders", nargs='+', default=["selected-images"], help="A list of S3 folders (prefixes) to process.")
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR, help=f"Directory for original downloads. Default: {DEFAULT_DOWNLOAD_DIR}")
    parser.add_argument("--bg-removed-dir", default=DEFAULT_BG_REMOVED_DIR, help=f"Directory for background-removed images. Default: {DEFAULT_BG_REMOVED_DIR}")
    parser.add_argument("--final-dir", default=DEFAULT_FINAL_DIR, help=f"Directory for final images with background. Default: {DEFAULT_FINAL_DIR}")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of concurrent worker threads. Default: {DEFAULT_WORKERS}")

    args = parser.parse_args()

    # --- Setup ---
    init_background_removal_client()
    if not background_removal_client:
        logger.error("Exiting because background removal client could not be initialized.")
        exit(1)

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.error("AWS credentials not found in environment variables. Please set REACT_APP_AWS_ACCESS_KEY_ID and REACT_APP_AWS_SECRET_ACCESS_KEY. Exiting.")
        exit(1)

    # Create output directories
    os.makedirs(args.download_dir, exist_ok=True)
    os.makedirs(args.bg_removed_dir, exist_ok=True)
    os.makedirs(args.final_dir, exist_ok=True)
    
    # --- Execution ---
    # Create a single S3 client for listing keys
    s3_client = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    all_s3_keys = get_all_s3_keys(args.bucket, args.folders, s3_client)

    if not all_s3_keys:
        logger.info("No images to process. Exiting.")
        exit(0)

    # Check for already processed images to allow for resuming
    logger.info("Checking for already processed images to skip...")
    keys_to_process = []
    for key in all_s3_keys:
        base_filename = os.path.basename(key)
        name, _ = os.path.splitext(base_filename)
        final_path = os.path.join(args.final_dir, f"{name}_final.jpg")
        if not os.path.exists(final_path):
            keys_to_process.append(key)

    skipped_count = len(all_s3_keys) - len(keys_to_process)
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} images that were already processed.")
    
    if not keys_to_process:
        logger.info("All images have already been processed. Exiting.")
        exit(0)
    
    logger.info(f"Found {len(keys_to_process)} new images to process.")

    # Process all images concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_image_pipeline, key, args.bucket, args.download_dir, args.bg_removed_dir, args.final_dir)
            for key in keys_to_process
        ]

        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(futures), total=len(keys_to_process), desc="Processing images"):
            try:
                future.result()  # We call result() to raise any exceptions from the thread
            except Exception as exc:
                logger.error(f'A worker generated an exception: {exc}')

    logger.info("Script finished.")
