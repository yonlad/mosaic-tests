"""
Background remover module for removing backgrounds from images.
Ported from camera_server.py.
"""
import io
import os
import base64
import requests # For fetching image from URL if temp_file_path is URL
from PIL import Image
from io import BytesIO
from flask import current_app # For logging, config, extensions
from gradio_client import Client, handle_file # handle_file is important
import uuid
import traceback # For detailed error logging

# Global Gradio API client
background_removal_client = None

def init_background_removal_client(config, logger):
    """
    Initialize the background removal Gradio client.
    """
    global background_removal_client
    
    # In original camera_server.py, client was initialized at top level.
    # We do it here, called from create_app.
    try:
        

        #Get Hugging Face Token for authentication
        hf_token = config.get('HF_TOKEN')
        if hf_token:
            logger.info("Initializing background removal client (not-lain/background-removal)...")
            background_removal_client = Client("not-lain/background-removal", hf_token=hf_token)
            logger.info("Background removal client initialized successfully.")
        else:
            logger.warning("No HF_TOKEN found, initializing background remover client without authentication")
            background_removal_client = Client("not-lain/background-removal")
            logger.info("Background removal client initialized successfully WITHOUT AUTHENTICATION.")

    except Exception as e:
        logger.error(f"Error initializing background removal client: {e}")
        logger.error(traceback.format_exc())
        background_removal_client = None # Ensure it's None on failure

def remove_background_from_image(image_url, session_id, logger):
    """
    Remove the background from an image using its S3 URL.
    Args:
        image_url: URL to the image (presigned S3 URL)
        session_id: Session ID for S3 storage path
        logger: Application logger
    Returns:
        tuple: (success, result_data)
               result_data is output_key (str) on success,
               or dict with {"key": output_key, "warning": ...} for fallback,
               or error_message (str) on failure.
    """
    global background_removal_client # Use the initialized global client

    s3_client = current_app.extensions.get('s3_client')
    s3_bucket = current_app.config.get('S3_BUCKET')

    try:
        # Extract original filename base from URL (stripping query params)
        url_path = image_url.split('?')[0]
        original_file_name_from_url = os.path.basename(url_path)
        file_name_base, _ = os.path.splitext(original_file_name_from_url)
        
        # Output will always be JPEG as per original logic after PIL processing
        output_file_name_s3 = f"{file_name_base}.jpg" # Retain original base name, ensure .jpg

        if background_removal_client:
            logger.info(f"Attempting background removal for image URL (first 50 chars): {image_url[:50]}...")
            # Use handle_file for Gradio client, it handles URLs and local paths
            # The predict API name was "/image" in the original
            prediction_result = background_removal_client.predict(
                image=handle_file(image_url),
                api_name="/image" 
            )
            
            # Original logic expected a list with a file path
            if isinstance(prediction_result, list) and len(prediction_result) > 0:
                temp_file_path = prediction_result[0] # Path to the processed image (often .webp)
                logger.info(f"Background removal API returned temp file path: {temp_file_path}")

                # Open the resulting image with PIL
                if temp_file_path.startswith(('http://', 'https://')):
                    img_response = requests.get(temp_file_path)
                    img_response.raise_for_status() # Check for HTTP errors
                    processed_img = Image.open(BytesIO(img_response.content))
                else: # Assumed local path
                    processed_img = Image.open(temp_file_path)
                
                # Convert to RGB with white background if it has alpha (e.g., from PNG or WEBP)
                if processed_img.mode in ('RGBA', 'LA'):
                    logger.info(f"Processed image has alpha channel ({processed_img.mode}), converting to RGB with white background.")
                    white_bg = Image.new("RGB", processed_img.size, (255, 255, 255))
                    # Paste using alpha channel as mask
                    alpha_channel = processed_img.split()[-1] # Get the alpha channel
                    white_bg.paste(processed_img, (0,0), mask=alpha_channel)
                    final_img_rgb = white_bg
                elif processed_img.mode != 'RGB':
                    logger.info(f"Processed image mode is {processed_img.mode}, converting to RGB.")
                    final_img_rgb = processed_img.convert('RGB')
                else:
                    final_img_rgb = processed_img # Already RGB
                
                img_byte_arr = BytesIO()
                final_img_rgb.save(img_byte_arr, format='JPEG', quality=95) # Save as JPEG
                img_byte_arr.seek(0)
                
                output_s3_key = f"selected-images/{session_id}/{output_file_name_s3}"
                
                s3_client.upload_fileobj(
                    img_byte_arr,
                    s3_bucket,
                    output_s3_key,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
                logger.info(f"Successfully uploaded background-removed image to S3: {output_s3_key}")

                # Clean up temporary file if it was local and exists
                if not temp_file_path.startswith(('http://', 'https://')) and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(f"Removed temporary file: {temp_file_path}")
                    except Exception as e_remove:
                        logger.warning(f"Error removing temporary file {temp_file_path}: {e_remove}")
                
                return True, output_s3_key
            else:
                logger.error(f"Unexpected result format from background removal API: {prediction_result}")
                return False, f"Unexpected result format from background removal API: {prediction_result}"
        else:
            # Fallback: Background removal client not available, copy original image
            logger.warning("Background removal client not available. Using original image as fallback.")
            
            # Fetch the original image from the S3 URL
            original_img_response = requests.get(image_url)
            original_img_response.raise_for_status()
            img = Image.open(BytesIO(original_img_response.content))
            
            if img.mode != 'RGB': # Ensure it's RGB before saving as JPEG
                img = img.convert('RGB')
            
            img_byte_arr_fallback = BytesIO()
            img.save(img_byte_arr_fallback, format='JPEG', quality=95)
            img_byte_arr_fallback.seek(0)
            
            output_s3_key_fallback = f"selected-images/{session_id}/{output_file_name_s3}"
            
            s3_client.upload_fileobj(
                img_byte_arr_fallback,
                s3_bucket,
                output_s3_key_fallback,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            logger.info(f"Fallback: Copied original image to S3: {output_s3_key_fallback}")
            return True, {"key": output_s3_key_fallback, "warning": "Background removal not available, original image was used."}
    
    except requests.exceptions.RequestException as req_e:
        logger.error(f"HTTP request error during background removal (fetching image or result): {req_e}")
        logger.error(traceback.format_exc())
        return False, f"Network error during background removal: {str(req_e)}"
    except Exception as e:
        logger.error(f"Error in background removal process: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)
