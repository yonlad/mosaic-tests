#!/usr/bin/env python

import os
import math
import random
import traceback
import gc
import boto3
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from scipy import ndimage
from datetime import datetime
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from skimage.filters import threshold_otsu
    from skimage import color, feature, exposure
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not installed. Some advanced features will be limited.")
    SKIMAGE_AVAILABLE = False
    threshold_otsu = None

# Load environment variables
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('REACT_APP_AWS_REGION', 'us-east-2')
AWS_ACCESS_KEY_ID = os.getenv('REACT_APP_AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('REACT_APP_AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('REACT_APP_S3_BUCKET', 'eternity-mirror-project')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Mosaic configuration - optimized for seamless detailed appearance
THUMBNAIL_SIZE = (2, 2)  # Very small for seamless blending
INTERNAL_THUMBNAIL_SIZE = (64, 64)  # Higher resolution for better detail
CELL_SIZE = (2, 2)  # Match thumbnail size for dense, seamless coverage
CONTRAST_FACTOR = 1.2  # Slight contrast boost for better feature definition
BRIGHTNESS_THRESHOLD = 240  # Slightly lower to include more detail areas
THUMBNAIL_LIMIT = 3000  # More thumbnails for better variety
SKIP_PROBABILITY = 0.0  # No skipping for complete coverage
FOREGROUND_THRESHOLD = 0.03  # More sensitive to capture fine details
POSITION_RANDOMNESS = 0.0  # Perfect grid alignment
DETAIL_SENSITIVITY = 0.5  # Not used but kept for compatibility
USE_VARIABLE_SIZES = False  # Uniform sizes for seamless appearance
EDGE_ALIGNMENT = False  # Disabled for performance
MIN_THUMBNAIL_SCALE = 0.8  # Not used when variable sizes disabled
MAX_THUMBNAIL_SCALE = 1.0  # Not used when variable sizes disabled

def get_average_color(img):
    """Calculate the average color of an image."""
    img_array = np.array(img)
    avg_color = np.mean(img_array, axis=(0, 1))
    return avg_color

def calculate_detail_level(cell_image, edges_array, cell_x, cell_y, cell_width, cell_height):
    """Calculate the level of detail in a cell based on edge density"""
    # Extract the edge data for this cell
    cell_edges = edges_array[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
    # Calculate the average edge strength as a measure of detail
    edge_density = np.mean(cell_edges) / 255.0  # Normalize to 0-1
    return edge_density

def create_detail_map(image, edges):
    """Create a map of detail levels across the image"""
    # Blur the edges to get regions of detail
    detail_map = ndimage.gaussian_filter(edges, sigma=5.0)
    return detail_map / np.max(detail_map) if np.max(detail_map) > 0 else detail_map  # Normalize to 0-1

def get_edge_orientation(edges, x, y, window_size=15):
    """Get the dominant edge orientation at a location"""
    # Extract local window
    h, w = edges.shape
    x1, y1 = max(0, x-window_size//2), max(0, y-window_size//2)
    x2, y2 = min(w, x+window_size//2), min(h, y+window_size//2)
    window = edges[y1:y2, x1:x2]
    
    if not SKIMAGE_AVAILABLE or np.sum(window) == 0:
        return 0  # Default no rotation
        
    try:
        # Use Hough transform to find dominant line orientation
        h, theta, d = feature.hough_line(window > 0)
        if len(h) > 0 and np.max(h) > 0:
            # Return the angle of the most prominent line
            idx = np.argmax(h)
            angle_rad = theta[idx]
            angle_deg = angle_rad * 180 / np.pi
            # Convert to 0, 90, 180, or 270 for simplicity
            angles = [0, 90, 180, 270]
            closest_angle = min(angles, key=lambda a: abs((angle_deg % 180) - a))
            return closest_angle
    except Exception as e:
        print(f"Error in edge orientation detection: {e}")
    
    return 0  # Default no rotation

def find_best_match(target_color, thumbnails, cell_detail=0.0):
    """Find the thumbnail with the closest average color to the target color.
    
    Enhanced version for better color accuracy with tiny thumbnails.
    """
    min_distance = float('inf')
    best_match = None
    
    target_color = np.array(target_color).astype(float)
    
    # Use perceptually weighted RGB distance for better color matching
    # These weights approximate human color perception
    weights = np.array([0.299, 0.587, 0.114])  # Standard luminance weights
    
    # For very small thumbnails, we need more precise color matching
    # Add a small amount of randomness to avoid repetitive patterns
    candidates = []
    
    for thumbnail in thumbnails:
        # Calculate weighted Euclidean distance in RGB space
        diff = target_color - thumbnail['avg_color']
        distance = np.sqrt(np.sum((diff * weights) ** 2))
        
        candidates.append((distance, thumbnail))
    
    # Sort by distance and pick from the best matches
    candidates.sort(key=lambda x: x[0])
    
    # For tiny thumbnails, occasionally pick from top 3 matches to avoid repetition
    if len(candidates) >= 3 and random.random() < 0.15:  # 15% chance for variety
        best_match = candidates[random.randint(0, 2)][1]
    else:
        best_match = candidates[0][1]
    
    return best_match

def enhance_image(img, contrast_factor=CONTRAST_FACTOR):
    """Enhanced image processing for better detail preservation with tiny thumbnails."""
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create a copy for processing
    enhanced_img = img.copy()
    
    # Apply subtle contrast enhancement for better feature definition
    if contrast_factor > 1.0:
        enhancer = ImageEnhance.Contrast(enhanced_img)
        enhanced_img = enhancer.enhance(contrast_factor)
    
    # Apply very subtle sharpening to help with detail detection
    enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    return enhanced_img

def fetch_thumbnails_from_s3(limit=THUMBNAIL_LIMIT, prefix="selected-images/"):
    """Fetch thumbnail images from S3 bucket and process for mosaic use."""
    thumbnails = []
    thumbnails_lock = threading.Lock()
    
    def process_s3_object(key):
        """Process a single S3 object and return thumbnail data."""
        try:
            # Download the image from S3
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            image_data = response['Body'].read()
            
            # Open image using PIL
            img = Image.open(BytesIO(image_data))
            
            # Ensure the image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create internal resolution version directly (skip preview step)
            hi_res_img = img.resize(INTERNAL_THUMBNAIL_SIZE, Image.LANCZOS)
            
            # Calculate average color from the resized image
            img_array = np.array(hi_res_img)
            avg_color = np.mean(img_array, axis=(0, 1)).astype(int)
            
            return {
                's3_key': key,
                'avg_color': avg_color,
                'image': hi_res_img,
                'display_size': THUMBNAIL_SIZE
            }
            
        except Exception as e:
            print(f"Error processing S3 image {key}: {e}")
            return None
    
    try:
        print(f"Fetching up to {limit} thumbnails from S3...")
        
        # Get list of all image keys first
        paginator = s3_client.get_paginator('list_objects_v2')
        image_keys = []
        
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_keys.append(key)
                    if len(image_keys) >= limit:
                        break
            
            if len(image_keys) >= limit:
                break
        
        print(f"Found {len(image_keys)} images, processing with concurrent downloads...")
        
        # Process images concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_key = {executor.submit(process_s3_object, key): key for key in image_keys[:limit]}
            
            # Collect results as they complete
            for future in as_completed(future_to_key):
                result = future.result()
                if result:
                    with thumbnails_lock:
                        thumbnails.append(result)
                        if len(thumbnails) % 100 == 0:
                            print(f"Processed {len(thumbnails)} thumbnails...")
                
        print(f"Successfully fetched {len(thumbnails)} thumbnails for processing")
        
    except Exception as e:
        print(f"Error fetching thumbnails from S3: {e}")
        traceback.print_exc()
        
    return thumbnails

def create_mosaic(img, thumbnails):
    """Create a mosaic of the input image using the provided thumbnails."""
    try:
        # Size check - restrict to reasonable dimensions to prevent memory issues
        max_dimension = 800
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.BILINEAR)
            print(f"Resized source image to {new_width}x{new_height}")
            
        # Save a copy of the original image for reference and color matching
        original_img = img.copy()
            
        # Enhance the image for edge detection ONLY
        # The enhanced version is only used for detail detection, not color matching
        enhanced_img = enhance_image(img)
        
        # Calculate cell grid
        n_cols = math.floor(img.width / CELL_SIZE[0])
        n_rows = math.floor(img.height / CELL_SIZE[1])
        
        # Ensure a reasonable number of cells
        max_cells = 3000
        if n_cols * n_rows > max_cells:
            scale_factor = math.sqrt(max_cells / (n_cols * n_rows))
            n_cols = max(1, math.floor(n_cols * scale_factor))
            n_rows = max(1, math.floor(n_rows * scale_factor))
            print(f"Limiting mosaic to {n_cols}x{n_rows} cells for stability")
        
        # Resize images to match cell grid
        enhanced_img = enhanced_img.resize((n_cols * CELL_SIZE[0], n_rows * CELL_SIZE[1]), Image.BILINEAR)
        original_img = original_img.resize((n_cols * CELL_SIZE[0], n_rows * CELL_SIZE[1]), Image.BILINEAR)
        
        # Calculate the mosaic dimensions - matching the grid spacing
        mosaic_width = n_cols * CELL_SIZE[0]
        mosaic_height = n_rows * CELL_SIZE[1]
        
        # Calculate the high-resolution mosaic dimensions 
        scale_factor = INTERNAL_THUMBNAIL_SIZE[0] / THUMBNAIL_SIZE[0]
        hi_res_width = int(mosaic_width * scale_factor)
        hi_res_height = int(mosaic_height * scale_factor)
        
        # Create the high-resolution canvas - use a light gray instead of pure white
        mosaic = Image.new('RGB', (hi_res_width, hi_res_height), (245, 245, 245))
        
        filled_cells = 0
        total_cells = n_cols * n_rows
        print(f"Creating high-resolution mosaic grid: {n_cols}x{n_rows} at {scale_factor}x scale")
        
        # Use enhanced image for basic edge detection (simplified)
        enhanced_gray = enhanced_img.convert('L')
        edges = enhanced_gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy arrays for processing
        enhanced_array = np.array(enhanced_gray)
        edges_array = np.array(edges)
        
        # Simplified foreground detection using Otsu's method
        try:
            if threshold_otsu:
                threshold = threshold_otsu(enhanced_array)
            else:
                # Fallback if skimage is not available
                threshold = np.mean(enhanced_array) * 0.8
        except:
            # Fallback if method fails
            threshold = np.mean(enhanced_array) * 0.8
        
        # Create foreground mask - invert so darker areas (typically subject) are True
        foreground_mask = enhanced_array < threshold
        
        # Apply basic morphological operations to clean up the mask
        foreground_mask = ndimage.binary_closing(foreground_mask, structure=np.ones((5, 5)))
        foreground_mask = ndimage.binary_opening(foreground_mask, structure=np.ones((3, 3)))
        foreground_mask = ndimage.binary_dilation(foreground_mask, structure=np.ones((7, 7)))
        
        print(f"Created simplified foreground mask for subject detection")
        
        # Process grid by extracting cell average colors - use ORIGINAL image for colors
        cell_colors = []
        cell_is_foreground = []
        
        for y in range(n_rows):
            row_colors = []
            row_foreground = []
            for x in range(n_cols):
                cell_x = x * CELL_SIZE[0]
                cell_y = y * CELL_SIZE[1]
                # Use the original (non-enhanced) image for color extraction
                cell = original_img.crop((cell_x, cell_y, cell_x + CELL_SIZE[0], cell_y + CELL_SIZE[1]))
                avg_color = np.array(get_average_color(cell).astype(int))
                row_colors.append(avg_color)
                
                # Check if this cell is part of the foreground (person)
                cell_mask = foreground_mask[cell_y:cell_y + CELL_SIZE[1], cell_x:cell_x + CELL_SIZE[0]]
                is_foreground = np.mean(cell_mask) > FOREGROUND_THRESHOLD
                row_foreground.append(is_foreground)
                
                del cell
            cell_colors.append(row_colors)
            cell_is_foreground.append(row_foreground)
        
        # Release the source images from memory
        del enhanced_img
        del original_img
        del enhanced_gray
        del edges
        gc.collect()

        # Process each cell to place high-resolution thumbnails
        for y in range(n_rows):
            for x in range(n_cols):
                # Skip if not part of the foreground (person)
                if not cell_is_foreground[y][x]:
                    continue
                
                # No skipping for dense coverage (SKIP_PROBABILITY = 0.0)
                    
                # Get the average color
                avg_color = cell_colors[y][x]
                
                # Skip very bright cells (keep this for background filtering)
                brightness = np.mean(avg_color)
                if brightness > BRIGHTNESS_THRESHOLD:
                    continue
                    
                # Find the best match (simplified function)
                best_match = find_best_match(avg_color, thumbnails)
                if not best_match:
                    continue
                    
                # Calculate the high-resolution position (no randomness for uniform grid)
                hi_res_x = int(x * CELL_SIZE[0] * scale_factor)
                hi_res_y = int(y * CELL_SIZE[1] * scale_factor)

                # Ensure the thumbnail stays within the mosaic bounds
                hi_res_x = max(0, min(hi_res_x, hi_res_width - INTERNAL_THUMBNAIL_SIZE[0]))
                hi_res_y = max(0, min(hi_res_y, hi_res_height - INTERNAL_THUMBNAIL_SIZE[1]))
                
                # Use uniform thumbnail size (no variable sizes)
                thumbnail_to_paste = best_match['image']
                
                # No rotation for uniform appearance and better performance
                
                # Place the high-resolution thumbnail
                mosaic.paste(thumbnail_to_paste, (hi_res_x, hi_res_y))
                filled_cells += 1
                
                # Periodic status updates
                if filled_cells % 200 == 0:
                    print(f"Placed {filled_cells} thumbnails so far...")
        
        print(f"High-resolution mosaic created with {filled_cells}/{total_cells} cells filled")
        print(f"Final mosaic dimensions before rotation: {hi_res_width}x{hi_res_height} pixels")

        # rotate if needed
        if mosaic.width > mosaic.height:
            rotated_mosaic = mosaic.transpose(Image.ROTATE_270)
        else:
            rotated_mosaic = mosaic
        
        print(f"Rotated mosaic to portrait orientation: {rotated_mosaic.width}x{rotated_mosaic.height} pixels")
        
        # Resize the mosaic to be smaller in relation to the white background canvas
        WHITE_BACKGROUND_SIZE = (rotated_mosaic.width, rotated_mosaic.height)
 
        # Calculate the aspect ratio of the original mosaic
        aspect_ratio = rotated_mosaic.width / rotated_mosaic.height
 
        # create a new white background canvas
        white_background = Image.new('RGB', WHITE_BACKGROUND_SIZE, (245, 245, 245))
         
        # Calculate the new dimensions that fit within the white background
        new_width = math.floor(rotated_mosaic.width / 2)
        new_height = math.floor(rotated_mosaic.height / 2)
         
        # Resize the mosaic to fit within the white background
        resized_mosaic = rotated_mosaic.resize((new_width, new_height), Image.LANCZOS)
         
        # Calculate the position to center the mosaic within the white background
        x = (WHITE_BACKGROUND_SIZE[0] - new_width) // 2
        y = (WHITE_BACKGROUND_SIZE[1] - new_height) // 2
 
        # Paste the resized mosaic onto the white background
        white_background.paste(resized_mosaic, (x, y))
         
        # Return the final mosaic
        return white_background

    except Exception as e:
        print(f"Error in mosaic creation: {e}")
        print(traceback.format_exc())
        raise

def save_mosaic(mosaic, output_path, quality=92):
    """Save the mosaic to a file."""
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the mosaic
        mosaic.save(output_path, format="JPEG", quality=quality)
        print(f"Mosaic saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving mosaic: {e}")
        return False

def load_image_from_local(image_path):
    """Load image from local file."""
    try:
        img = Image.open(image_path)
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading local image: {e}")
        return None

def load_image_from_s3(s3_key):
    """Load image from S3 bucket."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        image_data = response['Body'].read()
        img = Image.open(BytesIO(image_data))
        # Ensure the image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading S3 image: {e}")
        return None

def main():
    # Declare globals at the beginning of the function
    global THUMBNAIL_SIZE, INTERNAL_THUMBNAIL_SIZE, CELL_SIZE, CONTRAST_FACTOR
    global BRIGHTNESS_THRESHOLD, THUMBNAIL_LIMIT, SKIP_PROBABILITY
    global FOREGROUND_THRESHOLD, POSITION_RANDOMNESS, DETAIL_SENSITIVITY
    global USE_VARIABLE_SIZES, EDGE_ALIGNMENT, MIN_THUMBNAIL_SCALE
    
    parser = argparse.ArgumentParser(description='Generate a mosaic from an image using thumbnails from S3')
    parser.add_argument('--image', type=str, default='capture_20250502_131625.jpg', 
                        help='Local image path or S3 key')
    parser.add_argument('--is-s3-key', action='store_true',
                        help='Flag to indicate if the image is an S3 key')
    parser.add_argument('--thumbnail-limit', type=int, default=THUMBNAIL_LIMIT,
                        help='Maximum number of thumbnails to use')
    parser.add_argument('--thumbnail-size', type=int, nargs=2, default=list(THUMBNAIL_SIZE),
                        help='Thumbnail size for layout (w h)')
    parser.add_argument('--internal-thumbnail-size', type=int, nargs=2, default=list(INTERNAL_THUMBNAIL_SIZE),
                        help='Internal thumbnail size for high-res detail (w h)')
    parser.add_argument('--cell-size', type=int, nargs=2, default=list(CELL_SIZE),
                        help='Cell size for spacing thumbnails (w h)')
    parser.add_argument('--contrast-factor', type=float, default=CONTRAST_FACTOR,
                        help='Contrast enhancement factor')
    parser.add_argument('--brightness-threshold', type=int, default=BRIGHTNESS_THRESHOLD,
                        help='Brightness threshold for skipping cells')
    parser.add_argument('--skip-probability', type=float, default=SKIP_PROBABILITY,
                        help='Probability of skipping a cell (lower = more thumbnails)')
    parser.add_argument('--foreground-threshold', type=float, default=FOREGROUND_THRESHOLD,
                        help='Threshold for foreground detection')
    parser.add_argument('--position-randomness', type=float, default=POSITION_RANDOMNESS,
                        help='Randomness in thumbnail positions')
    parser.add_argument('--detail-sensitivity', type=float, default=DETAIL_SENSITIVITY,
                        help='Sensitivity to image details (0-1)')
    parser.add_argument('--variable-sizes', action='store_true', default=USE_VARIABLE_SIZES,
                        help='Use variable thumbnail sizes based on detail')
    parser.add_argument('--edge-alignment', action='store_true', default=EDGE_ALIGNMENT,
                        help='Align thumbnails with detected edges')
    parser.add_argument('--min-thumbnail-scale', type=float, default=MIN_THUMBNAIL_SCALE,
                        help='Minimum scale factor for thumbnails in detailed areas')
    
    args = parser.parse_args()
    
    # Update global configuration based on arguments
    THUMBNAIL_SIZE = tuple(args.thumbnail_size)
    INTERNAL_THUMBNAIL_SIZE = tuple(args.internal_thumbnail_size)
    CELL_SIZE = tuple(args.cell_size)
    CONTRAST_FACTOR = args.contrast_factor
    BRIGHTNESS_THRESHOLD = args.brightness_threshold
    THUMBNAIL_LIMIT = args.thumbnail_limit
    SKIP_PROBABILITY = args.skip_probability
    FOREGROUND_THRESHOLD = args.foreground_threshold
    POSITION_RANDOMNESS = args.position_randomness
    DETAIL_SENSITIVITY = args.detail_sensitivity
    USE_VARIABLE_SIZES = args.variable_sizes
    EDGE_ALIGNMENT = args.edge_alignment
    MIN_THUMBNAIL_SCALE = args.min_thumbnail_scale
    
    # Print current configuration
    print("Mosaic Configuration:")
    print(f"  Thumbnail Size: {THUMBNAIL_SIZE}")
    print(f"  Internal Thumbnail Size: {INTERNAL_THUMBNAIL_SIZE}")
    print(f"  Cell Size: {CELL_SIZE}")
    print(f"  Contrast Factor: {CONTRAST_FACTOR}")
    print(f"  Brightness Threshold: {BRIGHTNESS_THRESHOLD}")
    print(f"  Thumbnail Limit: {THUMBNAIL_LIMIT}")
    print(f"  Skip Probability: {SKIP_PROBABILITY}")
    print(f"  Foreground Threshold: {FOREGROUND_THRESHOLD}")
    print(f"  Position Randomness: {POSITION_RANDOMNESS}")
    print(f"  Detail Sensitivity: {DETAIL_SENSITIVITY}")
    print(f"  Use Variable Sizes: {USE_VARIABLE_SIZES}")
    print(f"  Edge Alignment: {EDGE_ALIGNMENT}")
    print(f"  Min Thumbnail Scale: {MIN_THUMBNAIL_SCALE}")
    
    start_time = time.time()
    
    # Load the source image
    if args.is_s3_key:
        print(f"Loading source image from S3: {args.image}")
        source_image = load_image_from_s3(args.image)
    else:
        print(f"Loading source image from local file: {args.image}")
        source_image = load_image_from_local(args.image)
    
    if not source_image:
        print("Failed to load source image. Exiting.")
        return
    
    print(f"Source image loaded successfully. Dimensions: {source_image.size}")
    
    # Fetch thumbnails from S3
    thumbnails = fetch_thumbnails_from_s3(limit=THUMBNAIL_LIMIT)
    
    if not thumbnails:
        print("No thumbnails available for creating mosaic. Exiting.")
        return
    
    # Create the mosaic
    print(f"Creating mosaic using {len(thumbnails)} thumbnails...")
    mosaic = create_mosaic(source_image, thumbnails)
    
    # Clean up memory
    del source_image
    del thumbnails
    gc.collect()
    
    # Save the mosaic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"mosaic_{timestamp}.jpg"
    output_path = os.path.join("output", output_filename)
    save_mosaic(mosaic, output_path)
    
    # Save a smaller thumbnail for quick viewing
    thumbnail_size = (800, 1200)
    mosaic_thumbnail = mosaic.copy()
    mosaic_thumbnail.thumbnail(thumbnail_size, Image.LANCZOS)
    thumbnail_path = os.path.join("output", f"thumbnail_{timestamp}.jpg")
    save_mosaic(mosaic_thumbnail, thumbnail_path, quality=80)
    
    end_time = time.time()
    print(f"Mosaic generation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
