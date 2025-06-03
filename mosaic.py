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

# Mosaic configuration - you can adjust these parameters to experiment
THUMBNAIL_SIZE = (1, 1)  # Size for display/layout - smaller for photorealistic
INTERNAL_THUMBNAIL_SIZE = (100, 100)  # Resolution for zoom detail
CELL_SIZE = (1, 1)  # Spacing between thumbnails (controls density) - smaller for photorealistic
CONTRAST_FACTOR = 1.0  # Contrast enhancement disabled by default
BRIGHTNESS_THRESHOLD = 250  # Include almost all brightness levels
THUMBNAIL_LIMIT = 8000  # Number of thumbnails to use (adjust based on memory) - more for better variety
SKIP_PROBABILITY = 0.0  # No skipping for complete coverage
FOREGROUND_THRESHOLD = 0.08  # Sensitivity to foreground detection
POSITION_RANDOMNESS = 0.05  # Very low randomness for precision
DETAIL_SENSITIVITY = 1.0  # Maximum detail sensitivity
USE_VARIABLE_SIZES = True  # Use variable thumbnail sizes for better detail
EDGE_ALIGNMENT = True  # Align thumbnails with edges for better detail definition
MIN_THUMBNAIL_SCALE = 0.1  # Very small thumbnails for fine details
MAX_THUMBNAIL_SCALE = 1.0  # Maximum scale factor for thumbnails

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
    
    Args:
        target_color: Target RGB color to match
        thumbnails: List of thumbnail objects
        cell_detail: Detail level of the cell (0-1)
    
    Returns:
        Best matching thumbnail
    """
    min_distance = float('inf')
    best_match = None
    
    target_color = np.array(target_color).astype(float)
    
    # Convert to LAB color space if scikit-image is available
    if SKIMAGE_AVAILABLE:
        try:
            # Reshape and normalize to 0-1 range for skimage
            target_rgb = target_color.reshape(1, 1, 3) / 255.0
            # Convert to LAB color space
            target_lab = color.rgb2lab(target_rgb).reshape(3)
        except Exception as e:
            print(f"Error converting to LAB color space: {e}")
            # Fallback to RGB with weighted components
            target_lab = None
    else:
        target_lab = None
    
    # Weight candidates differently based on detail level
    # Higher detail cells need more accurate color matching
    color_weight = 0.7 + 0.3 * cell_detail  # 0.7-1.0 based on detail
    texture_weight = 1.0 - color_weight
    
    candidate_thumbnails = []
    
    # First pass: calculate distances and keep top candidates
    for thumbnail in thumbnails:
        if target_lab is not None:
            # Use LAB color space for perceptual accuracy
            thumbnail_rgb = thumbnail['avg_color'].reshape(1, 1, 3) / 255.0
            thumbnail_lab = color.rgb2lab(thumbnail_rgb).reshape(3)
            
            # LAB distance (DE76)
            distance = np.sum((target_lab - thumbnail_lab) ** 2)
        else:
            # Fallback to weighted RGB if LAB is not available
            # Green is more perceptually important, followed by red, then blue
            weights = np.array([0.3, 0.6, 0.1])  # RGB weights
            
            # Calculate weighted Euclidean distance
            weighted_diff = (target_color - thumbnail['avg_color']) * weights
            distance = np.sum(weighted_diff**2)
        
        # Add candidates with distance and thumbnail
        candidate_thumbnails.append((distance, thumbnail))
    
    # Sort by distance and pick the best match
    if candidate_thumbnails:
        # Sort by distance
        candidate_thumbnails.sort(key=lambda x: x[0])
        # Choose from the top candidates (introducing slight randomness for variety)
        top_n = max(1, min(5, int(len(candidate_thumbnails) * 0.05)))
        if top_n > 1 and random.random() < 0.3:  # 30% chance to pick a non-optimal match for variety
            best_match = random.choice(candidate_thumbnails[:top_n])[1]
        else:
            best_match = candidate_thumbnails[0][1]
    
    return best_match

def enhance_image(img, contrast_factor=CONTRAST_FACTOR):
    """Minimal image enhancement focusing mainly on edge detection.
    
    This version is designed to be very gentle on colors while still
    providing enough edge information for detail detection.
    """
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create a copy for edge detection only - we won't modify the original colors
    edge_img = img.copy()
    
    # Apply a VERY mild sharpening for edge detection
    # This won't affect the color matching, just helps with detail recognition
    edge_img = edge_img.filter(ImageFilter.SHARPEN)
    
    # Skip the adaptive histogram equalization - it can cause color shifts
    
    # Skip contrast enhancement unless explicitly requested
    if contrast_factor > 1.0:
        enhancer = ImageEnhance.Contrast(edge_img)
        edge_img = enhancer.enhance(contrast_factor)
    
    # Skip brightness enhancement entirely
    
    return edge_img

def fetch_thumbnails_from_s3(limit=THUMBNAIL_LIMIT, prefix="selected-images/"):
    """Fetch thumbnail images from S3 bucket and process for mosaic use."""
    thumbnails = []
    
    try:
        print(f"Fetching up to {limit} thumbnails from S3...")
        
        # Create a paginator for listing objects
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # We'll count how many thumbnails we've processed
        count = 0
        
        # Store all keys first so we can shuffle them for more variety
        all_image_keys = []
        
        # Paginate through results to collect all keys
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                # Skip if not an image file
                key = obj['Key']
                if not key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                all_image_keys.append(key)
        
        print(f"Found {len(all_image_keys)} potential thumbnail images")
        
        # Shuffle to ensure variety - important for photorealistic result
        random.shuffle(all_image_keys)
        
        # Limit to the requested number
        selected_keys = all_image_keys[:limit]
        
        # Process selected keys
        for key in selected_keys:
            try:
                response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                image_data = response['Body'].read()
                
                # Open image using PIL
                img = Image.open(BytesIO(image_data))
                
                # Ensure the image is in RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create a high-resolution version for internal storage
                hi_res_img = img.resize(INTERNAL_THUMBNAIL_SIZE, Image.LANCZOS)
                
                # Create a preview version for color matching (smaller memory footprint)
                preview_img = img.resize(THUMBNAIL_SIZE, Image.BILINEAR)
                
                # Calculate average color from the preview (faster than the hi-res)
                img_array = np.array(preview_img)
                avg_color = np.mean(img_array, axis=(0, 1)).astype(int)
                
                # Calculate color variance for texture assessment
                color_variance = np.var(img_array, axis=(0, 1)).mean()
                
                # Add to our thumbnails list with extended metadata
                thumbnails.append({
                    's3_key': key,
                    'avg_color': avg_color,
                    'image': hi_res_img,  # Store the high-res version
                    'display_size': THUMBNAIL_SIZE,  # Remember the display size
                    'color_variance': color_variance  # Higher values = more texture
                })
                
                # Clean up the preview to save memory
                del preview_img
                
                count += 1
                
            except Exception as e:
                print(f"Error processing S3 image {key}: {e}")
        
        print(f"Successfully processed {len(thumbnails)} thumbnails")
        
        # Sort thumbnails by color properties to help with matching
        if SKIMAGE_AVAILABLE:
            try:
                # Convert RGB to HSV for better sorting
                for thumbnail in thumbnails:
                    rgb = thumbnail['avg_color'].reshape(1, 1, 3) / 255.0
                    hsv = color.rgb2hsv(rgb)[0, 0]
                    thumbnail['hsv'] = hsv
                
                # Sort first by hue, then by saturation for better distribution
                thumbnails.sort(key=lambda x: (x['hsv'][0], x['hsv'][1]))
            except Exception as e:
                print(f"HSV sorting failed: {e}")
                # Fallback to RGB sorting
                thumbnails.sort(key=lambda x: sum(x['avg_color']))
        else:
            # Fallback for no skimage - sort by brightness
            thumbnails.sort(key=lambda x: sum(x['avg_color']))
        
        print(f"Thumbnails sorted by color properties for optimal matching")
        
    except Exception as e:
        print(f"Error fetching thumbnails from S3: {e}")
        traceback.print_exc()
        
    return thumbnails

def create_mosaic(img, thumbnails):
    """Create a photorealistic mosaic of the input image using the provided thumbnails."""
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
        
        # Calculate cell grid - use smaller cells for photorealistic result
        # For truly photorealistic results, we want very small cells (1x1 or 2x2)
        n_cols = math.floor(img.width / CELL_SIZE[0])
        n_rows = math.floor(img.height / CELL_SIZE[1])
        
        # Ensure a reasonable number of cells
        max_cells = 10000  # Increased for higher density
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
        
        # Use enhanced image for edge detection
        enhanced_gray = enhanced_img.convert('L')
        edges = enhanced_gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy arrays for processing
        enhanced_array = np.array(enhanced_gray)
        edges_array = np.array(edges)
        original_array = np.array(original_img)
        
        # Generate a detail map for adaptive thumbnail placement
        detail_map = create_detail_map(enhanced_array, edges_array)
        
        # Create a foreground mask using multiple techniques
        
        # 1. Use Otsu's method for automatic thresholding on the grayscale image
        try:
            if threshold_otsu:
                threshold = threshold_otsu(enhanced_array)
            else:
                # Fallback if skimage is not available
                threshold = np.mean(enhanced_array) * 0.8
        except:
            # Fallback if method fails
            threshold = np.mean(enhanced_array) * 0.8
        
        # 2. Create the initial mask - invert so darker areas (typically subject) are True
        foreground_mask1 = enhanced_array < threshold
        
        # 3. Use edge detection to enhance the mask
        edge_threshold = np.max(edges_array) * 0.2  # Lower threshold to catch more edges
        edge_mask = edges_array > edge_threshold
        
        # 4. Combine masks with logical OR to get a more comprehensive foreground mask
        combined_mask = np.logical_or(foreground_mask1, edge_mask)
        
        # 5. Apply morphological operations to clean up the mask
        # First close small holes
        foreground_mask = ndimage.binary_closing(combined_mask, structure=np.ones((7, 7)))
        # Then remove small isolated regions
        foreground_mask = ndimage.binary_opening(foreground_mask, structure=np.ones((3, 3)))
        # Expand the mask slightly to cover the edges better
        foreground_mask = ndimage.binary_dilation(foreground_mask, structure=np.ones((9, 9)))
        
        # 6. Use connected component analysis to keep only the largest blob (likely the person)
        # Label connected components
        labeled_mask, num_features = ndimage.label(foreground_mask)
        if num_features > 1:
            # Find the largest component (likely the person)
            component_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background (label 0)
            largest_component = np.argmax(component_sizes) + 1  # +1 because labels start at 1
            # Keep only the largest component
            foreground_mask = labeled_mask == largest_component
            # Dilate again for smoother edges
            foreground_mask = ndimage.binary_dilation(foreground_mask, structure=np.ones((5, 5)))
        
        print(f"Created enhanced foreground mask for subject detection with {num_features} features detected")
        
        # Create a specific background mask with a buffer zone
        background_mask = create_background_mask(foreground_mask)
        
        # Use Canny edge detection for more precise edge orientation if skimage is available
        canny_edges = None
        if SKIMAGE_AVAILABLE and EDGE_ALIGNMENT:
            try:
                canny_edges = feature.canny(enhanced_array, sigma=2.0)
                print("Applied Canny edge detection for edge-aware thumbnail placement")
            except Exception as e:
                print(f"Canny edge detection failed: {e}")
        
        # Determine dominant color of background for themed thumbnails
        # Calculate average color in background regions
        background_pixels = original_array[background_mask]
        if len(background_pixels) > 0:
            avg_background_color = np.mean(background_pixels, axis=0).astype(int)
            # Create a subset of thumbnails that match the background color theme
            print(f"Creating background color theme with average color: {avg_background_color}")
            background_thumbnails = filter_thumbnails_by_color(thumbnails, avg_background_color)
            print(f"Selected {len(background_thumbnails)} thumbnails for background areas")
        else:
            background_thumbnails = thumbnails
        
        # Process grid by extracting cell average colors first - use ORIGINAL image for colors
        cell_colors = []
        cell_is_foreground = []
        cell_is_background = []
        cell_detail_levels = []
        
        for y in range(n_rows):
            row_colors = []
            row_foreground = []
            row_background = []
            row_details = []
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
                
                # Check if this cell is part of the background
                cell_bg_mask = background_mask[cell_y:cell_y + CELL_SIZE[1], cell_x:cell_x + CELL_SIZE[0]]
                is_background = np.mean(cell_bg_mask) > 0.5  # More than half background
                row_background.append(is_background)
                
                # Calculate detail level for this cell
                detail_level = calculate_detail_level(cell, edges_array, cell_x, cell_y, CELL_SIZE[0], CELL_SIZE[1])
                detail_level = min(1.0, detail_level * DETAIL_SENSITIVITY * 2.0)  # Amplify and clamp
                row_details.append(detail_level)
                
                del cell
            cell_colors.append(row_colors)
            cell_is_foreground.append(row_foreground)
            cell_is_background.append(row_background)
            cell_detail_levels.append(row_details)
        
        # Release the source images from memory
        del enhanced_img
        del original_img
        del enhanced_gray
        del edges
        gc.collect()

        # Process each cell to place high-resolution thumbnails
        # For photorealistic results, we need to fill EVERY cell
        for y in range(n_rows):
            for x in range(n_cols):
                # Get the detail level for this cell
                detail_level = cell_detail_levels[y][x]
                
                # Get the average color
                avg_color = cell_colors[y][x]
                
                # Choose thumbnail set based on cell location
                if cell_is_foreground[y][x]:
                    # Use all thumbnails for foreground to get best color match
                    current_thumbnails = thumbnails
                    # Higher detail level for important areas
                    detail_level = max(detail_level, 0.4)
                elif cell_is_background[y][x]:
                    # Use themed thumbnails for background
                    current_thumbnails = background_thumbnails
                else:
                    # Buffer zone - use all thumbnails
                    current_thumbnails = thumbnails
                
                # Find the best match with more weight on color accuracy in detailed areas
                best_match = find_best_match(avg_color, current_thumbnails, detail_level)
                if not best_match:
                    continue
                    
                # Calculate the high-resolution position
                hi_res_x = int(x * CELL_SIZE[0] * scale_factor)
                hi_res_y = int(y * CELL_SIZE[1] * scale_factor)
                
                # For photorealistic results, minimize randomness in positioning
                # Only apply minimal randomness in less detailed areas
                if detail_level < 0.3:
                    # Calculate the maximum random offset - very small!
                    max_offset_x = int(CELL_SIZE[0] * scale_factor * POSITION_RANDOMNESS * 0.2)
                    max_offset_y = int(CELL_SIZE[1] * scale_factor * POSITION_RANDOMNESS * 0.2)
                    
                    # Generate random offsets
                    rand_offset_x = random.randint(-max_offset_x, max_offset_x)
                    rand_offset_y = random.randint(-max_offset_y, max_offset_y)
                    
                    # Apply the random offsets
                    hi_res_x += rand_offset_x
                    hi_res_y += rand_offset_y

                # Ensure the thumbnail stays within the mosaic bounds
                hi_res_x = max(0, min(hi_res_x, hi_res_width - INTERNAL_THUMBNAIL_SIZE[0]))
                hi_res_y = max(0, min(hi_res_y, hi_res_height - INTERNAL_THUMBNAIL_SIZE[1]))
                
                # Determine thumbnail size based on detail level if enabled
                if USE_VARIABLE_SIZES:
                    # Areas with high detail get smaller thumbnails for better definition
                    # For photorealistic results, we want very small thumbnails in detailed areas
                    thumbnail_scale = max(MIN_THUMBNAIL_SCALE, 1.0 - (detail_level * 0.7))
                    
                    # Calculate new dimensions - ensure we're covering the cell completely
                    new_width = int(INTERNAL_THUMBNAIL_SIZE[0] * thumbnail_scale)
                    new_height = int(INTERNAL_THUMBNAIL_SIZE[1] * thumbnail_scale)
                    
                    # Resize the thumbnail
                    thumbnail_to_paste = best_match['image'].resize((new_width, new_height), Image.LANCZOS)
                else:
                    thumbnail_to_paste = best_match['image']
                
                # Apply rotation if edge alignment is enabled and we have edge data
                # But only in detailed areas - this increases photorealism
                if EDGE_ALIGNMENT and canny_edges is not None and detail_level > 0.5:
                    try:
                        # Get cell center in original image coordinates
                        center_x = x * CELL_SIZE[0] + CELL_SIZE[0] // 2
                        center_y = y * CELL_SIZE[1] + CELL_SIZE[1] // 2
                        
                        # Get edge orientation at this point
                        rotation_angle = get_edge_orientation(canny_edges, center_x, center_y)
                        
                        # Apply rotation if we got a valid angle
                        if rotation_angle != 0:
                            thumbnail_to_paste = thumbnail_to_paste.rotate(rotation_angle, resample=Image.BILINEAR, expand=False)
                    except Exception as e:
                        pass
                
                # Place the high-resolution thumbnail
                mosaic.paste(thumbnail_to_paste, (hi_res_x, hi_res_y))
                filled_cells += 1
                
                # Periodic status updates
                if filled_cells % 500 == 0:
                    print(f"Placed {filled_cells} thumbnails so far...")
        
        print(f"High-resolution mosaic created with {filled_cells}/{total_cells} cells filled")
        print(f"Final mosaic dimensions before rotation: {hi_res_width}x{hi_res_height} pixels")

        # rotate if needed
        if mosaic.width > mosaic.height:
            rotated_mosaic = mosaic.transpose(Image.ROTATE_270)
        else:
            rotated_mosaic = mosaic
        
        print(f"Rotated mosaic to portrait orientation: {rotated_mosaic.width}x{rotated_mosaic.height} pixels")
        
        # For photorealistic output, we want to keep the full resolution
        # Skip the background canvas and just return the mosaic directly
        return rotated_mosaic

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

def create_background_mask(foreground_mask, dilate_size=20):
    """Creates a specific mask for background areas with a buffer zone around the foreground"""
    # Create a copy of the foreground mask
    # Dilate it substantially to create a buffer zone
    buffer_mask = ndimage.binary_dilation(foreground_mask, structure=np.ones((dilate_size, dilate_size)))
    
    # Background is everything that's not in the buffer zone
    background_mask = ~buffer_mask
    
    return background_mask

def filter_thumbnails_by_color(thumbnails, target_color, threshold=60):
    """Filter thumbnails to match a specific color theme"""
    filtered = []
    target_color = np.array(target_color)
    
    for thumbnail in thumbnails:
        color_diff = np.sqrt(np.sum((thumbnail['avg_color'] - target_color)**2))
        if color_diff < threshold:
            filtered.append(thumbnail)
    
    # If we didn't get enough matches, return some portion of the originals
    if len(filtered) < len(thumbnails) * 0.1:
        # Sort by closest color match and take top 10%
        sorted_thumbnails = sorted(thumbnails, 
                                 key=lambda x: np.sqrt(np.sum((x['avg_color'] - target_color)**2)))
        filtered = sorted_thumbnails[:max(10, int(len(thumbnails) * 0.1))]
        
    return filtered

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
