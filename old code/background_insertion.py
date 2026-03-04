"""
A script to take images with transparent backgrounds from a directory,
place them on a randomly selected colored background, and save them
to a new directory.
"""
import os
import argparse
import random
from PIL import Image
import logging
from tqdm import tqdm
import distinctipy

# --- Configuration ---

"""
# Background Colors and their weights for randomization
# Format: { "color_name": ((R, G, B), weight) }
# The weights are relative and do not need to sum to 100.
BACKGROUND_COLORS = {
    "white": ((255, 255, 255), 30),
    "black": ((0, 0, 0), 30),
    "skin_color": ((244,227,214), 10),
    "fair_skin": ((245, 222, 179), 10),
    "light_medium_skin": ((222, 184, 135), 10),
    "pale_skin": ((255, 248, 220), 10),
    "walnut_skin": ((101, 55, 23), 10),
}

"""

# Generate 100 visually distinct colors
# The colors are generated as RGB tuples with values between 0 and 255.
BACKGROUND_COLORS = [
    # tuple(int(c * 255) for c in color) for color in distinctipy.get_colors(100)
    (0, 0, 0),
]

# Default directories
DEFAULT_INPUT_DIR = "bg_removed"
DEFAULT_OUTPUT_DIR = "black_background"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Image Processing Functions ---

def place_on_background(image_with_alpha, color):
    """
    Places a PIL image with an alpha channel onto a solid colored background.
    """
    if image_with_alpha.mode not in ('RGBA', 'LA'):
        logger.error(f"Image must have an alpha channel to be placed on a background. Got {image_with_alpha.mode}.")
        return None
        
    background = Image.new("RGB", image_with_alpha.size, color)
    # The RGBA image is pasted onto the RGB background using its own alpha channel as the mask
    background.paste(image_with_alpha, (0, 0), image_with_alpha)
    return background

def choose_background_color():
    """
    Chooses a random background color from the generated list.
    """
    if not BACKGROUND_COLORS:
        logger.warning("BACKGROUND_COLORS list is empty. Defaulting to white background.")
        return (255, 255, 255)
    
    return random.choice(BACKGROUND_COLORS)

# --- Main Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add colored backgrounds to images with transparency.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help=f"Directory for images with removed backgrounds. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory for final images with new backgrounds. Default: {DEFAULT_OUTPUT_DIR}")
    
    args = parser.parse_args()

    # --- Setup ---
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}. Please create it or specify a different directory.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Reading images from: {args.input_dir}")
    logger.info(f"Saving final images to: {args.output_dir}")

    # --- Execution ---
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.png')]
    
    if not image_files:
        logger.warning(f"No .png images found in {args.input_dir}. Nothing to process.")
        exit(0)

    logger.info(f"Found {len(image_files)} images to process.")

    for filename in tqdm(image_files, desc="Applying backgrounds"):
        try:
            input_path = os.path.join(args.input_dir, filename)
            image_with_alpha = Image.open(input_path)

            color = choose_background_color()
            final_image = place_on_background(image_with_alpha, color)
            
            if not final_image:
                logger.warning(f"Skipping {filename} due to a processing error.")
                continue

            # Create the new filename
            name, _ = os.path.splitext(filename)
            # Ensure the output name is clean, removing suffixes like '_bg_removed'
            if name.endswith('_bg_removed'):
                name = name[:-11]
            output_filename = f"{name}_final.jpg"
            output_path = os.path.join(args.output_dir, output_filename)

            final_image.save(output_path, "JPEG", quality=95)

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    logger.info("Script finished.")
