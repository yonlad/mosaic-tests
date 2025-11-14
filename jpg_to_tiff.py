# convert a jpg image to a tiff image

import os
from PIL import Image

def jpg_to_tiff(jpg_path, tiff_path):
    img = Image.open(jpg_path)
    img.save(tiff_path, format="png")

jpg_to_tiff("highest_res_2px.jpg", "highest_res_2px.png")