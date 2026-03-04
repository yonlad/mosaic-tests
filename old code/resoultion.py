from PIL import Image
Image.MAX_IMAGE_PIXELS = None

result = Image.open("latest-yonatan (1).png")  # or whatever your output filename is
print(f'Output dimensions: {result.size}')