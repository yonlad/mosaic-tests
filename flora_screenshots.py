import os
import boto3
import cv2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS S3 configuration
BUCKET_NAME = 'pistoletto.moe'
VIDEO_FOLDER = 'video-blends/'
SCREENSHOTS_FOLDER = 'flora_screenshots_output/'

# Create screenshots folder if it doesn't exist
if not os.path.exists(SCREENSHOTS_FOLDER):
    os.makedirs(SCREENSHOTS_FOLDER)

# Initialize S3 client
# Make sure you have your AWS credentials in a .env file:
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_SESSION_TOKEN=your_session_token (optional)
s3 = boto3.client('s3')

def get_video_files_from_s3(bucket, prefix):
    """List video files in a S3 folder."""
    video_files = []
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].lower().endswith(('.mp4')):
                video_files.append(obj['Key'])
    return video_files

def capture_first_frame(video_path, output_path):
    """Capture the first frame of a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"Successfully captured screenshot for {video_path} to {output_path}")
            cap.release()
            return True
        else:
            print(f"Error: Could not read frame from {video_path}")
            cap.release()
            return False
    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")
        return False

def main():
    """Main function to process videos from S3."""
    print("Starting video processing pipeline...")
    video_files = get_video_files_from_s3(BUCKET_NAME, VIDEO_FOLDER)
    
    if not video_files:
        print("No video files found in the specified S3 folder.")
        return

    for video_key in video_files:
        video_filename = os.path.basename(video_key)
        # Define a temporary path for the video file
        temp_video_path = os.path.join('temp_videos', video_filename)
        # Ensure the temporary directory exists
        os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
        screenshot_filename = os.path.splitext(video_filename)[0] + '.png'
        screenshot_path = os.path.join(SCREENSHOTS_FOLDER, screenshot_filename)

        print(f"Processing video: {video_key}")

        try:
            # Download video from S3 to a temporary location
            print(f"Downloading {video_key} to {temp_video_path}...")
            s3.download_file(BUCKET_NAME, video_key, temp_video_path)
            
            # Capture the first frame
            if capture_first_frame(temp_video_path, screenshot_path):
                 # Optional: Upload screenshot back to S3 if needed
                 # s3.upload_file(screenshot_path, BUCKET_NAME, f"screenshots/{screenshot_filename}")
                 pass

        except Exception as e:
            print(f"Failed to process {video_key}: {e}")
        finally:
            # Clean up the downloaded video file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print(f"Removed temporary file: {temp_video_path}")

    print("Pipeline finished.")

if __name__ == "__main__":
    main()
