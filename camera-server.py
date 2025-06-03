from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import gphoto2 as gp
import os
from datetime import datetime
import time
import traceback
import base64
import boto3
from dotenv import load_dotenv
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import math
import random
from io import BytesIO
import gc
from skimage.filters import threshold_otsu
from scipy import ndimage
import threading
from contextlib import contextmanager
import atexit
import re
import logging
import uuid
import requests
import json
from gradio_client import Client, handle_file

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv('REACT_APP_AWS_REGION', 'us-east-2')
AWS_ACCESS_KEY_ID = os.getenv('REACT_APP_AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('REACT_APP_AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('REACT_APP_S3_BUCKET', 'eternity-mirror-project')
DYNAMODB_TABLE = os.getenv('REACT_APP_DYNAMODB_TABLE', 'eternity-mirror-users')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Initialize DynamoDB client
dynamodb_client = boto3.client(
    'dynamodb',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

app = Flask(__name__)

# Apply CORS with more detailed configuration
CORS(app, 
     origins=["*", "http://localhost:3000", "http://localhost:3001"], 
     supports_credentials=True,
     # Allow all headers
     allow_headers="*",
     # Add methods that might be used for complex operations
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     # Extend max age for preflight requests to reduce frequency of OPTIONS calls
     max_age=3600)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

MOCK_MODE = False  # Set to False when you have a real camera connected

# Mosaic configuration
THUMBNAIL_SIZE = (2, 3)  # Size for display/layout (keep this the same)
INTERNAL_THUMBNAIL_SIZE = (250, 250)  # 4x higher resolution for zoom detail
CELL_SIZE = (3, 3)  # Spacing between thumbnails (keep this the same) (controls density of thumbnails)
CONTRAST_FACTOR = 2  # Increased contrast enhancement for better edge detection
BRIGHTNESS_THRESHOLD = 130  # Threshold for skipping very bright areas
THUMBNAIL_LIMIT = 4000  # Fewer high-resolution thumbnails to manage memory
SKIP_PROBABILITY = 0.05  # Keep the same skip probability (the lower the number, the more thumbnails used in mosaic)
FOREGROUND_THRESHOLD = 0.15  # Lower threshold to be more sensitive to foreground detection
POSITOIN_RANDOMNESS = 0.3  # Randomness in position of the foreground object

# Add initialization state tracking at the server level
initialization_in_progress = False
last_init_attempt = 0

# Add persistent USB management variables
usb_reset_lock = threading.RLock()  # Lock for USB reset operations
last_usb_reset_time = 0  # Track when we last did a USB reset
usb_error_count = 0  # Track USB errors to detect when we need deeper cleanup

# Add a background thread to periodically check and clean up USB resources
usb_cleanup_thread = None
stop_cleanup_thread = False

def periodic_usb_cleanup():
    """Background thread function to periodically check and clean up USB resources."""
    global stop_cleanup_thread, usb_error_count
    
    cleanup_interval = 300  # seconds (5 minutes)
    force_cleanup_interval = 1800  # seconds (30 minutes)
    last_force_cleanup = time.time()
    
    print("Starting periodic USB cleanup thread")
    
    try:
        while not stop_cleanup_thread:
            current_time = time.time()
            
            # Check if we need to do a cleanup based on error count
            if usb_error_count >= 3:
                print("USB error count threshold reached, performing scheduled cleanup")
                cleanup_usb_resources()
            
            # Force cleanup periodically regardless of error count
            if current_time - last_force_cleanup >= force_cleanup_interval:
                print("Performing periodic forced USB cleanup")
                cleanup_usb_resources()
                last_force_cleanup = current_time
                
            # Sleep for a bit before checking again (check every 30 seconds)
            for _ in range(30):  # 30 seconds in 1-second increments for responsive shutdown
                if stop_cleanup_thread:
                    break
                time.sleep(1)
    except Exception as e:
        print(f"Error in USB cleanup thread: {e}")
    finally:
        print("USB cleanup thread stopped")

def start_usb_cleanup_thread():
    """Start the USB cleanup background thread."""
    global usb_cleanup_thread, stop_cleanup_thread
    
    if usb_cleanup_thread is None or not usb_cleanup_thread.is_alive():
        stop_cleanup_thread = False
        usb_cleanup_thread = threading.Thread(target=periodic_usb_cleanup)
        usb_cleanup_thread.daemon = True  # Allow the thread to be terminated when the main program exits
        usb_cleanup_thread.start()
        print("USB cleanup thread started")

def stop_usb_cleanup_thread():
    """Stop the USB cleanup background thread."""
    global stop_cleanup_thread
    
    if usb_cleanup_thread is not None and usb_cleanup_thread.is_alive():
        stop_cleanup_thread = True
        print("Signaling USB cleanup thread to stop")

@contextmanager
def usb_operation_context():
    """Context manager for USB operations to ensure proper cleanup
    even when operations fail with exceptions."""
    global usb_error_count
    
    try:
        yield  # Execute the wrapped code
    except Exception as e:
        error_str = str(e).lower()
        # Check if this is a USB-related error
        if any(term in error_str for term in ["usb", "claim", "device", "i/o", "busy", "communication"]):
            with usb_reset_lock:
                usb_error_count += 1
                print(f"USB error detected, incrementing count to {usb_error_count}")
        raise  # Re-raise the exception
    finally:
        # Force garbage collection after USB operations
        gc.collect()

def cleanup_usb_resources():
    """Perform a thorough cleanup of USB resources"""
    global usb_error_count
    
    print("Performing thorough USB resource cleanup...")
    
    try:
        # Reset Canon USB devices
        import usb.core
        import usb.util
        
        # Canon's vendor ID
        CANON_VENDOR_ID = 0x04a9
        
        # Find all Canon devices
        devices = list(usb.core.find(find_all=True, idVendor=CANON_VENDOR_ID))
        print(f"Found {len(devices)} Canon USB devices")
        
        for dev in devices:
            try:
                print(f"Resetting Canon USB device {dev.idProduct:04x}")
                dev.reset()
            except Exception as e:
                print(f"Error resetting USB device: {e}")
        
        # Force USB re-enumeration
        print("Forcing USB re-enumeration...")
        all_devices = list(usb.core.find(find_all=True))
        print(f"Found {len(all_devices)} total USB devices")
        
        # Force garbage collection to clean up any lingering references
        print("Forcing garbage collection for USB resource cleanup")
        gc.collect()
        
        # Reset the error counter
        usb_error_count = 0
        
        return True
    except Exception as e:
        print(f"Error during USB cleanup: {e}")
        return False

def terminate_macos_camera_services():
    """Attempt to terminate macOS services that might interfere with the camera"""
    try:
        print("Attempting to terminate macOS camera services...")
        
        # More aggressive approach - kill ALL potential camera interfering processes
        # List of known macOS camera services
        camera_services = [
            "ptpcamera",
            "mscamerad",
            "VDCAssistant",
            "AppleCameraAssistant",
            "USBAgent",
            "gphoto2",  # Kill any other gphoto2 processes
            "PTPCamera", # Alternate name sometimes used
            "imagecaptureext", # Image Capture Extension
            "Image Capture Extension" # Full name version
        ]
        
        # Try to terminate each service, first without sudo
        killed_count = 0
        for service in camera_services:
            try:
                # First try without sudo (this will work for user processes)
                print(f"Attempting to kill {service}...")
                result = os.system(f"killall -9 {service} 2>/dev/null")
                if result == 0:
                    print(f"Successfully terminated {service}")
                    killed_count += 1
                else:
                    print(f"Could not terminate {service} (may not be running)")
            except Exception as e:
                print(f"Error terminating {service}: {e}")
        
        # Wait a moment for services to terminate
        time.sleep(2)
        
        # Force USB re-enumeration more aggressively
        try:
            print("Forcing aggressive USB cleanup...")
            import usb.core
            import usb.util
            
            # Get all USB devices
            all_devices = list(usb.core.find(find_all=True))
            print(f"Found {len(all_devices)} USB devices")
            
            # Disconnect all camera-related USB devices
            for dev in all_devices:
                try:
                    # Check if this might be a Canon device or a PTP/camera device
                    if (hasattr(dev, 'idVendor') and 
                        (dev.idVendor == 0x04a9 or  # Canon
                         (dev.bDeviceClass == 6) or  # Imaging class
                         ("PTP" in str(dev)) or      # PTP in device info
                         ("Camera" in str(dev)))):   # Camera in device info
                        
                        print(f"Resetting USB device: {dev.idVendor:04x}:{dev.idProduct:04x}")
                        try:
                            # Try to reset it
                            dev.reset()
                            print("USB reset successful")
                        except:
                            # If reset fails, try detaching kernel driver
                            print("Reset failed, trying to detach kernel driver...")
                            try:
                                if dev.is_kernel_driver_active(0):
                                    dev.detach_kernel_driver(0)
                                    print("Detached kernel driver")
                            except:
                                pass
                        
                        # Always release the device
                        usb.util.dispose_resources(dev)
                        print("Released device resources")
                except Exception as e:
                    print(f"Error handling device: {e}")
        except Exception as e:
            print(f"Error during USB cleanup: {e}")
        
        # Clean up USB resources
        cleanup_usb_resources()
        
        # Reset global camera variables to force complete reinitialization
        global camera_control, initialization_in_progress
        if camera_control:
            try:
                camera_control.cleanup()
            except:
                pass
            camera_control = None
        initialization_in_progress = False
        
        # Force garbage collection
        print("Forcing garbage collection...")
        gc.collect()
        time.sleep(1)
        gc.collect()
        
        print(f"Terminated {killed_count} camera-related services")
        return True
    except Exception as e:
        print(f"Error terminating camera services: {e}")
        return False

class CameraControl:
    def __init__(self, skip_init=False):
        self.camera = None
        self.is_ready = False
        
        # Don't automatically initialize - let the route handle this
        # if not skip_init:
        #     self.initialize_camera()
    
    def initialize_camera(self):
        """
        Initialize the camera with multiple fallback approaches and error handling.
        Returns a dictionary with status, message, and camera_info if successful.
        """
        global usb_error_count
        
        # Set a maximum number of retries for auto-detection to prevent infinite loops
        MAX_DETECTION_RETRIES = 3
        detection_attempts = 0
        
        # Set a timeout for the entire initialization process
        init_start_time = time.time()
        MAX_INIT_TIME = 30  # seconds
        
        try:
            print("Initializing camera...")
            
            # First check if camera is already initialized and ready
            if self.camera and self.is_ready:
                print("Camera is already initialized and ready")
                return {
                    "status": "success",
                    "message": "Camera already initialized",
                    "camera_info": {"model": "Already initialized camera"}
                }
                
            # Clean up any existing camera instance before trying to initialize
            self.cleanup()
            
            # If we've accumulated USB errors, do a cleanup first
            if usb_error_count >= 3:
                print(f"USB error threshold reached ({usb_error_count}), performing cleanup before initialization")
                cleanup_usb_resources()
                # Reset the counter - we'll see if we get more errors during this attempt
                usb_error_count = 0
            
            # Current initialization state
            self.is_ready = False
            
            # Reset error counters on new initialization
            self.consecutive_errors = 0
            
            # Use our USB operation context for safer handling
            with usb_operation_context():
                # Try the most basic initialization first - this bypasses all abilities setup
                try:
                    print("Attempting basic initialization without auto-detect or abilities setup...")
                    self.camera = gp.Camera()
                    
                    # Create a context object
                    context = gp.Context()
                    
                    # Directly init without setting abilities
                    print("Initializing with basic approach...")
                    self.camera.init(context)
                    print("Camera initialized successfully with basic approach!")
                    
                    # Test if the camera is working
                    try:
                        print("Testing camera functionality...")
                        abilities = self.camera.get_abilities()
                        print(f"Camera model: {abilities.model}")
                        
                        # Try to get camera config
                        print("Getting camera config...")
                        config = self.camera.get_config()
                        print("Camera config retrieved successfully")
                        
                        # If we got here, mark camera as ready
                        self.is_ready = True
                        # Reset USB error count on success
                        usb_error_count = 0
                        return {
                            "status": "success",
                            "message": "Camera initialized successfully with basic approach",
                            "camera_info": {"model": abilities.model}
                        }
                    except Exception as test_error:
                        print(f"Camera initialized but functionality test failed: {test_error}")
                        # Continue to more complex initialization approaches
                except gp.GPhoto2Error as basic_error:
                    error_msg = str(basic_error).lower()
                    
                    # Track USB errors
                    if "usb" in error_msg or "claim" in error_msg or "device" in error_msg:
                        with usb_reset_lock:
                            usb_error_count += 1
                            print(f"USB error during basic init, incrementing count to {usb_error_count}")
                            
                        # If we've accumulated enough USB errors, do a cleanup
                        if usb_error_count >= 3:
                            print("USB error threshold reached, performing cleanup")
                            cleanup_usb_resources()
                        
                    if "Could not claim the USB device" in str(basic_error):
                        print("USB device is locked - cannot initialize")
                        self.camera = None
                        self.is_ready = False
                        raise Exception("Camera is locked by another process. Close any other apps using the camera or restart the computer.")
                    else:
                        print(f"Basic initialization failed: {basic_error}")
                        # Continue to the full auto-detection approach
                
                # If we reach here, basic initialization failed or didn't fully work
                # Try our original auto-detection approach as a fallback
                print("Basic initialization didn't succeed, trying full detection approach...")
                
                # If we've accumulated USB errors, try a cleanup first
                if usb_error_count >= 2:
                    print("Multiple USB errors detected, attempting cleanup before continuing")
                    cleanup_usb_resources()
                
                # First try auto-detection with timeout and error handling
                try:
                    print("Attempting auto-detection...")
                    # Create camera and context objects
                    context = gp.Context()
                    self.camera = gp.Camera()
                    
                    # Try to auto-detect with a timeout
                    detect_timeout = 8  # seconds
                    start_time = time.time()
                    camera_found = False
                    consecutive_errors = 0
                    MAX_CONSECUTIVE_ERRORS = 3  # After this many errors, skip to next approach
                    
                    while (time.time() - start_time < detect_timeout and 
                           not camera_found and 
                           detection_attempts < MAX_DETECTION_RETRIES and
                           consecutive_errors < MAX_CONSECUTIVE_ERRORS):
                        try:
                            detection_attempts += 1
                            print(f"Detection attempt {detection_attempts}/{MAX_DETECTION_RETRIES}...")
                            
                            # Check if we've exceeded the overall initialization time limit
                            if time.time() - init_start_time > MAX_INIT_TIME:
                                print("Initialization time limit exceeded, aborting")
                                break
                            
                            print("Loading port info and abilities lists...")
                            port_info_list = gp.PortInfoList()
                            port_info_list.load()
                            abilities_list = gp.CameraAbilitiesList()
                            abilities_list.load(context)
                            
                            print("Detecting cameras...")
                            cameras = abilities_list.detect(port_info_list, context)
                            
                            if cameras:
                                # Found at least one camera
                                camera_model = cameras[0][0]
                                port_path = cameras[0][1]
                                print(f"Detected camera: {camera_model} at {port_path}")
                                
                                try:
                                    # Get port index
                                    idx = port_info_list.lookup_path(port_path)
                                    if idx < 0:
                                        print(f"Warning: Invalid port index {idx} for {port_path}")
                                        raise ValueError(f"Invalid port index for {port_path}")
                                    
                                    # Set camera port info
                                    print(f"Setting port info at index {idx}")
                                    port_info = port_info_list[idx]
                                    self.camera.set_port_info(port_info)
                                    
                                    # Skip abilities setup since that's what's failing
                                    print("Skipping problematic abilities setup...")
                                    
                                    # Now init with the selected camera
                                    print("Initializing camera with context...")
                                    self.camera.init(context)
                                    print("Camera auto-detected and initialized successfully!")
                                    camera_found = True
                                    
                                    # Reset consecutive errors on success
                                    consecutive_errors = 0
                                except Exception as e:
                                    print(f"Error during camera port setup: {e}")
                                    # Increment consecutive errors
                                    consecutive_errors += 1
                                    
                                    # Track USB errors
                                    error_str = str(e).lower()
                                    if "usb" in error_str or "claim" in error_str or "device" in error_str:
                                        with usb_reset_lock:
                                            usb_error_count += 1
                                            print(f"USB error during port setup, incrementing count to {usb_error_count}")
                                        
                                        # If we've hit the threshold, clean up now
                                        if usb_error_count >= 3:
                                            print("USB error threshold reached, breaking detection loop to try cleanup")
                                            break
                                    
                                    # Don't raise here, just try again in the loop or fall back to standard init
                            else:
                                print("No cameras detected in this attempt.")
                                consecutive_errors += 1
                                
                        except Exception as detect_error:
                            consecutive_errors += 1
                            print(f"Auto-detection error: {detect_error}")
                            
                            # Track USB-related errors
                            error_str = str(detect_error).lower()
                            if "usb" in error_str or "claim" in error_str or "device" in error_str:
                                with usb_reset_lock:
                                    usb_error_count += 1
                                    print(f"USB error during detection, incrementing count to {usb_error_count}")
                                
                                # If we've hit the threshold, try cleaning up
                                if usb_error_count >= 3:
                                    print("USB error threshold reached during detection, performing cleanup")
                                    cleanup_usb_resources()
                                
                            # Critical error handling - detect camera locking early
                            if "Could not claim the USB device" in str(detect_error):
                                print("USB device is locked during detection phase. Camera may be in use by another process.")
                                # Clear the camera object to prevent incorrect ready state
                                self.camera = None
                                self.is_ready = False
                                # Use Exception instead of gp.GPhoto2Error
                                raise Exception("Camera is locked by another process. Close any other apps using the camera and try again. Error: " + str(detect_error))
                            
                            print(f"Detection error: {detect_error}, retrying...")
                            if time.time() - start_time >= detect_timeout:
                                print("Detection timeout reached")
                                break
                            time.sleep(1)
                    
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        print(f"Exceeded maximum consecutive errors ({MAX_CONSECUTIVE_ERRORS}), skipping to next approach")
                    
                    # If we couldn't detect any cameras within the timeout or had errors, 
                    # fall back to standard initialization
                    if not camera_found:
                        print("Auto-detection failed or had errors, falling back to standard initialization...")
                        # Create a fresh camera object
                        self.camera = gp.Camera()
                        # Use standard initialization
                        print("Using standard initialization without abilities setup...")
                        self.camera.init(context)
                    
                except gp.GPhoto2Error as e:
                    # Track USB errors
                    error_msg = str(e).lower()
                    if "usb" in error_msg or "claim" in error_msg or "device" in error_msg:
                        with usb_reset_lock:
                            usb_error_count += 1
                            print(f"USB error during initialization, incrementing count to {usb_error_count}")
                    
                    # Enhanced handling for common USB errors
                    if "Could not claim the USB device" in str(e):
                        print("Camera is locked by another process - cannot initialize")
                        # Clear camera and set not ready
                        self.camera = None
                        self.is_ready = False
                        # Raise a more specific error that will be caught by the main try/except
                        raise Exception("Camera is locked by another process. Close any other apps using the camera or restart the computer.")
                    elif "I/O" in str(e) or "timeout" in str(e):
                        print(f"I/O or timeout error: {e}, camera may be sleeping")
                        # Try standard initialization
                        self.camera = gp.Camera()
                        # Create a context for this initialization
                        self.camera.init(context)
                    elif "set_abilities" in str(e) or "argument" in str(e).lower() or "type" in str(e).lower():
                        print(f"Error in camera abilities setup: {e}")
                        # This is the specific error we're seeing - try the simplest initialization
                        print("Using simplified initialization without abilities...")
                        self.camera = gp.Camera()
                        self.camera.init()
                    else:
                        # For other errors, try a standard initialization with retry
                        print(f"Initial camera init failed: {e}, trying standard initialization...")
                        self.camera = gp.Camera()
                        
                        # Simple retry logic with timeout
                        init_timeout = 10  # seconds
                        start_time = time.time()
                        while time.time() - start_time < init_timeout:
                            try:
                                self.camera.init()
                                print("Camera initialized successfully after retries!")
                                break
                            except gp.GPhoto2Error as retry_error:
                                # Track USB errors during retry
                                retry_error_msg = str(retry_error).lower()
                                if "usb" in retry_error_msg or "claim" in retry_error_msg or "device" in retry_error_msg:
                                    with usb_reset_lock:
                                        usb_error_count += 1
                                        print(f"USB error during retry, incrementing count to {usb_error_count}")
                                
                                # Enhanced handling for USB claim errors during retry
                                if "Could not claim the USB device" in str(retry_error):
                                    print("Camera is locked by another process during retry - cannot initialize")
                                    # Clear camera and set not ready
                                    self.camera = None
                                    self.is_ready = False
                                    raise Exception("Camera is locked by another process. Close any other apps using the camera or restart the computer.")
                                
                                if time.time() - start_time >= init_timeout:
                                    raise Exception(f"Camera initialization timeout. Error: {str(retry_error)}")
                                print(f"Retrying initialization... Error was: {str(retry_error)}")
                                time.sleep(1)
                except Exception as general_error:
                    print(f"General error during auto-detection: {general_error}")
                    # Try the most basic initialization as a last resort
                    try:
                        print("Falling back to the most basic camera initialization...")
                        self.camera = gp.Camera()
                        self.camera.init()
                    except Exception as e:
                        print(f"Basic initialization also failed: {e}")
                        raise
                
                # If we got here, try to test camera capabilities
                try:
                    if self.camera:  # Only try to get abilities if camera exists
                        print("Testing camera capabilities...")
                        try:
                            abilities = self.camera.get_abilities()
                            print(f"Camera model: {abilities.model}")
                        except gp.GPhoto2Error as model_error:
                            if "Unknown model" in str(model_error) or "-105" in str(model_error):
                                print(f"Warning: Unable to detect camera model properly: {model_error}")
                                print("Continuing despite model detection issue...")
                            else:
                                raise
                        
                        # Try to get camera config
                        print("Getting camera config...")
                        config = self.camera.get_config()
                        print("Camera config retrieved successfully")
                    else:
                        # This should not happen, but if it does, raise an error
                        raise Exception("Camera initialization error: Camera object is None after initialization")
                except gp.GPhoto2Error as e:
                    # If we can't get capabilities but "could not claim USB device", 
                    # the camera is likely in use by another process
                    if "Could not claim the USB device" in str(e):
                        print("Camera is locked by another process during capabilities check")
                        # Clear camera and set not ready
                        self.camera = None
                        self.is_ready = False
                        raise Exception("Camera is locked by another process. Close any other apps using the camera or restart the computer.")
                    else:
                        print(f"Camera initialized but capabilities check failed: {e}")
                        # Continue as camera might be in sleep mode, but don't mark as ready yet
                
                # At this point, we've successfully initialized or we have a camera in sleep mode
                if self.camera:
                    print("Camera setup complete, ready to capture!")
                    self.is_ready = True
                    # Reset USB error count on success
                    usb_error_count = 0
                    
                    # Try to get camera model for info
                    camera_model = "Unknown"
                    try:
                        abilities = self.camera.get_abilities()
                        camera_model = abilities.model
                    except:
                        pass
                        
                    return {
                        "status": "success",
                        "message": "Camera initialized successfully",
                        "camera_info": {"model": camera_model}
                    }
                else:
                    print("Camera initialization incomplete")
                    self.is_ready = False
                    return {
                        "status": "error",
                        "message": "Camera initialization incomplete - camera object is None"
                    }
                
        except Exception as e:
            print(f"Camera initialization failed: {str(e)}")
            print("Full traceback:")
            print(traceback.format_exc())
            
            # Track USB errors during exception handling
            error_msg = str(e).lower()
            if "usb" in error_msg or "claim" in error_msg or "device" in error_msg:
                with usb_reset_lock:
                    usb_error_count += 1
                    print(f"USB error in exception handler, incrementing count to {usb_error_count}")
            
            # Check for battery-related keywords in the error
            if any(term in str(e).lower() for term in ["battery", "power", "energy"]):
                print("Camera initialization may have failed due to low battery")
            
            # Check for abilities-related errors
            if "set_abilities" in str(e) or "argument" in str(e) or "type" in str(e):
                print("Camera abilities error detected - this may be fixed by reconnecting the camera")
            
            # Ensure we don't mark camera as ready if there's any error
            self.is_ready = False
            self.camera = None
            
            # Don't mark camera as ready if there's a USB claim error - propagate the error instead
            if "Could not claim the USB device" in str(e) or "locked" in str(e).lower():
                print("Camera is locked by another process")
                return {
                    "status": "error",
                    "message": "Camera is locked by another process. Close any other apps using the camera or restart the computer.",
                    "error_type": "usb_locked"
                }
                
            # Propagate abilities errors with clear message
            if "set_abilities" in str(e) or "argument" in str(e) or "type" in str(e):
                return {
                    "status": "error",
                    "message": f"Camera initialization error: Unable to set camera abilities correctly. Try disconnecting and reconnecting the camera.",
                    "error_type": "abilities_error"
                }
            
            # Return error for any other exceptions
            return {
                "status": "error",
                "message": f"Camera initialization failed: {str(e)}"
            }
    
    def check_camera_status(self):
        """Check if the camera is still responsive after a period of inactivity"""
        if not self.camera:
            print("No camera instance exists")
            return False
        
        try:
            print("Testing camera connection status...")
            
            # First check if the camera status is accessible
            try:
                # Try to get camera summary - a lightweight operation to check connectivity
                summary = self.camera.get_summary()
                print("Camera summary successful")
                return True
            except gp.GPhoto2Error as summary_error:
                print(f"Camera summary check failed: {summary_error}")
                error_message = str(summary_error)
                
                # Check if this seems like a sleep state
                if any(term in error_message.lower() for term in [
                    "i/o problem", "timeout", "communication", 
                    "error (-1)", "error (-53)", "error (-110)", "busy"
                ]):
                    print("Camera appears to be in sleep mode - needs physical interaction to wake up")
                    return False
                
                # Check if this is a disconnection issue
                if any(term in error_message.lower() for term in [
                    "no camera", "unknown model", "[-105]"
                ]):
                    print("Camera appears to be disconnected")
                    return False
                
                # For other errors, camera status is uncertain
                print("Camera exists but may be in an error state")
                return False
                
        except Exception as e:
            print(f"Unexpected error checking camera status: {str(e)}")
            return False
    
    def _init_capture(self):
        try:
            file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
            camera_file = self.camera.file_get(
                file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
            del camera_file
        except gp.GPhoto2Error:
            pass
    
    def upload_to_s3(file_bytes, s3_key, filename):
                            try:
                                # Upload the image to S3
                                s3_client.upload_fileobj(
                                    io.BytesIO(file_bytes),
                                    S3_BUCKET,
                                    s3_key,
                                    ExtraArgs={'ContentType': 'image/jpeg'}
                                )
                                print(f"Successfully uploaded image to S3: {s3_key}")
                                
                                # Return the S3 key and filename
                                # Reset USB error count on success
                                usb_error_count = 0
                                return {
                                    "filename": filename,
                                    "s3_key": s3_key
                                }
                            except Exception as s3_error:
                                print(f"Error uploading to S3: {s3_error}")
                                # If S3 upload fails, we still have the image data,
                                # so return it as base64 as a fallback
                                base64_data = base64.b64encode(file_bytes).decode('ascii')
                                # Reset USB error count on success
                                usb_error_count = 0
                                return {
                                    "filename": filename,
                                    "data": base64_data,
                                    "s3_error": str(s3_error)
                                }
    
    
    def capture_image(self, session_id=None):
        """
        Capture an image with improved USB error handling.
        """
        global usb_error_count
        
        # Check if we need USB cleanup based on error count
        if usb_error_count >= 3:
            print("USB error count high before capture, performing cleanup first")
            cleanup_usb_resources()
        
        try:
            print("\nCapturing image...")
            
            # Mock mode for testing without camera
            if MOCK_MODE:
                # Create a mock image (red square)
                # Create a 640x480 red image
                img = Image.fromarray(np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                buffer.seek(0)
                file_bytes = buffer.getvalue()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                
                # If session_id is provided, upload to S3
                if session_id:
                    s3_key = f"raw-captures/{session_id}/{filename}"
                    
                    print(f"Uploading mock image to S3: {s3_key}")
                    try:
                        # Upload the image to S3
                        s3_client.upload_fileobj(
                            io.BytesIO(file_bytes),
                            S3_BUCKET,
                            s3_key,
                            ExtraArgs={'ContentType': 'image/jpeg'}
                        )
                        print(f"Successfully uploaded mock image to S3: {s3_key}")
                        
                        return {
                            "filename": filename,
                            "s3_key": s3_key
                        }
                    except Exception as s3_error:
                        print(f"Error uploading to S3: {s3_error}")
                        base64_data = base64.b64encode(file_bytes).decode('ascii')
                        return {
                            "filename": filename,
                            "data": base64_data,
                            "s3_error": str(s3_error)
                        }
                else:
                    base64_data = base64.b64encode(file_bytes).decode('ascii')
                    return {
                        "filename": filename,
                        "data": base64_data
                    }
            
            # Real camera code continues below...
            
            # We'll add a check for focus errors
            focus_error_count = 0
            max_focus_retries = 2  # Allow 2 internal retries for focus issues
            
            for i in range(max_focus_retries + 1):
                try:
                    # Use our USB operation context for safer handling
                    with usb_operation_context():
                        file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"capture_{timestamp}.jpg"
                        
                        # Get the camera file
                        camera_file = self.camera.file_get(
                            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
                        
                        # Get the binary data
                        file_data = camera_file.get_data_and_size()
                        file_bytes = bytes(file_data)
                        
                        # If session_id is provided, upload to S3
                        if session_id:
                            s3_key = f"raw-captures/{session_id}/{filename}"
                            
                            print(f"Uploading image to S3: {s3_key}")

                            try:
                                # Upload the image to S3
                                s3_client.upload_fileobj(
                                    io.BytesIO(file_bytes),
                                    S3_BUCKET,
                                    s3_key,
                                    ExtraArgs={'ContentType': 'image/jpeg'}
                                )
                                print(f"Successfully uploaded image to S3: {s3_key}")
                                
                                # Return the S3 key and filename
                                # Reset USB error count on success
                                usb_error_count = 0
                                return {
                                    "filename": filename,
                                    "s3_key": s3_key
                                }
                            except Exception as s3_error:
                                print(f"Error uploading to S3: {s3_error}")
                                # If S3 upload fails, we still have the image data,
                                # so return it as base64 as a fallback
                                base64_data = base64.b64encode(file_bytes).decode('ascii')
                                # Reset USB error count on success
                                usb_error_count = 0
                                return {
                                    "filename": filename,
                                    "data": base64_data,
                                    "s3_error": str(s3_error)
                                }

                            """
                            
                            

                            # return an immediate response indicating that the upload has started
                            response = {
                                "status": "success",
                                "message": "Image upload started",
                                "s3_key": s3_key
                            }
                            """

                            # Send the response back to the client
                            # This allows the client to show a processing message
                            # Start the upload in a separate thread or asynchronously
                            #threading.Thread(target=self.upload_to_s3, args=(file_bytes, s3_key, filename)).start()#
                            
                            #return jsonify(response)#

                            return self.upload_to_s3(file_bytes, s3_key, filename)
                        
                        else:
                            # If no session_id provided, return base64 as before (backward compatibility)
                            base64_data = base64.b64encode(file_bytes).decode('ascii')
                            # Reset USB error count on success
                            usb_error_count = 0
                            return {
                                "filename": filename,
                                "data": base64_data
                            }
                        
                except gp.GPhoto2Error as e:
                    error_str = str(e).lower()
                    
                    # Track USB errors
                    if "usb" in error_str or "claim" in error_str or "device" in error_str or "i/o" in error_str:
                        with usb_reset_lock:
                            usb_error_count += 1
                            print(f"USB error during capture, incrementing count to {usb_error_count}")
                            
                        # If we've accumulated too many USB errors, try cleanup
                        if usb_error_count >= 3:
                            print("USB error threshold reached during capture, performing cleanup")
                            cleanup_usb_resources()
                            
                    # Add gp_result_as_string to our list of focus-related errors
                    if (any(term in error_str for term in ["focus", "proximity", "too close"]) or 
                        "-110" in error_str or 
                        "gp_result_as_string" in error_str):
                        # This looks like a focus/proximity error
                        focus_error_count += 1
                        if i < max_focus_retries:
                            print(f"Focus/proximity error detected, retry {i+1}/{max_focus_retries}: {error_str}")
                            time.sleep(0.5)  # Short pause between retries
                            continue
                        else:
                            # We've run out of internal retries
                            raise Exception("You might be too close to the camera. Take a step back and try again.")
                    else:
                        # Not a focus error, just raise it
                        raise
            
            # We shouldn't reach here, but just in case
            raise Exception("Unknown error during capture")
        except gp.GPhoto2Error as e:
            # Track USB errors in final error handler
            error_str = str(e).lower()
            if "usb" in error_str or "claim" in error_str or "device" in error_str or "i/o" in error_str:
                with usb_reset_lock:
                    usb_error_count += 1
                    print(f"USB error in final handler, incrementing count to {usb_error_count}")
                    
            print(f"Error capturing image: {e}")
            print(traceback.format_exc())
            raise
        except Exception as e:
            print(f"Unexpected error during capture: {e}")
            print(traceback.format_exc())
            raise
    
    def cleanup(self):
        """
        Thoroughly clean up camera resources to prevent USB device claim issues.
        """
        print("\nPerforming thorough camera cleanup...")
        
        try:
            if self.camera:
                try:
                    # Try to exit the camera properly first
                    print("Exiting camera...")
                    self.camera.exit()
                    print("Camera exit successful")
                except Exception as e:
                    print(f"Error during camera exit: {e}")
                
                # Always set camera to None regardless of exit success
                self.camera = None
            
            # Set state to not ready
            self.is_ready = False
            
            # Force garbage collection to clean up any lingering references
            gc.collect()
            
            # Check if we need to do deeper USB cleanup based on error count
            global usb_error_count
            if usb_error_count >= 3:
                print(f"USB error count ({usb_error_count}) exceeds threshold, performing deep USB cleanup")
                cleanup_usb_resources()
            
            print("Camera cleanup complete")
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
            # Still make sure camera reference is removed
            self.camera = None
            self.is_ready = False

    def pre_warm_camera(self):
        """
        Pre-warm the camera to wake it from sleep by trying a simple preview operation.
        This is a gentler approach than force_reset, trying to wake up the camera without
        completely reinitializing the connection.
        """
        print("Attempting to pre-warm (wake up) camera...")
        
        try:
            if not self.camera:
                print("No camera object exists, cannot pre-warm")
                return False
            
            # First try a benign operation - check the battery level if possible
            try:
                print("Checking camera battery level...")
                config = self.camera.get_config()
                # Try to find battery level if the camera supports it
                battery_section = None
                for i in range(config.count_children()):
                    child = config.get_child(i)
                    if child.get_name() == 'status' or child.get_name() == 'battery':
                        battery_section = child
                        break
                
                if battery_section:
                    for j in range(battery_section.count_children()):
                        child = battery_section.get_child(j)
                        if 'battery' in child.get_name().lower():
                            battery_level = child.get_value()
                            print(f"Battery level: {battery_level}")
                            # If battery is very low, this might be the cause of connectivity issues
                            if isinstance(battery_level, str) and any(term in battery_level.lower() for term in ['low', 'empty', 'critical']):
                                print("WARNING: Battery level is low - this might cause connection problems")
                            elif isinstance(battery_level, (int, float)) and battery_level < 25:
                                print("WARNING: Battery level is below 25% - this might cause connection problems")
            except Exception as battery_error:
                print(f"Could not check battery level: {battery_error}")
                # Continue anyway, this was just informational
            
            # Next try a preview capture - this often wakes up the camera
            print("Attempting preview capture to wake camera...")
            try:
                # Try to capture a preview image (doesn't save to memory card)
                preview_file = self.camera.capture_preview()
                if preview_file:
                    print("Successfully captured preview image - camera is awake")
                    return True
            except gp.GPhoto2Error as preview_error:
                print(f"Preview capture failed: {preview_error}")
                # If it's an I/O error, the camera might be in deep sleep
                if any(term in str(preview_error).lower() for term in ['i/o', 'timeout', 'busy']):
                    print("Camera appears to be in deep sleep, preview failed")
                    return False
                # If it mentions focus or lens errors, the camera is at least partially responsive
                elif any(term in str(preview_error).lower() for term in ['focus', 'lens']):
                    print("Camera is responsive but has lens/focus issues")
                    return True
            
            # If we got here without returning, try one more basic operation
            print("Trying camera summary as fallback...")
            summary = self.camera.get_summary()
            if summary:
                print("Camera is responsive to summary request, marking as awake")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error during camera pre-warm: {e}")
            print(traceback.format_exc())
            return False

    def force_reset(self):
        """
        Completely reset camera and libgphoto2 state for recovery from serious errors.
        This is a more aggressive recovery method when normal init/wake methods fail.
        """
        print("Performing force reset of camera connection...")
        reset_success = False
        
        try:
            # First try to properly exit any existing camera
            if self.camera:
                try:
                    self.camera.exit()
                    print("Successfully exited existing camera connection")
                except Exception as e:
                    print(f"Error exiting camera during force reset: {e}")
                    # Continue with reset even if exit fails
                
                # Clear the camera object
                self.camera = None
            
            # Set current state to not ready
            self.is_ready = False
            
            # Force garbage collection to ensure all references are cleared
            gc.collect()
            
            # Wait a moment for USB reset - longer wait for more reliable reset
            print("Waiting for USB bus to reset...")
            time.sleep(3)
            
            # Try the simplest possible initialization first (no abilities setup)
            try:
                print("Attempting simple initialization without abilities setup...")
                self.camera = gp.Camera()
                self.camera.init()
                print("Simple initialization successful!")
                self.is_ready = True
                reset_success = True
                return True
            except gp.GPhoto2Error as simple_error:
                print(f"Simple initialization failed: {simple_error}")
                if "Could not claim the USB device" in str(simple_error):
                    print("Camera is still locked by another process")
                    self.is_ready = False
                    self.camera = None
                    # No point continuing if device is claimed
                    return False
                # For other errors, continue with more advanced recovery methods
            
            # Create a completely new camera object
            print("Creating new camera object after reset...")
            self.camera = gp.Camera()
            
            # Try to initialize with extended timeout
            print("Attempting to initialize after reset...")
            init_timeout = 20  # Extended timeout for reset recovery
            start_time = time.time()
            success = False
            
            # Track critical errors that might indicate we need physical reconnection
            critical_errors = 0
            critical_error_types = []
            
            while time.time() - start_time < init_timeout and not success:
                try:
                    # Create a fresh context for each attempt
                    context = gp.Context()
                    self.camera.init(context)
                    print("Camera initialized successfully after force reset!")
                    success = True
                    reset_success = True
                except gp.GPhoto2Error as e:
                    error_message = str(e)
                    print(f"Reset initialization attempt failed: {error_message}")
                    
                    # Check for abilities errors
                    if "set_abilities" in error_message or "argument" in error_message or "type" in error_message:
                        print("Encountered abilities error during reset. This may require physical reconnection.")
                        critical_errors += 1
                        critical_error_types.append("abilities error")
                        
                        # Try one more approach - create a camera without any abilities setup
                        try:
                            print("Trying last resort initialization without abilities...")
                            self.camera = gp.Camera()
                            self.camera.init()
                            print("Basic initialization successful!")
                            success = True
                            reset_success = True
                            break
                        except Exception as basic_error:
                            print(f"Basic initialization also failed: {basic_error}")
                            # If this also fails, we need physical reconnection
                            critical_errors += 1
                    
                    # Track critical errors
                    if ("Unknown model" in error_message or 
                        "[-105]" in error_message or 
                        "I/O error" in error_message or
                        "permission denied" in error_message.lower()):
                        critical_errors += 1
                        critical_error_types.append(error_message)
                    
                    # For "Unknown model" errors, the camera might be physically disconnected
                    if "Unknown model" in error_message or "[-105]" in error_message:
                        print("Camera physical connection may be lost. Hardware reconnection might be needed.")
                        # Try one more time after a longer delay
                        print("Waiting for possible hardware recovery...")
                        time.sleep(5)
                        try:
                            # Try to reinitialize the USB context by creating completely new objects
                            self.camera = None
                            gc.collect()
                            
                            self.camera = gp.Camera()  # Create a new camera object
                            self.camera.init(context)
                            print("Camera recovered after extended wait!")
                            success = True
                            reset_success = True
                        except Exception as retry_error:
                            print(f"Final retry failed: {retry_error}")
                            # If this fails with the same error, likely needs physical reconnection
                            if "Unknown model" in str(retry_error) or "[-105]" in str(retry_error):
                                print("Definitive hardware reconnection required")
                                self.is_ready = False
                                self.camera = None
                                return False
                    
                    # If we get USB device claimed error, the camera might be working with another process
                    if "Could not claim the USB device" in error_message:
                        print("Camera appears to be in use by another process after reset")
                        
                        # Try to forcibly release the camera by reinitializing the camera context
                        print("Trying to forcibly release camera from other processes...")
                        try:
                            # Clear all existing objects and try a more aggressive approach
                            self.camera = None
                            gc.collect()
                            
                            # Longer wait to allow OS to clear USB claims
                            time.sleep(5)  # Extended wait time for USB release
                            
                            # Create a completely new camera instance
                            self.camera = gp.Camera()
                            context = gp.Context()  # Fresh context
                            self.camera.init(context)
                            print("Camera successfully claimed after force reset!")
                            success = True
                            reset_success = True
                        except Exception as reclaim_error:
                            print(f"Failed to reclaim camera: {reclaim_error}")
                            if "Could not claim the USB device" in str(reclaim_error):
                                print("Camera is still locked by another process. Manual intervention required.")
                                self.is_ready = False
                                self.camera = None
                                return False
                    
                    # If we have too many critical errors, break early
                    if critical_errors >= 3:
                        print(f"Multiple critical errors detected: {critical_error_types}")
                        print("This likely indicates a hardware issue requiring manual intervention")
                        break
                    
                    # Break out early if we've been trying too long with serious errors
                    if time.time() - start_time > init_timeout * 0.7 and not success:
                        print("Reset taking too long, likely hardware issue")
                        break
                    
                    # Wait before retrying
                    time.sleep(1)
            
            # If we were successful, mark camera as ready
            if success:
                self.is_ready = True
                print("Camera force reset successful!")
            else:
                self.is_ready = False
                print("Camera force reset failed")
                if self.camera:
                    try:
                        self.camera.exit()
                    except:
                        pass
                self.camera = None
            
            return reset_success
            
        except Exception as e:
            print(f"Error during force reset: {e}")
            print(traceback.format_exc())
            self.is_ready = False
            if self.camera:
                try:
                    self.camera.exit()
                except:
                    pass
            self.camera = None
            return False

    def auto_wake(self, max_attempts=2):
        """
        Comprehensive auto-wake function that tries multiple strategies to wake up a sleeping camera.
        Returns True if successful, False otherwise.
        """
        print("Starting comprehensive auto-wake sequence...")
        
        if not self.camera:
            print("No camera object exists, cannot auto-wake")
            return False
        
        # Track wake attempts
        wake_success = False
        attempts = 0
        
        while not wake_success and attempts < max_attempts:
            attempts += 1
            print(f"Auto-wake attempt {attempts}/{max_attempts}")
            
            # Strategy 1: Simple pre-warm (gentlest approach)
            try:
                print("Strategy 1: Gentle pre-warm...")
                wake_success = self.pre_warm_camera()
                if wake_success:
                    print("Gentle pre-warm successful!")
                    self.is_ready = True
                    return True
                else:
                    print("Gentle pre-warm failed, trying next strategy")
            except Exception as e:
                print(f"Error during gentle pre-warm: {e}")
            
            # Strategy 2: Try to take a preview (slightly more aggressive)
            try:
                print("Strategy 2: Preview capture wake...")
                preview_file = self.camera.capture_preview()
                if preview_file:
                    print("Preview capture successful, camera is awake!")
                    wake_success = True
                    self.is_ready = True
                    return True
            except Exception as e:
                print(f"Error during preview capture wake: {e}")
                # If we get an error, the camera might be deeply asleep
                # Continue to more aggressive methods
            
            # Strategy 3: Try accessing basic camera properties
            try:
                print("Strategy 3: Config access wake...")
                # Try to access and set a simple camera property
                config = self.camera.get_config()
                # Just accessing config can sometimes wake the camera
                print("Successfully accessed camera config")
                
                # Test if camera is responsive
                is_ready = self.check_camera_status()
                if is_ready:
                    print("Camera is now responsive after config access")
                    wake_success = True
                    self.is_ready = True
                    return True
                else:
                    print("Camera still not fully responsive after config access")
            except Exception as e:
                print(f"Error during config access wake: {e}")
            
            # Strategy 4: Force reset (most aggressive)
            if attempts >= max_attempts - 1:  # Only on final attempt
                try:
                    print("Strategy 4: Force reset (final attempt)...")
                    reset_success = self.force_reset()
                    if reset_success:
                        print("Force reset successful!")
                        wake_success = True
                        self.is_ready = True
                        return True
                    else:
                        print("Force reset failed")
                except Exception as e:
                    print(f"Error during force reset: {e}")
            
            # Wait between attempts
            if not wake_success and attempts < max_attempts:
                print(f"Waiting before next auto-wake attempt...")
                time.sleep(2)  # Wait between attempts
        
        # Final status
        if wake_success:
            print("Auto-wake sequence successful!")
            self.is_ready = True
            return True
        else:
            print("All auto-wake strategies failed")
            return False

# Create global camera control instance
camera_control = None

@app.route('/', methods=['GET'])
def test():
    return jsonify({
        "status": "success",
        "message": "Camera server is running"
    })



@app.route('/init', methods=['POST'])
def init_camera():
    """Initialize the camera via API request"""
    global camera_control, initialization_in_progress, last_init_attempt, usb_error_count
    
    # Check if initialization was recently attempted or is in progress
    if initialization_in_progress:
        return jsonify({
            "status": "error",
            "message": "Initialization already in progress",
            "error_code": "INIT_IN_PROGRESS"
        }), 409

    initialization_in_progress = True
    
    try:
        print("Creating new camera control instance...")
        # Create a CameraControl instance without initializing
        camera_control = CameraControl(skip_init=True)
        
        # First initialization attempt
        print("Starting first initialization attempt...")
        try:
            print("Initializing camera...")
            result = camera_control.initialize_camera()
            if result.get("status") == "success":
                initialization_in_progress = False
                print("Camera initialization successful!")
                # Update global state
                last_init_attempt = time.time()
                usb_error_count = 0  # Reset error count on success
                
                return jsonify({
                    "status": "success",
                    "message": "Camera initialized successfully",
                    "camera_info": result.get("camera_info", {})
                }), 200
            else:
                print(f"First initialization attempt failed: {result.get('message', 'Unknown error')}")
                error_message = result.get("message", "Unknown error")
                
                # Detect if the error is related to device claim and might need critical fix
                if "Could not claim the USB device" in error_message or "locked by another process" in error_message:
                    critical_fix_needed = True
                else:
                    critical_fix_needed = False
        except Exception as e:
            print(f"First initialization attempt raised exception: {str(e)}")
            error_message = str(e)
            # Detect if the error is related to device claim
            if "Could not claim the USB device" in error_message or "locked by another process" in error_message:
                critical_fix_needed = True
            else:
                critical_fix_needed = False
        
        # If first attempt failed, try service termination approach
        print("Moving to termination approach")
        print("Initial initialization failed. Attempting to terminate camera services and trying again...")
        
        # Try terminating camera services
        success = terminate_macos_camera_services()
        
        # Force a cleanup of resources
        if camera_control:
            camera_control.cleanup()
            
        # Perform additional cleanup and wait longer for resources to be released
        cleanup_usb_resources()
        time.sleep(3)  # Increased wait time to 3 seconds after service termination
        
        print("After termination, creating fresh camera control...")
        camera_control = CameraControl(skip_init=True)
        
        # Second initialization attempt
        print("Second initialization attempt after service termination...")
        try:
            print("Initializing camera...")
            result = camera_control.initialize_camera()
            if result.get("status") == "success":
                initialization_in_progress = False
                print("Camera initialization successful after service termination!")
                # Update global state
                last_init_attempt = time.time()
                usb_error_count = 0  # Reset error count on success
                
                return jsonify({
                    "status": "success",
                    "message": "Camera initialized successfully after service termination",
                    "camera_info": result.get("camera_info", {})
                }), 200
            else:
                print(f"Second initialization attempt failed: {result.get('message', 'Unknown error')}")
                error_message = result.get("message", "Unknown error")
                
                # Check if the error indicates a critical USB issue
                if "Could not claim the USB device" in error_message or "locked by another process" in error_message:
                    critical_fix_needed = True
                else:
                    critical_fix_needed = False
        except Exception as e:
            print(f"Second initialization attempt raised exception: {str(e)}")
            error_message = str(e)
            # Check if the error indicates a critical USB issue
            if "Could not claim the USB device" in error_message or "locked by another process" in error_message:
                critical_fix_needed = True
            else:
                critical_fix_needed = False
        
        # If both attempts failed and we need a critical fix
        initialization_in_progress = False
        if critical_fix_needed:
            # Provide special instructions for critical USB issues
            return jsonify({
                "status": "critical_error",
                "message": "Camera is locked by another process or can't be accessed. Advanced recovery needed.",
                "error_details": error_message,
                "recovery_options": {
                    "critical_fix_endpoint": "/critical_usb_fix",
                    "webcam_mode": "If you're unable to resolve this issue, consider switching to webcam mode.",
                    "message": "The camera is locked at a system level. Click 'Critical Fix' for recovery options."
                }
            }), 503
        else:
            # Return normal error for non-critical issues
            return jsonify({
                "status": "error",
                "message": f"Failed to initialize camera: {error_message}",
                "suggestion": "Try using webcam mode or physically reconnect the camera and try again."
            }), 500
            
    except Exception as e:
        print(f"Unexpected error during initialization: {str(e)}")
        traceback.print_exc()
        initialization_in_progress = False
        return jsonify({
            "status": "error",
            "message": f"Unexpected error during initialization: {str(e)}",
            "suggestion": "Try using webcam mode or physically reconnect the camera and try again."
        }), 500

@app.route('/capture', methods=['POST'])
def capture():
    global camera_control
    try:
        # Get the session ID from the request
        data = request.get_json() or {}
        session_id = data.get('sessionId')
        
        if not session_id:
            return jsonify({
                "status": "error",
                "message": "Session ID is required for direct S3 upload"
            }), 400
        
        # If camera is not initialized or had failed, try to initialize it
        if not camera_control:
            try:
                print("Camera not initialized, attempting to initialize before capture")
                camera_control = CameraControl()
            except Exception as e:
                print(f"Error initializing camera before capture: {e}")
                print("Attempting capture anyway, in case camera works despite initialization errors")
        
        # Auto-wake check: If camera exists but might be sleeping, try to wake it before capture
        if camera_control and hasattr(camera_control, 'camera') and camera_control.camera:
            print("Checking if camera needs to be woken up before capture...")
            try:
                # Do a simple status check to detect sleep state
                is_awake = camera_control.check_camera_status()
                
                if not is_awake:
                    print("Camera appears to be sleeping, attempting comprehensive auto-wake before capture...")
                    # Use our enhanced auto-wake method
                    wake_success = camera_control.auto_wake(max_attempts=2)  # Quick try with 2 attempts
                    
                    if wake_success:
                        print("Successfully auto-woke camera before capture")
                    else:
                        print("WARNING: Auto-wake attempts failed, but proceeding with capture attempt")
            except Exception as wake_error:
                print(f"Error during pre-capture auto-wake: {wake_error}")
                # Continue with capture attempt despite wake error
        
        # Check battery level before attempting to capture
        battery_status = None
        try:
            if hasattr(camera_control, 'camera') and camera_control.camera:
                config = camera_control.camera.get_config()
                for i in range(config.count_children()):
                    child = config.get_child(i)
                    if child.get_name() in ['status', 'battery']:
                        for j in range(child.count_children()):
                            subchild = child.get_child(j)
                            if 'battery' in subchild.get_name().lower():
                                battery_value = subchild.get_value()
                                print(f"Battery before capture: {battery_value}")
                                if isinstance(battery_value, str):
                                    battery_status = battery_value
                                    if any(term in battery_value.lower() for term in ['low', 'empty', 'critical']):
                                        print("WARNING: Battery is low before capture attempt")
                                        # Continue anyway, we'll just log the warning
                                elif isinstance(battery_value, (int, float)):
                                    battery_status = f"{battery_value}%"
                                    if battery_value < 15:
                                        print(f"WARNING: Battery at critical level ({battery_value}%) before capture")
                                    elif battery_value < 25:
                                        print(f"WARNING: Battery is low ({battery_value}%) before capture")
        except Exception as battery_error:
            print(f"Could not check battery before capture: {battery_error}")
        
        max_retries = 3
        last_error = None
        focus_errors = 0  # Track focus errors specifically
        sleep_errors = 0  # Track sleep/IO errors specifically
        battery_errors = 0  # Track battery-related errors
        
        # Try multiple times to capture an image
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Capture attempt {attempt}/{max_retries}")
                # Pass the session ID to capture_image method
                capture_result = camera_control.capture_image(session_id)
                
                # Check if we got an S3 key
                if "s3_key" in capture_result:
                    # Return the S3 key (direct S3 upload succeeded)
                    return jsonify({
                        "status": "success",
                        "filename": capture_result["filename"],
                        "s3_key": capture_result["s3_key"],
                        "battery_status": battery_status
                    })
                else:
                    # Fallback to base64 if direct upload failed
                    return jsonify({
                        "status": "success",
                        "filename": capture_result["filename"],
                        "image_data": capture_result["data"],
                        "s3_error": capture_result.get("s3_error"),
                        "battery_status": battery_status
                    })
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"Error during capture attempt {attempt}: {e}")
                
                # Track different types of errors
                if (any(term in error_msg for term in ["focus", "proximity", "too close"]) or 
                    "-110" in str(e) or 
                    "gp_result_as_string" in str(e)):
                    focus_errors += 1
                    print(f"Detected focus/proximity error: {e}")
                
                # Check for sleep/IO errors
                if (any(term in error_msg for term in ["i/o", "timeout", "communication", "busy"]) or
                    "-1" in str(e) or "-53" in str(e)):
                    sleep_errors += 1
                    print(f"Detected sleep/IO error: {e}")
                    
                    # Auto-wake attempt if this is a sleep/IO error
                    if sleep_errors > 0 and attempt < max_retries:
                        print("Sleep/IO error detected - attempting comprehensive auto-wake before next capture attempt")
                        try:
                            # Use our enhanced auto_wake method
                            if camera_control:
                                print("Attempting auto-wake during capture retry...")
                                wake_success = camera_control.auto_wake(max_attempts=2)  # Use 2 attempts for quick recovery
                                
                                if wake_success:
                                    print("Successfully auto-woke camera during capture retry")
                                    # Wait a moment for camera to stabilize
                                    time.sleep(1)
                                else:
                                    print("Auto-wake failed, will still try next capture attempt")
                        except Exception as wake_error:
                            print(f"Auto-wake during capture retry failed: {wake_error}")
                
                # Check for battery-related errors
                if any(term in error_msg for term in ["battery", "power", "energy"]):
                    battery_errors += 1
                    print(f"Detected possible battery-related error: {e}")
                
                last_error = e
                # Small delay between retries
                time.sleep(0.5)
        
        # If we get here, all retries failed
        
        # Prioritize error messages based on type (battery > focus > sleep > general)
        if battery_errors > 0:
            error_message = "Camera battery appears to be low. Please charge or replace the battery."
            error_type = "battery"
            print(f"Returning battery error response after detecting {battery_errors} battery-related errors")
        elif focus_errors > 0:
            error_message = "You might be too close to the camera. Take a step back and try again."
            error_type = "proximity"
            print(f"Returning proximity error response after detecting {focus_errors} focus errors")
        elif sleep_errors > 0:
            # If we had sleep errors, note that auto-wake was attempted
            error_message = "Camera appears to be unresponsive or in sleep mode. Try resetting the camera connection."
            error_type = "sleep"
            print(f"Returning sleep error response after detecting {sleep_errors} sleep/IO errors")
            return jsonify({
                "status": "error",
                "message": error_message,
                "error_type": error_type,
                "needs_reset": True,
                "auto_wake_attempted": True,
                "battery_status": battery_status
            }), 500
        else:
            # If we didn't detect any specific errors, use the last error message
            error_message = str(last_error) if last_error else "All capture attempts failed"
            error_type = "general"
            
            # Still check the last error message for specific keywords
            if ("battery" in error_message.lower() or 
                "power" in error_message.lower()):
                error_message = "Camera battery appears to be low. Please charge or replace the battery."
                error_type = "battery"
            elif ("focus" in error_message.lower() or 
                "too close" in error_message.lower() or 
                "-110" in error_message or
                "gp_result_as_string" in error_message):
                error_message = "You might be too close to the camera. Take a step back and try again."
                error_type = "proximity"
            elif ("i/o" in error_message.lower() or 
                  "timeout" in error_message.lower() or
                  "communication" in error_message.lower() or
                  "-1" in error_message or
                  "-53" in error_message):
                error_message = "Camera appears to be unresponsive or in sleep mode. Try resetting the camera connection."
                error_type = "sleep"
                return jsonify({
                    "status": "error",
                    "message": error_message,
                    "error_type": error_type,
                    "needs_reset": True,
                    "auto_wake_attempted": True,
                    "battery_status": battery_status
                }), 500
        
        return jsonify({
            "status": "error",
            "message": error_message,
            "error_type": error_type,
            "needs_reset": sleep_errors > 0 or battery_errors > 0,
            "battery_status": battery_status
        }), 500
    except Exception as e:
        error_message = str(e)
        error_type = "general"
        needs_reset = False
        
        # Check for specific error types
        if ("battery" in error_message.lower() or 
            "power" in error_message.lower()):
            error_type = "battery"
            error_message = "Camera battery appears to be low. Please charge or replace the battery."
            needs_reset = True
        elif ("focus" in error_message.lower() or 
            "too close" in error_message.lower() or 
            "proximity" in error_message.lower() or 
            "-110" in error_message or
            "gp_result_as_string" in error_message):
            error_type = "proximity"
            error_message = "You might be too close to the camera. Take a step back and try again."
        elif ("i/o" in error_message.lower() or 
              "timeout" in error_message.lower() or 
              "communication" in error_message.lower() or
              "-1" in error_message or
              "-53" in error_message):
            error_type = "sleep"
            error_message = "Camera appears to be unresponsive or in sleep mode. Try resetting the camera connection."
            needs_reset = True
            
            # Try an auto-wake attempt as a last resort
            if camera_control:
                try:
                    print("Attempting emergency auto-wake during capture error handling...")
                    # Use our comprehensive auto-wake method
                    wake_success = camera_control.auto_wake(max_attempts=3)  # More aggressive with 3 attempts
                    
                    if wake_success:
                        print("Emergency auto-wake successful! Retrying capture...")
                        # Try one final capture attempt
                        try:
                            capture_result = camera_control.capture_image(session_id)
                            
                            # If we got here, the emergency wake-up and capture worked!
                            if "s3_key" in capture_result:
                                return jsonify({
                                    "status": "success",
                                    "filename": capture_result["filename"],
                                    "s3_key": capture_result["s3_key"],
                                    "emergency_recovery": True
                                })
                            else:
                                # Fallback to base64 if direct upload failed
                                return jsonify({
                                    "status": "success",
                                    "filename": capture_result["filename"],
                                    "image_data": capture_result["data"],
                                    "s3_error": capture_result.get("s3_error"),
                                    "emergency_recovery": True
                                })
                        except Exception as final_capture_error:
                            print(f"Emergency capture attempt failed: {final_capture_error}")
                except Exception as emergency_error:
                    print(f"Emergency auto-wake failed: {emergency_error}")
            
            # If we reach here, the emergency recovery failed
            return jsonify({
                "status": "error",
                "message": error_message,
                "error_type": error_type,
                "needs_reset": needs_reset,
                "auto_wake_attempted": True
            }), 500
        
        print(f"Error during capture: {error_message}")
        print(traceback.format_exc())
        
        return jsonify({
            "status": "error",
            "message": error_message,
            "error_type": error_type,
            "needs_reset": needs_reset
        }), 500

"""

@app.route('/upload_status', methods=['GET'])
def upload_status():
    s3_key = request.args.get('s3_key')
    if not s3_key:
        return jsonify({"status": "error", "message": "s3_key is required"}), 400
    
    try:
        # Check if the object exists in S3
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        if response:
            return jsonify({"status": "success", "message": "Upload completed successfully", "s3_key": s3_key}), 200
    except Exception as e:
        print(f"Error checking upload status for {s3_key}: {e}")
        return jsonify({"status": "error", "message": "Upload not found or an error occurred", "error": str(e)}), 404
""" 
    


@app.route('/cleanup', methods=['POST'])
def cleanup():
    global camera_control
    if camera_control:
        camera_control.cleanup()
        camera_control = None
    return jsonify({"status": "success"})

# Helper functions for mosaic generation

def get_average_color(img):
    """Calculate the average color of an image."""
    img_array = np.array(img)
    avg_color = np.mean(img_array, axis=(0, 1))
    return avg_color

def find_best_match(target_color, thumbnails):
    """Find the thumbnail with the closest average color to the target color."""
    min_distance = float('inf')
    best_match = None
    
    target_color = np.array(target_color)
    
    for thumbnail in thumbnails:
        # Use perceptual color distance (weighted RGB) for better matching
        # Green is more perceptually important, followed by red, then blue
        weights = np.array([0.3, 0.6, 0.1])  # RGB weights
        
        # Calculate weighted Euclidean distance
        weighted_diff = (target_color - thumbnail['avg_color']) * weights
        distance = np.sum(weighted_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            best_match = thumbnail
    
    return best_match

def enhance_image(img, contrast_factor=CONTRAST_FACTOR):
    """Enhance the image by increasing contrast for more defined features."""
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Apply a sharpening filter first
    img = img.filter(ImageFilter.SHARPEN)
    
    # Then enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    # Also slightly enhance brightness to make subject pop more
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(1.1)
    
    return img

def fetch_thumbnails_from_s3(limit=THUMBNAIL_LIMIT):
    """Fetch thumbnail images from S3 bucket and process for mosaic use."""
    thumbnails = []
    
    try:
        print(f"Fetching up to {limit} thumbnails from S3...")
        
        # Create a paginator for listing objects
        paginator = s3_client.get_paginator('list_objects_v2')
        
        # Only look at selected-images directory in our bucket
        prefix = "selected-images/"
        
        # We'll count how many thumbnails we've processed
        count = 0
        
        # Paginate through results
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                # Skip if not an image file
                key = obj['Key']
                if not key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                # Download the image from S3
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
                    
                    # Add to our thumbnails list
                    thumbnails.append({
                        's3_key': key,
                        'avg_color': avg_color,
                        'image': hi_res_img,  # Store the high-res version
                        'display_size': THUMBNAIL_SIZE  # Remember the display size
                    })
                    
                    # Clean up the preview to save memory
                    del preview_img
                    
                    count += 1
                    if count >= limit:
                        break
                        
                except Exception as e:
                    print(f"Error processing S3 image {key}: {e}")
                    
            if count >= limit:
                break
        
        print(f"Successfully fetched {len(thumbnails)} high-resolution thumbnails")
                
        # If no thumbnails found or very few, create mock colored squares
        if len(thumbnails) < 10:
            print("Not enough thumbnails found, adding mock thumbnails...")
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                     (255, 255, 0), (255, 0, 255), (0, 255, 255),
                     (128, 0, 0), (0, 128, 0), (0, 0, 128)]
            
            for i, color in enumerate(colors):
                img = Image.new('RGB', INTERNAL_THUMBNAIL_SIZE, color)
                thumbnails.append({
                    's3_key': f"mock_thumbnail_{i}",
                    'avg_color': np.array(color),
                    'image': img,
                    'display_size': THUMBNAIL_SIZE
                })
    except Exception as e:
        print(f"Error fetching thumbnails from S3: {e}")
        print(traceback.format_exc())
        
        # If there's an error, provide mock thumbnails
        print("Creating mock thumbnails due to error...")
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (255, 0, 255), (0, 255, 255),
                 (128, 0, 0), (0, 128, 0), (0, 0, 128)]
        
        for i, color in enumerate(colors):
            img = Image.new('RGB', INTERNAL_THUMBNAIL_SIZE, color)
            thumbnails.append({
                's3_key': f"mock_thumbnail_{i}",
                'avg_color': np.array(color),
                'image': img,
                'display_size': THUMBNAIL_SIZE
            })
    
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
            
        # Save a copy of the original image for reference
        original_img = img.copy()
            
        # Enhance the image for better processing
        img = enhance_image(img)
        
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
        
        # Resize image to match cell grid
        img = img.resize((n_cols * CELL_SIZE[0], n_rows * CELL_SIZE[1]), Image.BILINEAR)
        original_img = original_img.resize((n_cols * CELL_SIZE[0], n_rows * CELL_SIZE[1]), Image.BILINEAR)
        
        # Calculate the mosaic dimensions - matching the grid spacing
        mosaic_width = n_cols * CELL_SIZE[0]
        mosaic_height = n_rows * CELL_SIZE[1]
        
        # Calculate the high-resolution mosaic dimensions 
        scale_factor = INTERNAL_THUMBNAIL_SIZE[0] / THUMBNAIL_SIZE[0]  # e.g., 4.0
        hi_res_width = int(mosaic_width * scale_factor)
        hi_res_height = int(mosaic_height * scale_factor)
        
        # Create the high-resolution canvas - use a light gray instead of pure white
        mosaic = Image.new('RGB', (hi_res_width, hi_res_height), (245, 245, 245))
        
        filled_cells = 0
        total_cells = n_cols * n_rows
        print(f"Creating high-resolution mosaic grid: {n_cols}x{n_rows} at {scale_factor}x scale")
        
        # Convert image to grayscale for analysis
        img_gray = img.convert('L')
        
        # Apply an edge detection filter for better feature detection
        edges = img_gray.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy arrays for processing
        img_array = np.array(img_gray)
        edges_array = np.array(edges)
        
        # Create a foreground mask using multiple techniques
        
        # 1. Use Otsu's method for automatic thresholding on the grayscale image
        try:
            threshold = threshold_otsu(img_array)
        except:
            # Fallback if skimage is not available or fails
            threshold = np.mean(img_array) * 0.8
        
        # 2. Create the initial mask - invert so darker areas (typically subject) are True
        foreground_mask1 = img_array < threshold
        
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
        
        # Process grid by extracting cell average colors first
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
                # Lower threshold to be more inclusive
                is_foreground = np.mean(cell_mask) > FOREGROUND_THRESHOLD
                row_foreground.append(is_foreground)
                
                del cell
            cell_colors.append(row_colors)
            cell_is_foreground.append(row_foreground)
        
        # Release the source images from memory
        del img
        del original_img
        del img_gray
        del edges
        del foreground_mask
        gc.collect()

        # Process each cell to place high-resolution thumbnails
        for y in range(n_rows):
            for x in range(n_cols):
                # Skip if not part of the foreground (person)
                if not cell_is_foreground[y][x]:
                    continue
                
                # Less aggressive skip for the subject area to retain more detail
                if random.random() < (SKIP_PROBABILITY * 0.5):
                    continue
                    
                # Get the average color
                avg_color = cell_colors[y][x]
                
                # Skip very bright cells
                brightness = np.mean(avg_color)
                if brightness > BRIGHTNESS_THRESHOLD:
                    continue
                    
                # Find the best match with more weight on color accuracy
                best_match = find_best_match(avg_color, thumbnails)
                if not best_match:
                    continue
                    
                # Calculate the high-resolution position
                hi_res_x = int(x * CELL_SIZE[0] * scale_factor + 
                              (CELL_SIZE[0] * scale_factor - INTERNAL_THUMBNAIL_SIZE[0]) // 2)
                hi_res_y = int(y * CELL_SIZE[1] * scale_factor + 
                              (CELL_SIZE[1] * scale_factor - INTERNAL_THUMBNAIL_SIZE[1]) // 2)
                
                # calculate the maximum random offset in pixels based on cell size
                max_offset_x = int(CELL_SIZE[0] * scale_factor * POSITOIN_RANDOMNESS)
                max_offset_y = int(CELL_SIZE[1] * scale_factor * POSITOIN_RANDOMNESS)
                
                # Generate random offests
                rand_offset_x = random.randint(-max_offset_x, max_offset_x)
                rand_offset_y = random.randint(-max_offset_y, max_offset_y)
                
                # Apply the random offsets
                hi_res_x += rand_offset_x
                hi_res_y += rand_offset_y

                #ensure the thumbnail stays within the mosaic bounds
                hi_res_x = max(0, min(hi_res_x, hi_res_width - INTERNAL_THUMBNAIL_SIZE[0]))
                hi_res_y = max(0, min(hi_res_y, hi_res_height - INTERNAL_THUMBNAIL_SIZE[1]))
                
                # Place the high-resolution thumbnail
                mosaic.paste(best_match['image'], (hi_res_x, hi_res_y))
                filled_cells += 1
                
                # Periodic status updates
                if filled_cells % 100 == 0:
                    print(f"Placed {filled_cells} thumbnails so far...")
        
        print(f"High-resolution mosaic created with {filled_cells}/{total_cells} cells filled")
        print(f"Final mosaic dimensions before rotation: {hi_res_width}x{hi_res_height} pixels")

        # rotate if neeeded
        if (mosaic.width > mosaic.height):
            rotated_mosaic = mosaic.transpose(Image.ROTATE_270)
        else:
            rotated_mosaic = mosaic
        
        print(f"Rotated mosaic to portrait orientation: {rotated_mosaic.width}x{rotated_mosaic.height} pixels")
        
        # Resize the mosaic to be smaller in relation to the white background canvas
        WHITE_BACKGROUND_SIZE = (rotated_mosaic.width, rotated_mosaic.height)
 
        # Calculate the aspect ratio of the original mosaic
        aspect_ratio = rotated_mosaic.width / rotated_mosaic.height
 
        #create a new white background canvas
        white_background = Image.new('RGB', WHITE_BACKGROUND_SIZE, (245, 245, 245))
         
        # Calculate the new dimensions that fit within the white background
        new_width = math.floor(rotated_mosaic.width / 4)
        new_height = math.floor(rotated_mosaic.height / 4)
         
        # Resize the mosaic to fit within the white background
        resized_mosaic = rotated_mosaic.resize((new_width, new_height), Image.LANCZOS)
         
        # Calculate the position to center the mosaic within the white background
        x = (WHITE_BACKGROUND_SIZE[0] - new_width) // 2
        y = (WHITE_BACKGROUND_SIZE[1] - new_height) // 2
 
        # Paste the resized mosaic onto the white background
        white_background.paste(resized_mosaic, (x, y))
         
        # Return the final mosaic
        return white_background
        #return rotated_mosaic

    except Exception as e:
        print(f"Error in mosaic creation: {e}")
        print(traceback.format_exc())
        raise

@app.route('/create_mosaic', methods=['POST'])
def generate_mosaic():
    try:
        print("Starting mosaic creation...")
        data = request.get_json() or {}
        
        # Get the source image S3 key and session ID
        source_key = data.get('sourceImageKey')
        session_id = data.get('sessionId')
        create_new = data.get('createNew', False)  # Default to False if not provided
        
        if not source_key or not session_id:
            return jsonify({
                "status": "error",
                "message": "Source image key and session ID are required"
            }), 400
        
        print(f"Processing mosaic request for image: {source_key}, createNew: {create_new}")
        
        # First check if a mosaic already exists for this session (unless createNew is True)
        if not create_new:
            try:
                print(f"Checking if mosaic already exists for session: {session_id}")
                # Check S3 for existing mosaics
                prefix = f"mosaics/{session_id}/"
                response = s3_client.list_objects_v2(
                    Bucket=S3_BUCKET,
                    Prefix=prefix
                )
                
                # If we found at least one mosaic
                if response.get('Contents') and len(response['Contents']) > 0:
                    # Sort by last modified to get the most recent one
                    existing_mosaics = sorted(
                        response['Contents'], 
                        key=lambda x: x['LastModified'],
                        reverse=True
                    )
                    
                    existing_key = existing_mosaics[0]['Key']
                    print(f"Found existing mosaic: {existing_key}")
                    
                    # Check if there's a thumbnail version as well
                    thumbnail_key = None
                    thumbnail_url = None
                    
                    # Look for thumbnail with same timestamp
                    filename_parts = existing_key.split('/')[-1].split('_')
                    if len(filename_parts) >= 2:
                        timestamp_part = filename_parts[-1].split('.')[0]
                        thumbnail_prefix = f"mosaics/{session_id}/mosaic_thumbnail_{timestamp_part}"
                        
                        thumbnail_response = s3_client.list_objects_v2(
                            Bucket=S3_BUCKET,
                            Prefix=thumbnail_prefix
                        )
                        
                        if thumbnail_response.get('Contents') and len(thumbnail_response['Contents']) > 0:
                            thumbnail_key = thumbnail_response['Contents'][0]['Key']
                            
                            # Generate a pre-signed URL for the thumbnail
                            thumbnail_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={
                                    'Bucket': S3_BUCKET,
                                    'Key': thumbnail_key
                                },
                                ExpiresIn=expiration
                            )
                    
                    # If no thumbnail found with matching timestamp, check for any thumbnail for this session
                    if not thumbnail_key:
                        thumbnail_prefix = f"mosaics/{session_id}/mosaic_thumbnail_"
                        
                        thumbnail_response = s3_client.list_objects_v2(
                            Bucket=S3_BUCKET,
                            Prefix=thumbnail_prefix
                        )
                        
                        if thumbnail_response.get('Contents') and len(thumbnail_response['Contents']) > 0:
                            # Sort by last modified to get the most recent one
                            existing_thumbnails = sorted(
                                thumbnail_response['Contents'], 
                                key=lambda x: x['LastModified'],
                                reverse=True
                            )
                            
                            thumbnail_key = existing_thumbnails[0]['Key']
                            
                            # Generate a pre-signed URL for the thumbnail
                            thumbnail_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={
                                    'Bucket': S3_BUCKET,
                                    'Key': thumbnail_key
                                },
                                ExpiresIn=expiration
                            )
                    
                    # Generate a pre-signed URL for the existing mosaic
                    expiration = 3600 * 24  # URL valid for 24 hours
                    presigned_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': S3_BUCKET,
                            'Key': existing_key
                        },
                        ExpiresIn=expiration
                    )
                    
                    # Return the existing mosaic info
                    return jsonify({
                        "status": "success",
                        "message": "Using existing mosaic",
                        "mosaic_key": existing_key,
                        "filename": existing_key.split('/')[-1],
                        "url": presigned_url,
                        "thumbnail_key": thumbnail_key,
                        "thumbnail_url": thumbnail_url,
                        "expiration": expiration
                    })
            except Exception as check_error:
                # Log the error but continue to create a new mosaic
                print(f"Error checking for existing mosaic (will create new one): {check_error}")
        else:
            print("Creating new mosaic as requested, ignoring any existing ones")

        # Download the source image from S3
        source_image = None
        try:
            print(f"Attempting to download image from S3: {source_key}")
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=source_key)
            source_image_data = response['Body'].read()
            source_image = Image.open(BytesIO(source_image_data))
            
            # Ensure the image is in RGB mode (in case it's a PNG with transparency)
            if source_image.mode != 'RGB':
                source_image = source_image.convert('RGB')
                
            print(f"Source image downloaded successfully, size: {source_image.size}")
        except Exception as e:
            error_msg = f"Failed to download source image: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
        
        # Fetch thumbnails from S3
        print("Fetching thumbnails for mosaic...")
        thumbnails = fetch_thumbnails_from_s3(limit=THUMBNAIL_LIMIT)
        
        if not thumbnails:
            return jsonify({
                "status": "error",
                "message": "No thumbnails available for creating mosaic"
            }), 500
        
        print(f"Creating mosaic using {len(thumbnails)} thumbnails...")
        
        # Create the mosaic
        mosaic = create_mosaic(source_image, thumbnails)
        
        # Clean up memory
        del source_image
        del thumbnails
        gc.collect()
        
        # Save the mosaic to a bytes buffer with high quality
        buffer = BytesIO()
        mosaic.save(buffer, format="JPEG", quality=92)  # Higher quality for better zoom detail
        buffer.seek(0)
        
        # Create a smaller thumbnail version for previewing in browser
        thumbnail_size = (800, 1200)  # Size that works well for previews
        mosaic_thumbnail = mosaic.copy()
        mosaic_thumbnail.thumbnail(thumbnail_size, Image.LANCZOS)
        
        # Save the thumbnail to a buffer (JPEG)
        thumbnail_buffer = BytesIO()
        mosaic_thumbnail.save(thumbnail_buffer, format="JPEG", quality=80)
        thumbnail_buffer.seek(0)
        
        # Also save as WebP format for better compatibility
        webp_thumbnail_buffer = BytesIO()
        mosaic_thumbnail.save(webp_thumbnail_buffer, format="WEBP", quality=80)
        webp_thumbnail_buffer.seek(0)
        
        # Clean up mosaic to free memory
        del mosaic_thumbnail
        gc.collect()
        
        # Clean up mosaic to free memory
        del mosaic
        gc.collect()
        
        # Upload the mosaic and thumbnails to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mosaic_filename = f"mosaic_{timestamp}.jpg"
        mosaic_key = f"mosaics/{session_id}/{mosaic_filename}"
        
        # Upload the full-size mosaic
        s3_client.upload_fileobj(
            buffer,
            S3_BUCKET,
            mosaic_key,
            ExtraArgs={
                'ContentType': 'image/jpeg',
                'CacheControl': 'max-age=31536000',  # 1 year caching
                'ContentDisposition': f'attachment; filename="{mosaic_filename}"'
            }
        )
        
        # Upload the JPEG thumbnail
        thumbnail_key = f"mosaics/{session_id}/thumbnail_{timestamp}.jpg"
        s3_client.upload_fileobj(
            thumbnail_buffer,
            S3_BUCKET,
            thumbnail_key,
            ExtraArgs={
                'ContentType': 'image/jpeg',
                'CacheControl': 'max-age=31536000',  # 1 year caching
            }
        )
        
        # Upload the WebP thumbnail
        webp_thumbnail_key = f"mosaics/{session_id}/thumbnail_{timestamp}.webp"
        s3_client.upload_fileobj(
            webp_thumbnail_buffer,
            S3_BUCKET,
            webp_thumbnail_key,
            ExtraArgs={
                'ContentType': 'image/webp',
                'CacheControl': 'max-age=31536000',  # 1 year caching
            }
        )
        
        # Generate a pre-signed URL with expiration time
        expiration = 3600 * 24  # URL valid for 24 hours
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': mosaic_key
            },
            ExpiresIn=expiration
        )
        
        # Generate a pre-signed URL for the thumbnail
        thumbnail_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': thumbnail_key
            },
            ExpiresIn=expiration
        )
        
        # Try to update the user record in DynamoDB
        try:
            update_user_mosaic_in_db(session_id, mosaic_key)
        except Exception as db_error:
            print(f"Error updating DynamoDB (non-critical): {str(db_error)}")
            # Continue even if DB update fails
        
        # Return the mosaic info
        return jsonify({
            "status": "success",
            "mosaic_key": mosaic_key,
            "filename": mosaic_filename,
            "url": presigned_url,
            "thumbnail_key": thumbnail_key,
            "thumbnail_url": thumbnail_url,
            "expiration": expiration
        })
        
    except Exception as e:
        error_msg = f"Error generating mosaic: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

# Helper function to update user mosaic in DynamoDB
def update_user_mosaic_in_db(session_id, mosaic_key):
    """Update the user record in DynamoDB with the generated mosaic key"""
    try:
        # Check if user exists in DynamoDB
        response = dynamodb_client.get_item(
            TableName=DYNAMODB_TABLE,
            Key={
                'user_id': {'S': session_id}
            }
        )
        
        # If user exists, update the mosaic key
        if 'Item' in response:
            print(f"Updating user {session_id} in DynamoDB with mosaic key {mosaic_key}")
            
            dynamodb_client.update_item(
                TableName=DYNAMODB_TABLE,
                Key={
                    'user_id': {'S': session_id}
                },
                UpdateExpression="SET mosaic_key = :mosaic_key, updated_at = :updated_at",
                ExpressionAttributeValues={
                    ':mosaic_key': {'S': mosaic_key},
                    ':updated_at': {'S': datetime.now().isoformat()}
                }
            )
            
            print(f"Successfully updated user {session_id} with mosaic key")
            return True
        else:
            print(f"User {session_id} not found in DynamoDB, skipping update")
            return False
    
    except Exception as e:
        print(f"Error updating user mosaic in DynamoDB: {str(e)}")
        return False

# Add a new endpoint to explicitly wake up a sleeping camera
@app.route('/wake_camera', methods=['POST'])
def wake_camera():
    global camera_control
    try:
        print("Attempting to wake up camera...")
        
        # If no camera control exists, try to create one
        if not camera_control:
            print("No camera control instance exists, creating one without initialization")
            camera_control = CameraControl(skip_init=True)  # Create without auto-init
        
        # Use our comprehensive auto-wake function
        wake_success = camera_control.auto_wake(max_attempts=3)  # Try up to 3 strategies
        
        if wake_success:
            return jsonify({
                "status": "success",
                "message": "Camera successfully woken up and ready",
                "auto_wakeup": True
            })
        else:
            # Since we're being called explicitly by the user to wake up,
            # try one more time with a force reset as a last resort
            print("Comprehensive auto-wake failed, attempting force reset as last resort...")
            reset_success = camera_control.force_reset()
            
            if reset_success:
                return jsonify({
                    "status": "success",
                    "message": "Camera successfully reset and ready",
                    "reset_required": True
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Camera couldn't be woken up automatically. Please check connection or try physically waking the camera.",
                    "auto_wake_attempted": True
                }), 500
            
    except Exception as e:
        error_details = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "auto_wake_attempted": True
        }
        print("Wake camera error:", error_details)
        return jsonify(error_details), 500

# Add a new endpoint to pre-warm the camera
@app.route('/pre_warm', methods=['POST'])
def pre_warm_camera():
    global camera_control
    try:
        print("Attempting to pre-warm camera...")
        
        # If camera control doesn't exist, try to initialize it
        if not camera_control:
            try:
                print("No camera control instance exists, creating one")
                camera_control = CameraControl(skip_init=True)  # Create without auto-init
            except Exception as init_error:
                print(f"Error initializing camera for pre-warm: {init_error}")
                error_message = str(init_error)
                
                # Return more specific error messages for common problems
                if "Unknown model" in error_message or "[-105]" in error_message:
                    return jsonify({
                        "status": "error",
                        "message": "No camera detected for pre-warm. Check connection.",
                        "error_code": "NO_CAMERA",
                        "hardware_issue": True,
                        "needs_reset": True
                    }), 500
                elif "Could not claim the USB device" in str(init_error):
                    return jsonify({
                        "status": "error",
                        "message": "Camera is locked by another process.",
                        "error_code": "CAMERA_LOCKED",
                        "needs_reset": True
                    }), 500
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Could not initialize camera: {str(init_error)}",
                        "error_code": "INIT_FAILED",
                        "needs_reset": True,
                        "detail": error_message
                    }), 500
        
        # Check if camera exists - if not, try to create it
        if not hasattr(camera_control, 'camera') or not camera_control.camera:
            print("No camera object exists, creating basic camera object")
            try:
                camera_control.camera = gp.Camera()
                camera_control.camera.init()
                print("Basic camera initialization successful!")
            except Exception as camera_init_error:
                print(f"Error creating basic camera: {camera_init_error}")
                error_message = str(camera_init_error)
                return jsonify({
                    "status": "error",
                    "message": f"Could not create camera object: {error_message}",
                    "error_code": "CAMERA_INIT_FAILED",
                    "needs_reset": True
                }), 500
            
        # Use our enhanced auto_wake method with multiple strategies
        success = camera_control.auto_wake(max_attempts=3)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Camera pre-warmed successfully",
                "auto_wakeup": True
            })
        else:
            # If auto-wake fails, try a force reset as last resort
            print("Comprehensive auto-wake failed, attempting force reset as last resort...")
            reset_success = camera_control.force_reset()
            
            if reset_success:
                return jsonify({
                    "status": "success",
                    "message": "Camera needed force reset but is now ready",
                    "reset_required": True
                })
            else:
                # If all automatic methods fail
                return jsonify({
                    "status": "error",
                    "message": "Automatic wake-up failed. Camera may need physical intervention.",
                    "error_code": "AUTO_WAKE_FAILED",
                    "needs_reset": True,
                    "auto_wake_attempted": True
                }), 500
            
    except Exception as e:
        error_message = str(e)
        print(f"Unexpected error during pre-warm: {error_message}")
        error_details = {
            "status": "error",
            "message": f"Unexpected error during pre-warm: {error_message}",
            "error_code": "UNEXPECTED_ERROR",
            "traceback": traceback.format_exc(),
            "needs_reset": True,
            "auto_wake_attempted": True
        }
        print("Pre-warm camera error:", error_details)
        return jsonify(error_details), 500

# Add a new endpoint for force reset
@app.route('/force_reset', methods=['POST'])
def force_reset_camera():
    global camera_control
    try:
        print("Received force reset request")
        
        # Create a new camera control if needed
        if not camera_control:
            print("No camera control instance exists, creating one without initialization")
            camera_control = CameraControl(skip_init=True)  # Create without init
        
        # Perform force reset
        reset_success = False
        hardware_issue = False
        message = "Camera connection reset attempted"
        recommendation = None
        error_code = None
        
        try:
            # Perform the actual reset operation
            reset_success = camera_control.force_reset()
            
            if reset_success:
                message = "Camera connection successfully reset"
                error_code = None
            else:
                # If reset fails but didn't raise an exception, likely a hardware issue
                message = "Failed to reset camera connection"
                recommendation = "Try unplugging and reconnecting the camera"
                error_code = "RESET_FAILED"
                hardware_issue = True
        except Exception as reset_error:
            # Handle exceptions during reset
            reset_success = False
            error_message = str(reset_error)
            message = f"Error during camera reset: {error_message}"
            error_code = "RESET_ERROR"
            
            # Check for hardware-related issues
            if any(term in error_message.lower() for term in ["unknown model", "[-105]", "io", "usb", "permission"]):
                hardware_issue = True
                recommendation = "Camera may need to be physically disconnected and reconnected"
            else:
                recommendation = "Try again or switch to webcam mode"
                
            print(f"Force reset operation error: {reset_error}")
        
        # Build response - use a consistent format but always return 200
        # This helps the frontend handle the response without HTTP error complications
        response = {
            "status": "success" if reset_success else "error",
            "message": message
        }
        
        # Add optional fields if relevant
        if recommendation:
            response["recommendation"] = recommendation
            
        if error_code:
            response["error_code"] = error_code
            
        if hardware_issue:
            response["hardware_issue"] = True
        
        return jsonify(response)
        
    except Exception as e:
        # Catch any unexpected errors
        error_message = str(e)
        print(f"Unexpected error during force reset: {error_message}")
        print(traceback.format_exc())
        
        # Still return 200 with error details
        return jsonify({
            "status": "error",
            "message": f"Unexpected error during force reset: {error_message}",
            "error_code": "UNEXPECTED_ERROR",
            "recommendation": "Camera may need to be physically disconnected and reconnected",
            "hardware_issue": True
        }), 500

@app.route('/reset_connection', methods=['POST'])
def reset_connection():
    """
    Special endpoint focused on handling camera connection issues, particularly abilities setup errors.
    This is a simpler approach than force_reset, focusing only on recreating the camera with basic initialization.
    """
    global camera_control
    
    try:
        print("Resetting camera connection...")
        
        # If camera control exists, clean it up properly
        if camera_control:
            try:
                camera_control.cleanup()
            except Exception as e:
                print(f"Error cleaning up existing camera: {e}")
            finally:
                camera_control = None
        
        # Force garbage collection
        gc.collect()
        
        # Wait for USB bus reset
        print("Waiting for USB bus reset...")
        time.sleep(3)
        
        # Try the simplest possible initialization
        try:
            print("Creating new camera with basic initialization...")
            camera_control = CameraControl(skip_init=True)  # Create without auto-init
            
            # Manually initialize with the simplest method
            camera_control.camera = gp.Camera()
            camera_control.camera.init()
            
            # Mark as ready if we got this far
            camera_control.is_ready = True
            
            print("Basic camera initialization successful!")
            return jsonify({
                "status": "success",
                "message": "Camera connection reset successfully"
            }), 200
            
        except gp.GPhoto2Error as e:
            error_message = str(e)
            print(f"Error during basic initialization: {error_message}")
            
            # Check for common errors
            if "Could not claim the USB device" in error_message:
                return jsonify({
                    "status": "error",
                    "message": "Camera is locked by another application. Close other apps using the camera.",
                    "error_code": "CAMERA_LOCKED",
                    "needs_physical_intervention": True
                }), 500
                
            elif "Unknown model" in error_message or "[-105]" in error_message:
                return jsonify({
                    "status": "error",
                    "message": "No camera detected. Try physically reconnecting the camera.",
                    "error_code": "NO_CAMERA",
                    "needs_physical_intervention": True
                }), 500
                
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Error resetting camera connection: {error_message}",
                    "error_code": "RESET_ERROR",
                    "needs_physical_intervention": True,
                    "detail": error_message
                }), 500
                
    except Exception as e:
        print(f"Unexpected error during connection reset: {e}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": f"Unexpected error during connection reset: {str(e)}",
            "error_code": "UNEXPECTED_ERROR",
            "needs_physical_intervention": True,
            "detail": str(e)
        }), 500

@app.route('/reset_usb', methods=['POST'])
def reset_usb():
    """
    Reset USB devices and clean up USB resources.
    This is a more thorough implementation that tries to fix persistent USB issues.
    """
    print("\n--- Performing Thorough USB Reset ---")
    
    # Reset global initialization state to prevent conflicts
    global camera_control
    global is_initializing
    global initialization_in_progress
    global last_init_attempt
    
    # Clear initialization flags
    is_initializing = False
    initialization_in_progress = False
    last_init_attempt = 0
    
    # Clean up any existing camera control
    if camera_control:
        try:
            print("Cleaning up existing camera control")
            camera_control.cleanup()
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
    
    # Set camera_control to None to force a fresh initialization next time
    camera_control = None
    
    # Perform thorough USB cleanup
    success = cleanup_usb_resources()
    
    # Reset USB error counter regardless of cleanup success
    global usb_error_count
    usb_error_count = 0
    
    # Create a detailed response
    if success:
        response = {
            "status": "success",
            "message": "USB devices reset and resources cleaned up successfully",
            "auto_recovery": True
        }
    else:
        response = {
            "status": "warning",
            "message": "USB reset attempted but may not have been fully successful",
            "recommendation": "If problems persist, try unplugging and reconnecting the camera, or restart your computer"
        }
    
    # Add an additional wait to let USB settle
    time.sleep(3)
    print("USB reset and cleanup complete")
    
    return jsonify(response)

@app.route('/terminate_camera_services', methods=['POST'])
def terminate_services():
    """API endpoint to terminate macOS camera services that might be interfering with the camera"""
    # Get global references
    global initialization_in_progress, camera_control, usb_error_count
    
    try:
        print("Received request to terminate camera services")
        
        # Terminate services
        success = terminate_macos_camera_services()
        
        # Try to cleanup camera resources
        if camera_control:
            camera_control.cleanup()
        
        # Reset global state
        initialization_in_progress = False
        usb_error_count = 0
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Camera services terminated successfully"
            }), 200
        else:
            return jsonify({
                "status": "partial",
                "message": "Some camera services could not be terminated"
            }), 200
    except Exception as e:
        print(f"Error during service termination: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/critical_usb_fix', methods=['POST'])
def critical_usb_fix():
    """API endpoint for critical USB issues requiring user interaction for recovery"""
    try:
        print("Received request for critical USB fix")
        
        # Generate commands for the user to copy/paste
        commands = {
            "reset_usb": {
                "title": "Reset USB System (requires admin privileges)",
                "description": "This command will restart macOS's USB subsystem. You'll need to enter your password.",
                "command": "sudo kextunload -b com.apple.driver.usb.AppleUSBHostController && sudo kextload -b com.apple.driver.usb.AppleUSBHostController"
            },
            "kill_ptpcamera": {
                "title": "Kill Camera Processes with Admin Rights",
                "description": "This command will forcefully terminate all camera services with administrator privileges.",
                "command": "sudo killall -9 ptpcamera mscamerad imagecaptureext PTPCamera VDCAssistant"
            },
            "unplug_instructions": {
                "title": "Physical Reconnection (most reliable solution)",
                "description": "For the most reliable recovery:",
                "steps": [
                    "1. Unplug the camera USB cable",
                    "2. Wait 10 seconds",
                    "3. Plug the camera back in",
                    "4. Wait 5 seconds before trying again"
                ]
            }
        }
        
        # Try some cleanup operations that don't need sudo
        terminate_macos_camera_services()
        cleanup_usb_resources()
        
        # Reset global state
        global initialization_in_progress, camera_control, usb_error_count
        initialization_in_progress = False
        usb_error_count = 0
        if camera_control:
            camera_control.cleanup()
            camera_control = None
            
        return jsonify({
            "status": "instructions",
            "message": "Critical USB fix instructions generated",
            "commands": commands
        }), 200
    except Exception as e:
        print(f"Error generating critical USB fix instructions: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error: {str(e)}"
        }), 500

# Flora API Configuration
FLORA_API_URL = 'https://florafauna-ai-dev--imgblend-vids-endpoint.modal.run'
FLORA_AUTH_TOKEN = 'c2VsZnBvcnRyYWl0cHJvamVjdDpTUFB4RkxPUkE='

def create_and_poll_blend(source_image_url, random_image_url, session_id, email):
    """
    Create a blend and poll for completion in a background thread
    
    Args:
        source_image_url: URL of the user's selected image
        random_image_url: URL of the random image to blend with
        session_id: User's session ID
        email: User's email address
    """
    thread = threading.Thread(
        target=_process_blend_in_background,
        args=(source_image_url, random_image_url, session_id, email)
    )
    thread.daemon = True
    thread.start()
    return True

def _process_blend_in_background(source_image_url, random_image_url, session_id, email):
    """
    Background worker function to handle blend creation, polling, and S3 upload
    """
    import time
    import json
    import requests
    import boto3
    import uuid
    from datetime import datetime
    
    try:
        print(f"📸 [BLEND STARTED] Beginning blend process for session {session_id}")
        logging.info(f"Starting blend process for session {session_id}")
        
        # Create DynamoDB client for blend records using the same credentials as other AWS services
        # Make sure to use the same region and credentials consistently
        dynamodb = boto3.resource(
            'dynamodb', 
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Use a constant or environment variable for the table name
        blend_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        blend_table = dynamodb.Table(blend_table_name)
        
        # Log the configuration (without sensitive data)
        logging.info(f"Using DynamoDB table '{blend_table_name}' in region '{AWS_REGION}'")
        
        # Step 1: Generate a unique ID for the blend
        blend_id = f"blend_{int(time.time())}_{uuid.uuid4().hex[:7]}"
        print(f"📸 [BLEND ID] Created blend ID: {blend_id}")
        
        # Step 2: Extract user information from random image URL
        random_user_email = None
        try:
            print(f"📸 [RANDOM USER] Attempting to extract user ID from random image URL: {random_image_url[:50]}...")
            
            # There are two possible approaches:
            # 1. Extract from the S3 key pattern in the URL
            # 2. Query the users DynamoDB table by the image key
            
            # Approach 1: Extract from URL pattern
            # The S3 URL might look like:
            # https://eternity-mirror-project.s3.amazonaws.com/selected-images/SESSION_ID/image.jpg
            # Or with a presigned URL:
            # https://eternity-mirror-project.s3.amazonaws.com/selected-images/SESSION_ID/image.jpg?X-Amz-Algorithm=...
            
            # Extract the S3 key from the URL by removing the base URL and query parameters
            s3_key = None
            
            # Remove any query parameters (everything after '?')
            clean_url = random_image_url.split('?')[0]
            
            # Extract the path part after the bucket name
            if 's3.amazonaws.com/' in clean_url:
                s3_key = clean_url.split('s3.amazonaws.com/')[1]
            elif f'{S3_BUCKET}.s3.' in clean_url:
                s3_key = clean_url.split(f'{S3_BUCKET}.s3.')[1].split('/', 1)[1]
            
            print(f"📸 [RANDOM USER] Extracted S3 key from URL: {s3_key}")
            
            # Extract the session ID from the key pattern if it includes 'selected-images/'
            if s3_key and 'selected-images/' in s3_key:
                # Expecting pattern like: selected-images/SESSION_ID/image.jpg
                parts = s3_key.split('/')
                selected_images_index = parts.index('selected-images')
                if selected_images_index + 1 < len(parts):
                    random_user_id = parts[selected_images_index + 1]
                    print(f"📸 [RANDOM USER] Extracted random user ID from path: {random_user_id}")
                    
                    # Try to look up this user's email in DynamoDB
                    user_table = dynamodb.Table(DYNAMODB_TABLE)
                    user_response = user_table.get_item(Key={'user_id': random_user_id})
                    if 'Item' in user_response:
                        random_user_email = user_response['Item'].get('email')
                        print(f"📸 [RANDOM USER] Found random user email: {random_user_email}")
            
            # If we couldn't get the email, log and continue without it
            if not random_user_email:
                print(f"📸 [RANDOM USER] Could not determine random user email. Proceeding without it.")
        except Exception as e:
            print(f"📸 [ERROR] Error extracting random user information: {str(e)}")
            # Non-fatal error, continue without the random user info
        
        # Step 3: Begin processing with Flora API
        # Prepare the API request
        print(f"📸 [FLORA API] Preparing request to Flora API")
        
        payload = {
            "prompt": "The photo of a person with a white background",
            "input_image_url": [
                source_image_url,
                random_image_url
            ],
            "strength_ab": 0.55,
            "seed": 0
        }
        
        # Make request to Flora API
        response = requests.post(
            FLORA_API_URL + '/',
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'accept': 'application/json',
                'Authorization': FLORA_AUTH_TOKEN
            }
        )
        
        if response.status_code != 200:
            print(f"📸 [ERROR] Flora API error: {response.status_code} - {response.text}")
            logging.error(f"Flora API error: {response.status_code} - {response.text}")
            return
        
        print(f"📸 [FLORA API] Received initial response from Flora API")
        
        # Parse the response
        result = response.json()
        request_id = result.get('request_id')
        progress_uuid = result.get('progress_uuid')
        polling_url = result.get('polling_url') or f"{FLORA_API_URL}/result?request_id={request_id}&progress_uuid={progress_uuid}"
        
        # Step 4: Create a temporary in-memory record to track this blend
        # Instead of storing in DynamoDB immediately, just create a reference for tracking
        blend_record = {
            'blend_id': blend_id,
            'session_id': session_id,
            'source_user_email': email,      # Source user's email (renamed for clarity)
            'random_user_email': random_user_email,  # Random image owner's email
            'email': email,                  # Keep original field for backward compatibility
            'source_image_url': source_image_url,
            'random_image_url': random_image_url,
            'blend_url': None,
            'status': 'processing',
            'polling_url': polling_url,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # We don't save to DynamoDB at this stage anymore
        print(f"📸 [TRACKING] Created temporary tracking record for blend: {blend_id}")
        
        # Step 5: Poll until completion or error
        max_polls = 100  # Maximum number of polling attempts
        poll_interval = 10  # Seconds between polls
        
        print(f"📸 [POLLING] Beginning to poll Flora API for results (every {poll_interval} seconds)")
        for i in range(max_polls):
            time.sleep(poll_interval)
            
            try:
                print(f"📸 [POLLING] Poll attempt {i+1}/{max_polls}")
                poll_response = requests.get(
                    polling_url,
                    headers={
                        'accept': 'application/json',
                        'Authorization': FLORA_AUTH_TOKEN
                    },
                    timeout=30
                )
                
                if poll_response.status_code != 200:
                    print(f"📸 [POLLING ERROR] Poll failed: {poll_response.status_code} - {poll_response.text}")
                    logging.error(f"Poll error: {poll_response.status_code} - {poll_response.text}")
                    continue
                
                poll_result = poll_response.json()
                status = poll_result.get('status', 'unknown')
                progress = poll_result.get('progress', 0)
                print(f"📸 [POLLING] Status: {status}, Progress: {progress}%")
                
                # Check if processing is complete
                if poll_result.get('status') == 'success':
                    print(f"📸 [SUCCESS] Blend processing completed successfully")
                    
                    # Get the video URL
                    flora_video_url = poll_result.get('image_url') or poll_result.get('output_url')
                    
                    if not flora_video_url:
                        print(f"📸 [ERROR] No video URL found in response")
                        logging.error("No video URL found in response")
                        break
                    
                    # Step 6: Download the video
                    print(f"📸 [DOWNLOAD] Downloading video from Flora API")
                    video_response = requests.get(flora_video_url, timeout=60)
                    
                    if video_response.status_code != 200:
                        print(f"📸 [ERROR] Failed to download video: {video_response.status_code}")
                        logging.error(f"Failed to download video: {video_response.status_code}")
                        break
                    
                    content_size = len(video_response.content) / 1024  # Size in KB
                    print(f"📸 [DOWNLOAD] Downloaded video successfully ({content_size:.2f} KB)")
                    
                    # Step 7: Upload to S3
                    print(f"📸 [S3 UPLOAD] Uploading video to S3")
                    s3_client = boto3.client(
                        's3',
                        region_name=AWS_REGION,
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                    )
                    timestamp = int(time.time())
                    s3_key = f"video-blends/{session_id}_{timestamp}.mp4"
                    
                    s3_client.put_object(
                        Bucket=S3_BUCKET,
                        Key=s3_key,
                        Body=video_response.content,
                        ContentType='video/mp4'
                    )
                    
                    print(f"📸 [S3 UPLOAD] Uploaded video to S3: {s3_key}")
                    
                    # Generate presigned URL
                    s3_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                        ExpiresIn=604800  # 1 week
                    )
                    
                    # Step 8: NOW save the completed blend to DynamoDB
                    print(f"📸 [DYNAMO DB] Saving blend record with completed status")
                    
                    # Update our tracking record with the final information
                    blend_record.update({
                        'status': 'completed',
                        'blend_url': s3_url,
                        'flora_video_url': flora_video_url,
                        's3_key': s3_key,
                        'updated_at': datetime.now().isoformat()
                    })

                    # !!! ENSURE gsi_pk is included when SAVING !!!
                    if 'gsi_pk' not in blend_record:
                         blend_record['gsi_pk'] = 'ALL_BLENDS'
                    
                    # Save the completed blend to DynamoDB
                    blend_table.put_item(Item=blend_record)
                    
                    print(f"📸 [COMPLETE] Blend processing completed and saved to S3: {s3_key}")
                    print(f"📸 [VIDEO URL] {s3_url}")
                    break
                
                elif poll_result.get('status') == 'error':
                    error_msg = poll_result.get('error', 'Unknown error')
                    print(f"📸 [ERROR] Blend failed: {error_msg}")
                    
                    # Update our tracking record with the error
                    blend_record.update({
                        'status': 'error',
                        'error_message': error_msg,
                        'updated_at': datetime.now().isoformat()
                    })

                    # !!! ENSURE gsi_pk is included when SAVING !!!
                    if 'gsi_pk' not in blend_record:
                        blend_record['gsi_pk'] = 'ALL_BLENDS'
                    
                    # Save the error record to DynamoDB
                    # (We still save error records to keep track of failures)
                    blend_table.put_item(Item=blend_record)
                    
                    break
            
            except Exception as e:
                print(f"📸 [ERROR] Error polling blend: {str(e)}")
                logging.error(f"Error polling blend: {str(e)}")
                # Continue to next poll attempt
        
        # If we've exhausted all poll attempts without success
        if i == max_polls - 1:
            print(f"📸 [TIMEOUT] Blend {blend_id} timed out after {max_polls} polling attempts")
            
            # Update our tracking record with the timeout error
            blend_record.update({
                'status': 'error',
                'error_message': f"Blend timed out after {max_polls} polling attempts",
                'updated_at': datetime.now().isoformat()
            })
            
            # Save the error record to DynamoDB
            blend_table.put_item(Item=blend_record)
    
    except Exception as e:
        print(f"📸 [CRITICAL ERROR] Error in blend background process: {str(e)}")
        logging.error(f"Error in blend background process: {str(e)}")
        traceback.print_exc()


@app.route('/create_blend', methods=['POST'])
def create_blend():
    """
    API endpoint to create a blend between two images
    """
    try:
        # Log basic diagnostic information
        logging.info(f"Received /create_blend request")
        logging.info(f"AWS Region: {AWS_REGION}")
        logging.info(f"S3 Bucket: {S3_BUCKET}")
        
        # Verify AWS credentials (without logging the actual values)
        has_aws_access_key = bool(AWS_ACCESS_KEY_ID)
        has_aws_secret_key = bool(AWS_SECRET_ACCESS_KEY)
        logging.info(f"Has AWS Access Key: {has_aws_access_key}")
        logging.info(f"Has AWS Secret Key: {has_aws_secret_key}")
        
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required parameters
        source_image_url = data.get('source_image_url')
        random_image_url = data.get('random_image_url')
        session_id = data.get('session_id')
        email = data.get('email')
        
        # Log request info (without exposing full URLs that might contain tokens)
        logging.info(f"Creating blend for session: {session_id}, email: {email}")
        logging.info(f"Source image: {source_image_url[:50]}...")
        logging.info(f"Random image: {random_image_url[:50]}...")
        
        # Validate required parameters
        if not all([source_image_url, random_image_url, session_id, email]):
            return jsonify({
                'error': 'Missing required parameters',
                'required': ['source_image_url', 'random_image_url', 'session_id', 'email']
            }), 400
        
        
        # Extract the selected image key from the users table
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        selected_image_key = None
        user_table = dynamodb.Table(DYNAMODB_TABLE)
        response = user_table.get_item(Key={'user_id': session_id})
        if 'Item' in response:
            selected_image_key = response['Item'].get('selected_image_key')
        
        # Save the blend interaction (this updates both tables)
        interaction_id = save_user_interaction(
            user_id=session_id,
            email=email,
            selected_image_key=selected_image_key,
            interaction_type="blend"
        )
        
        # Start background processing
        create_and_poll_blend(source_image_url, random_image_url, session_id, email)
        
        # Return success immediately (don't wait for completion)
        return jsonify({
            'success': True,
            'message': 'Blend processing started',
            'session_id': session_id,
            'interaction_id': interaction_id
        })
    
    except Exception as e:
        logging.error(f"Error creating blend: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# After the AWS configuration section at the top of the file
# Add this function to ensure the blend table exists

def ensure_blend_table_exists():
    """
    Verify that the DynamoDB table for blends exists and create it if not
    """
    try:
        blend_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        print(f"Verifying DynamoDB table '{blend_table_name}' exists...")
        
        # Create DynamoDB client
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Check if table exists
        existing_tables = [table.name for table in dynamodb.tables.all()]
        if blend_table_name in existing_tables:
            print(f"Table '{blend_table_name}' already exists")
            return True
        
        print(f"Creating DynamoDB table '{blend_table_name}'...")
        
        # Create the table
        table = dynamodb.create_table(
            TableName=blend_table_name,
            KeySchema=[
                {
                    'AttributeName': 'blend_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'blend_id',
                    'AttributeType': 'S'
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=blend_table_name)
        print(f"Table '{blend_table_name}' created successfully")
        return True
        
    except Exception as e:
        print(f"Error ensuring blend table exists: {str(e)}")
        return False

# After the ensure_blend_table_exists function, add this function:

def ensure_interactions_table_exists():
    """
    Verify that the DynamoDB table for user interactions exists and create it if not
    """
    try:
        interactions_table_name = os.getenv('REACT_APP_DYNAMODB_INTERACTIONS_TABLE', 'eternity-mirror-interactions')
        print(f"Verifying DynamoDB interactions table '{interactions_table_name}' exists...")
        
        # Create DynamoDB client
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Check if table exists
        existing_tables = [table.name for table in dynamodb.tables.all()]
        if interactions_table_name in existing_tables:
            print(f"Interactions table '{interactions_table_name}' already exists")
            return True
        
        print(f"Creating DynamoDB interactions table '{interactions_table_name}'...")
        
        # Create the table with a composite key (user_id + interaction_id)
        table = dynamodb.create_table(
            TableName=interactions_table_name,
            KeySchema=[
                {
                    'AttributeName': 'user_id',
                    'KeyType': 'HASH'  # Partition key
                },
                {
                    'AttributeName': 'interaction_id', 
                    'KeyType': 'RANGE'  # Sort key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'user_id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'interaction_id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'timestamp',
                    'AttributeType': 'S'
                }
            ],
            LocalSecondaryIndexes=[
                {
                    'IndexName': 'TimestampIndex',
                    'KeySchema': [
                        {
                            'AttributeName': 'user_id',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'timestamp',
                            'KeyType': 'RANGE'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    }
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=interactions_table_name)
        print(f"Interactions table '{interactions_table_name}' created successfully")
        return True
        
    except Exception as e:
        print(f"Error ensuring interactions table exists: {str(e)}")
        return False

# Helper function to save user interactions
def save_user_interaction(user_id, email, selected_image_key, interaction_type="capture"):
    """
    Save a user interaction to the interactions table
    
    Args:
        user_id: User's session ID
        email: User's email address (can be None for anonymous interactions)
        selected_image_key: S3 key of the selected image
        interaction_type: Type of interaction (capture, mosaic, blend, etc.)
    """
    try:
        # Generate a unique interaction ID
        interaction_id = f"{interaction_type}_{int(time.time())}_{uuid.uuid4().hex[:7]}"
        timestamp = datetime.now().isoformat()
        
        # Create DynamoDB resource
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Get the interactions table
        interactions_table_name = os.getenv('REACT_APP_DYNAMODB_INTERACTIONS_TABLE', 'eternity-mirror-interactions')
        interactions_table = dynamodb.Table(interactions_table_name)
        
        # Create the interaction record
        interaction_record = {
            'user_id': user_id,
            'interaction_id': interaction_id,
            'email': email,
            'selected_image_key': selected_image_key,
            'interaction_type': interaction_type,
            'timestamp': timestamp,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Save to the interactions table
        interactions_table.put_item(Item=interaction_record)
        
        print(f"📊 [INTERACTION] Saved {interaction_type} interaction {interaction_id} for user {user_id}")
        
        # Also update the main users table (for backward compatibility)
        users_table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Update or create the user record with the latest information
        user_record = {
            'user_id': user_id,
            'email': email,
            'selected_image_key': selected_image_key,
            'last_interaction_id': interaction_id,
            'last_interaction_type': interaction_type,
            'last_interaction_time': timestamp,
            'updated_at': timestamp
        }
        
        # Check if the user already exists
        response = users_table.get_item(Key={'user_id': user_id})
        if 'Item' in response:
            # If exists, preserve the created_at timestamp
            user_record['created_at'] = response['Item'].get('created_at', timestamp)
            
            # Preserve the mosaic key if it exists
            if 'mosaic_key' in response['Item']:
                user_record['mosaic_key'] = response['Item']['mosaic_key']
        else:
            # New user
            user_record['created_at'] = timestamp
        
        # Save to the users table
        users_table.put_item(Item=user_record)
        
        print(f"👤 [USER] Updated user record for {user_id}")
        
        return interaction_id
    
    except Exception as e:
        print(f"Error saving user interaction: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/user_interactions', methods=['GET'])
def get_user_interactions():
    """
    API endpoint to get a user's interactions
    """
    try:
        # Get query parameters
        user_id = request.args.get('user_id')
        email = request.args.get('email')
        
        if not user_id and not email:
            return jsonify({
                'error': 'Either user_id or email parameter is required'
            }), 400
        
        # Create DynamoDB resource
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Get the interactions table
        interactions_table_name = os.getenv('REACT_APP_DYNAMODB_INTERACTIONS_TABLE', 'eternity-mirror-interactions')
        interactions_table = dynamodb.Table(interactions_table_name)
        
        interactions = []
        
        if user_id:
            # Query by user_id (using the hash key)
            response = interactions_table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id),
                ScanIndexForward=False  # Sort by sort key (interaction_id) in descending order
            )
            interactions = response.get('Items', [])
        elif email:
            # Scan for email (less efficient, but necessary if only email is provided)
            # Note: In production, consider adding a GSI on email for better performance
            response = interactions_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('email').eq(email)
            )
            interactions = response.get('Items', [])
        
        # Sort by timestamp (newest first)
        interactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'user_id': user_id,
            'email': email,
            'interaction_count': len(interactions),
            'interactions': interactions
        })
    
    except Exception as e:
        logging.error(f"Error getting user interactions: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def get_user_by_email(email):
    """
    Retrieve a user's record from DynamoDB by email
    
    Args:
        email (str): The email address to search for
        
    Returns:
        dict: The user record if found, None otherwise
    """
    if not email:
        logging.warning("Cannot get user: No email provided")
        return None
        
    try:
        # Create DynamoDB resource
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        # Get the users table
        users_table_name = os.getenv('REACT_APP_DYNAMODB_TABLE', 'eternity-mirror-users')
        users_table = dynamodb.Table(users_table_name)
        
        # Scan the table for the email (note: this is not efficient for large tables)
        # In production, consider adding a GSI on email for better performance
        response = users_table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('email').eq(email)
        )
        
        items = response.get('Items', [])
        
        if items:
            # Return the most recently updated user record if multiple exist
            items.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            return items[0]
        else:
            logging.info(f"No user found with email: {email}")
            return None
            
    except Exception as e:
        logging.error(f"Error getting user by email: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return None

@app.route('/user_profile', methods=['GET'])
def get_user_profile():
    """
    Endpoint to get a user's profile information by email
    
    Query parameters:
    - email: The email address of the user to retrieve
    
    Returns:
        JSON with user profile data or an error message
    """
    try:
        # Get the email from the request parameters
        email = request.args.get('email')
        
        if not email:
            return jsonify({'error': 'Email parameter is required'}), 400
            
        # Get the user by email
        user = get_user_by_email(email)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        # Remove sensitive information if needed
        if 'password' in user:
            del user['password']
            
        return jsonify({'user': user}), 200
        
    except Exception as e:
        logging.error(f"Error retrieving user profile: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return jsonify({'error': f'Error retrieving user profile: {str(e)}'}), 500

# Initialize Gradio client for background removal
try:
    background_removal_client = Client("not-lain/background-removal")
    print("Background removal client initialized successfully")
except Exception as e:
    print(f"Error initializing background removal client: {e}")
    background_removal_client = None

@app.route('/remove_background', methods=['POST'])
def remove_background():
    """
    Endpoint to remove the background from an image stored in S3.
    Receives the S3 key of the raw image and returns the S3 key of the processed image.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        image_key = data.get('imageKey')
        session_id = data.get('sessionId')
        
        if not image_key or not session_id:
            return jsonify({"status": "error", "message": "Missing imageKey or sessionId"}), 400
        
        print(f"Removing background for image: {image_key}, session: {session_id}")
        
        # Generate a pre-signed URL to access the image
        image_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': image_key
            },
            ExpiresIn=3600
        )
        
        # Get the file extension and original filename base
        file_name = os.path.basename(image_key)
        file_name_base, _ = os.path.splitext(file_name)
        output_file_name = f"{file_name_base}.jpg"
        
        # Process the image with the background removal API
        if background_removal_client:
            # Call the Gradio API to remove the background
            result = background_removal_client.predict(
                image=handle_file(image_url),
                api_name="/image"
            )
            
            # Process the result (which should be a path to a temporary WEBP file)
            if isinstance(result, list) and len(result) > 0:
                temp_file_path = result[0]
                
                # Open the resulting image with PIL
                if temp_file_path.startswith(('http://', 'https://')):
                    response = requests.get(temp_file_path)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(temp_file_path)
                
                # Convert to RGB if needed (in case it's RGBA)
                if img.mode in ('RGBA', 'LA'):
                    # Create a white background
                    white_bg = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        white_bg.paste(img, mask=img.split()[3])
                    else:
                        white_bg.paste(img, mask=img.split()[1])
                    img = white_bg
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create a BytesIO object to store the JPEG
                img_byte_arr = BytesIO()
                # Save as JPEG with high quality
                img.save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr.seek(0)
                
                # Define the output S3 key
                output_key = f"selected-images/{session_id}/{output_file_name}"
                
                # Upload to S3
                s3_client.upload_fileobj(
                    img_byte_arr,
                    S3_BUCKET,
                    output_key,
                    ExtraArgs={'ContentType': 'image/jpeg'}
                )
                
                print(f"Successfully uploaded processed image to S3: {output_key}")
                
                # Clean up temporary file if it's a local file
                if not temp_file_path.startswith(('http://', 'https://')) and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        print(f"Error removing temporary file: {e}")
                
                return jsonify({
                    "status": "success",
                    "newKey": output_key
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Unexpected result format from background removal API: {result}"
                }), 500
        else:
            # Fallback if background removal client is not available - just copy the image
            original_image_response = requests.get(image_url)
            img = Image.open(BytesIO(original_image_response.content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create a BytesIO object to store the JPEG
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Define the output S3 key
            output_key = f"selected-images/{session_id}/{output_file_name}"
            
            # Upload to S3
            s3_client.upload_fileobj(
                img_byte_arr,
                S3_BUCKET,
                output_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            print(f"Warning: Background removal not available. Copied original image to: {output_key}")
            return jsonify({
                "status": "success",
                "newKey": output_key,
                "warning": "Background removal not available, original image was used."
            })
    
    except Exception as e:
        print(f"Error in background removal: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    

"""Refactoring AWS Service"""
@app.route('/list-session-images', methods=['POST'])
def list_session_images():
    """
    Endpoint to list all images for a given session ID.
    Receives the session ID and returns a list of image keys.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        # Define the prefix for the S3 objects
        prefix = f"raw-captures/{session_id}/"
        
        # Set up the parameters for listing objects
        params = {
            'Bucket': S3_BUCKET,
            'Prefix': prefix
        }
        
        # List objects in the S3 bucket
        response = s3_client.list_objects_v2(**params)
        
        # Check if there are any images
        if 'Contents' not in response or not response['Contents']:
            return jsonify({"status": "success", "images": []})
        
        # Process each image and generate signed URLs
        images = []
        for obj in response['Contents']:
            # Generate a signed URL for viewing the image
            view_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': obj['Key']
                },
                ExpiresIn=3600  # URL expires in 1 hour
            )
            
            # Add the image data to the list
            images.append({
                'key': obj['Key'],
                'url': view_url,
                'lastModified': obj['LastModified'].isoformat()  # Convert datetime to string
            })
        
        return jsonify({"status": "success", "images": images})
    
    except Exception as e:
        print(f"Error listing session images: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    

@app.route('/get_view_url', methods=['POST'])
def get_view_url():
    """
    Endpoint to get a signed URL for viewing an image stored in S3.
    Receives the S3 key of the image and returns the signed URL.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        key = data.get('key')
        force_download = data.get('forceDownload', False)
        
        if not key:
            return jsonify({"status": "error", "message": "Missing key parameter"}), 400
        
        # Remove any leading slashes from the key
        sanitized_key = key[1:] if key.startswith('/') else key
        
        # Check if the key inadvertently includes the bucket name and remove it if needed
        bucket_name_prefix = f"{S3_BUCKET}/"
        if sanitized_key.startswith(bucket_name_prefix):
            print(f"Key contains bucket name prefix ({bucket_name_prefix}), removing it")
            sanitized_key = sanitized_key[len(bucket_name_prefix):]
        
        # Create parameters object for S3 generate_presigned_url
        params = {
            'Bucket': S3_BUCKET,
            'Key': sanitized_key
        }
        
        # Add response-content-disposition parameter for download if requested
        if force_download:
            params['ResponseContentDisposition'] = f'attachment; filename="{sanitized_key.split("/")[-1]}"'
        
        print(f"Generating pre-signed URL for S3 key: {sanitized_key}")
        
        # Generate a pre-signed URL for viewing
        # ExpiresIn is a separate parameter, not part of Params
        url = s3_client.generate_presigned_url(
            'get_object', 
            Params=params,
            ExpiresIn=604800  # URL valid for 7 days (maximum allowed by AWS SigV4)
        )
        
        return jsonify({
            "status": "success",
            "url": url
        })
        
    except Exception as e:
        print(f"Error generating view URL: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/select_image', methods=['POST'])
def select_image():
    """
    Endpoint to select an image for a given session ID.
    Receives the session ID and the image key and returns the selected image.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        new_key = data.get('newKey')
        
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        if not new_key:
            return jsonify({"status": "error", "message": "Missing newKey"}), 400
            
        # List all images in the raw-captures folder for this session
        prefix = f"raw-captures/{session_id}/"
        list_params = {
            'Bucket': S3_BUCKET,
            'Prefix': prefix
        }
        
        try:
            list_response = s3_client.list_objects_v2(**list_params)
            
            # Delete all images from the raw-captures folder
            if 'Contents' in list_response and list_response['Contents']:
                print(f"Deleting {len(list_response['Contents'])} images from raw-captures folder")
                for obj in list_response['Contents']:
                    delete_params = {
                        'Bucket': S3_BUCKET,
                        'Key': obj['Key']
                    }
                    
                    s3_client.delete_object(**delete_params)
                
                return jsonify({
                    "status": "success",
                    "message": f"Deleted {len(list_response['Contents'])} images from raw-captures folder",
                    "newKey": new_key
                })
            else:
                return jsonify({
                    "status": "success",
                    "message": "No images to delete in raw-captures folder",
                    "newKey": new_key
                })
                
        except Exception as e:
            print(f"Error listing/deleting objects from S3: {str(e)}")
            traceback.print_exc()
            # Still return success with the new key even if deletion fails
            return jsonify({
                "status": "success",
                "message": f"Warning: Failed to delete raw captures: {str(e)}",
                "newKey": new_key
            })
        
    except Exception as e:
        print(f"Error in select_image: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500



@app.route('/get-random-previous-image', methods=['POST'])
def get_random_previous_image():
    """
    Endpoint to get a random previous image from the database.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        current_session_id = data.get('currentSessionId')
        if not current_session_id:
            return jsonify({"status": "error", "message": "Missing currentSessionId"}), 400
        
        # List all objects in the selected-images folder
        params = {
            'Bucket': S3_BUCKET,
            'Prefix': 'selected-images/'
        }
        
        response = s3_client.list_objects_v2(**params)
        
        if 'Contents' not in response or not response['Contents']:
            return jsonify({"status": "success", "image": None})
        
        # Filter out images from the current session
        other_user_images = [obj for obj in response['Contents'] 
                            if f"/{current_session_id}/" not in obj['Key']]
        
        if not other_user_images:
            return jsonify({"status": "success", "image": None})
        
        # Select a random image
        import random
        random_index = random.randint(0, len(other_user_images) - 1)
        random_image_key = other_user_images[random_index]['Key']
        
        # Generate a signed URL for the random image
        view_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': random_image_key
            },
            ExpiresIn=604800  # URL valid for 7 days
        )
        
        return jsonify({
            "status": "success",
            "image": {
                "key": random_image_key,
                "url": view_url
            }
        })
        
    except Exception as e:
        print(f"Error getting random previous image: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500



@app.route('/get_processed_video_url', methods=['POST'])
def get_processed_video_url():
    """
    Endpoint to get a signed URL for viewing a processed video stored in S3.
    Receives the session ID and returns the signed URL.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        # Define the video key
        video_key = f"processed-videos/{session_id}/output.mp4"
        
        try:
            # Check if the video exists
            s3_client.head_object(
                Bucket=S3_BUCKET,
                Key=video_key
            )
            
            # If no exception is raised, the video exists
            # Generate a signed URL for viewing
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': video_key
                },
                ExpiresIn=604800  # URL valid for 7 days
            )
            
            return jsonify({
                "status": "success",
                "url": url
            })
            
        except Exception as e:
            # Video doesn't exist or there was an error accessing it
            print(f"Video not found or access error: {str(e)}")
            return jsonify({
                "status": "success",
                "url": None,
                "message": "Video not found"
            })
            
    except Exception as e:
        print(f"Error getting processed video URL: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/delete_session_raw_captures', methods=['POST'])
def delete_session_raw_captures():
    """
    Endpoint to delete all raw captures for a given session ID.
    Receives the session ID and deletes all raw captures.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        # Get all images in the session folder
        prefix = f"raw-captures/{session_id}/"
        list_params = {
            'Bucket': S3_BUCKET,
            'Prefix': prefix
        }
        
        response = s3_client.list_objects_v2(**list_params)
        
        # Delete all images from the raw-captures folder
        deleted_count = 0
        if 'Contents' in response and response['Contents']:
            print(f"Deleting {len(response['Contents'])} images from raw-captures folder for session {session_id}")
            for obj in response['Contents']:
                delete_params = {
                    'Bucket': S3_BUCKET,
                    'Key': obj['Key']
                }
                
                s3_client.delete_object(**delete_params)
                deleted_count += 1
            
            return jsonify({
                "status": "success",
                "message": f"Deleted {deleted_count} images from raw-captures folder",
                "deletedCount": deleted_count
            })
        else:
            print(f"No images found to delete for session {session_id}")
            return jsonify({
                "status": "success",
                "message": "No images to delete in raw-captures folder",
                "deletedCount": 0
            })
            
    except Exception as e:
        print(f"Error deleting raw captures: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500





@app.route('/check_mosaic_exists', methods=['POST'])
def check_mosaic_exists():
    """
    Endpoint to check if a mosaic exists for a given session ID.
    Receives the session ID and returns True if a mosaic exists, False otherwise.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        # List objects in the mosaics folder for this session
        prefix = f"mosaics/{session_id}/"
        params = {
            'Bucket': S3_BUCKET,
            'Prefix': prefix
        }
        
        print(f"Checking for mosaics in bucket {S3_BUCKET} with prefix {prefix}")
        response = s3_client.list_objects_v2(**params)
        
        # If we found at least one mosaic
        if 'Contents' in response and response['Contents']:
            print(f"Found {len(response['Contents'])} objects in S3 for session {session_id}")
            
            # Sort by last modified to get the most recent one
            sorted_mosaics = sorted(response['Contents'], 
                                    key=lambda x: x['LastModified'], 
                                    reverse=True)
            
            # Log all found objects for debugging
            for index, item in enumerate(sorted_mosaics):
                print(f"S3 Object {index}: {item['Key']}, Modified: {item['LastModified']}")
            
            # Find the full-size mosaic
            mosaic_files = [item for item in sorted_mosaics 
                            if 'thumbnail' not in item['Key'] and 'mosaic_' in item['Key']]
            
            # Find JPEG thumbnails
            jpeg_thumbnails = [item for item in sorted_mosaics 
                              if 'thumbnail' in item['Key'] and item['Key'].endswith('.jpg')]
            
            # Find WebP thumbnails
            webp_thumbnails = [item for item in sorted_mosaics 
                              if 'thumbnail' in item['Key'] and item['Key'].endswith('.webp')]
            
            print(f"Found {len(mosaic_files)} full mosaics, {len(jpeg_thumbnails)} JPEG thumbnails, and {len(webp_thumbnails)} WebP thumbnails")
            
            if len(mosaic_files) == 0:
                print('No full mosaic files found')
                
                # If we only have thumbnails but no full mosaics, still return the thumbnail
                # Prefer JPEG thumbnails first, then WebP if no JPEG available
                if jpeg_thumbnails or webp_thumbnails:
                    print('Returning thumbnail-only result')
                    
                    # Prefer JPEG thumbnails first, but use WebP if only that's available
                    thumbnail_key = jpeg_thumbnails[0]['Key'] if jpeg_thumbnails else webp_thumbnails[0]['Key']
                    thumbnail_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': S3_BUCKET,
                            'Key': thumbnail_key
                        },
                        ExpiresIn=604800  # URL valid for 7 days
                    )
                    
                    # Look for a matching WebP version if we have a JPEG
                    webp_thumbnail_url = None
                    if jpeg_thumbnails and webp_thumbnails:
                        jpeg_timestamp = thumbnail_key.split('thumbnail_')[1].split('.')[0] if 'thumbnail_' in thumbnail_key else None
                        if jpeg_timestamp:
                            matching_webp = next((item for item in webp_thumbnails if jpeg_timestamp in item['Key']), None)
                            
                            if matching_webp:
                                webp_thumbnail_url = s3_client.generate_presigned_url(
                                    'get_object',
                                    Params={
                                        'Bucket': S3_BUCKET,
                                        'Key': matching_webp['Key']
                                    },
                                    ExpiresIn=604800  # URL valid for 7 days
                                )
                                print(f"Found matching WebP thumbnail: {matching_webp['Key']}")
                    
                    # Create a downloadable URL from the thumbnail too since we don't have the full image
                    download_url = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': S3_BUCKET,
                            'Key': thumbnail_key,
                            'ResponseContentDisposition': f'attachment; filename="{thumbnail_key.split("/")[-1]}"'
                        },
                        ExpiresIn=604800  # URL valid for 7 days
                    )
                    
                    return jsonify({
                        "status": "success",
                        "result": {
                            "success": True,
                            "mosaicKey": thumbnail_key,  # Use the thumbnail as the main key
                            "mosaicUrl": thumbnail_url,  # Use the thumbnail as the main URL
                            "downloadUrl": download_url,
                            "thumbnailUrl": thumbnail_url,
                            "webpThumbnailUrl": webp_thumbnail_url,
                            "filename": thumbnail_key.split('/')[-1],
                            "isThumbnailOnly": True
                        }
                    })
                
                return jsonify({"status": "success", "result": None})  # No mosaic exists yet
            
            mosaic_key = mosaic_files[0]['Key']
            
            # Get URLs for the full-size mosaic
            view_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': mosaic_key
                },
                ExpiresIn=604800  # URL valid for 7 days
            )
            
            download_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': mosaic_key,
                    'ResponseContentDisposition': f'attachment; filename="{mosaic_key.split("/")[-1]}"'
                },
                ExpiresIn=604800  # URL valid for 7 days
            )
            
            print(f"Found existing mosaic: {mosaic_key}")
            print(f"View URL: {view_url}")
            print(f"Download URL: {download_url}")
            
            # Check if there's a thumbnail version
            thumbnail_url = None
            thumbnail_key = None
            webp_thumbnail_url = None
            
            if jpeg_thumbnails:
                # Try to find a thumbnail that matches the mosaic timestamp
                mosaic_timestamp = mosaic_key.split('mosaic_')[1].split('.')[0] if 'mosaic_' in mosaic_key else None
                print(f"Looking for thumbnail matching timestamp: {mosaic_timestamp}")
                
                matching_thumbnail = None
                if mosaic_timestamp:
                    matching_thumbnail = next((item for item in jpeg_thumbnails if f'thumbnail_{mosaic_timestamp}' in item['Key']), None)
                
                # If no matching thumbnail found, just use the most recent one
                if not matching_thumbnail:
                    print('No matching thumbnail found, using most recent')
                    matching_thumbnail = jpeg_thumbnails[0]
                
                thumbnail_key = matching_thumbnail['Key']
                thumbnail_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET,
                        'Key': thumbnail_key
                    },
                    ExpiresIn=604800  # URL valid for 7 days
                )
                print(f"Using thumbnail: {thumbnail_key}")
                print(f"Thumbnail URL: {thumbnail_url}")
                
                # Look for matching WebP thumbnail
                if webp_thumbnails:
                    jpeg_timestamp = thumbnail_key.split('thumbnail_')[1].split('.')[0] if 'thumbnail_' in thumbnail_key else None
                    if jpeg_timestamp:
                        matching_webp = next((item for item in webp_thumbnails if jpeg_timestamp in item['Key']), None)
                        
                        if matching_webp:
                            webp_thumbnail_url = s3_client.generate_presigned_url(
                                'get_object',
                                Params={
                                    'Bucket': S3_BUCKET,
                                    'Key': matching_webp['Key']
                                },
                                ExpiresIn=604800  # URL valid for 7 days
                            )
                            print(f"Found matching WebP thumbnail: {matching_webp['Key']}")
            elif webp_thumbnails:
                # If no JPEG thumbnails, use WebP only
                print('No JPEG thumbnails found, using WebP')
                thumbnail_key = webp_thumbnails[0]['Key']
                thumbnail_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET,
                        'Key': thumbnail_key
                    },
                    ExpiresIn=604800  # URL valid for 7 days
                )
                print(f"Using WebP thumbnail: {thumbnail_key}")
            else:
                print('No thumbnails found for this mosaic')
            
            return jsonify({
                "status": "success",
                "result": {
                    "success": True,
                    "mosaicKey": mosaic_key,
                    "mosaicUrl": view_url,
                    "downloadUrl": download_url,
                    "thumbnailKey": thumbnail_key,
                    "thumbnailUrl": thumbnail_url,
                    "webpThumbnailUrl": webp_thumbnail_url,
                    "filename": mosaic_key.split('/')[-1]
                }
            })
        
        print(f"No mosaic objects found for session {session_id}")
        return jsonify({"status": "success", "result": None})  # No mosaic exists yet
        
    except Exception as e:
        print(f"Error checking for existing mosaic: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/trigger_mosaic_creation', methods=['POST'])
def trigger_mosaic_creation():
    """
    Endpoint to trigger mosaic creation for a given session ID.
    Receives the session ID and triggers mosaic creation.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        image_key = data.get('imageKey')
        session_id = data.get('sessionId')
        
        if not image_key:
            return jsonify({"status": "error", "message": "Missing imageKey"}), 400
        
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        # Check for existing mosaics first (for fallback purposes)
        existing_mosaic = None
        try:
            # Use the check_mosaic_exists logic but extract data directly
            prefix = f"mosaics/{session_id}/"
            params = {
                'Bucket': S3_BUCKET,
                'Prefix': prefix
            }
            
            print(f"Checking for existing mosaics in bucket {S3_BUCKET} with prefix {prefix}")
            response = s3_client.list_objects_v2(**params)
            
            # If we found at least one mosaic, process it for fallback purposes
            if 'Contents' in response and response['Contents']:
                print(f"Found {len(response['Contents'])} objects in S3 for session {session_id} (potential fallback)")
                existing_mosaic = True
        except Exception as e:
            print(f"Error checking for existing mosaics (non-critical): {str(e)}")
            # Continue execution - this is just for fallback purposes
        
        # Call the create_mosaic endpoint
        print(f"Making request to create_mosaic for image: {image_key}")
        try:
            # Prepare the request data
            mosaic_request_data = {
                "sourceImageKey": image_key,
                "sessionId": session_id,
                "createNew": True  # Signal to the server that we want a new mosaic even if one exists
            }
            
            # Call the local endpoint directly, bypassing the HTTP layer
            # This is more efficient than making an HTTP request to ourselves
            with app.test_request_context(
                '/create_mosaic',
                method='POST',
                json=mosaic_request_data
            ):
                # Call the create_mosaic function directly
                mosaic_result = generate_mosaic()
                
                # The result is a Response object, extract the data
                if isinstance(mosaic_result, tuple):
                    # This is a tuple of (response, status_code)
                    response_obj = mosaic_result[0]
                    status_code = mosaic_result[1]
                else:
                    # This is just a response object
                    response_obj = mosaic_result
                    status_code = 200
                
                # Get the response data - Flask response object has .get_json() method
                if hasattr(response_obj, 'get_json') and callable(response_obj.get_json):
                    response_data = response_obj.get_json()
                else:
                    # Fallback to parsing the data manually
                    response_data = json.loads(response_obj.data.decode('utf-8'))
                
                if status_code != 200:
                    print(f"Error response from create_mosaic: {status_code}, {response_data}")
                    
                    # If we have an existing mosaic and the creation failed, use it as fallback
                    if existing_mosaic:
                        print("Using existing mosaic as fallback after server error")
                        # Call check_mosaic_exists to get the full mosaic data
                        with app.test_request_context(
                            '/check_mosaic_exists',
                            method='POST',
                            json={"sessionId": session_id}
                        ):
                            fallback_result = check_mosaic_exists()
                            
                            # Parse the fallback result
                            if hasattr(fallback_result, 'get_json') and callable(fallback_result.get_json):
                                fallback_data = fallback_result.get_json()
                            else:
                                fallback_data = json.loads(fallback_result.data.decode('utf-8'))
                            
                            if fallback_data.get("status") == "success" and fallback_data.get("result"):
                                return jsonify({
                                    "status": "success",
                                    "result": fallback_data["result"]
                                })
                    
                    return jsonify({
                        "status": "error",
                        "message": response_data.get("message", "Error creating mosaic")
                    }), status_code
                
                # Process the successful response
                result = {
                    "success": True,
                    "mosaicKey": response_data.get("mosaic_key"),
                    "mosaicUrl": response_data.get("url"),
                    "downloadUrl": response_data.get("url"),  # We'll need to create a download URL
                    "thumbnailUrl": response_data.get("thumbnail_url"),
                    "thumbnailKey": response_data.get("thumbnail_key"),
                    "filename": response_data.get("filename")
                }
                
                # If download URL is not different, create one
                if response_data.get("url") and not result.get("downloadUrl"):
                    # Generate a pre-signed URL for downloading
                    result["downloadUrl"] = s3_client.generate_presigned_url(
                        'get_object',
                        Params={
                            'Bucket': S3_BUCKET,
                            'Key': response_data.get("mosaic_key"),
                            'ResponseContentDisposition': f'attachment; filename="{response_data.get("filename")}"'
                        },
                        ExpiresIn=604800  # URL valid for 7 days
                    )
                
                # If thumbnail URL not in response but thumbnail_key is, generate one
                if not result["thumbnailUrl"] and result["thumbnailKey"]:
                    try:
                        result["thumbnailUrl"] = s3_client.generate_presigned_url(
                            'get_object',
                            Params={
                                'Bucket': S3_BUCKET,
                                'Key': result["thumbnailKey"]
                            },
                            ExpiresIn=604800  # URL valid for 7 days
                        )
                    except Exception as e:
                        print(f"Error generating thumbnail URL: {str(e)}")
                
                # If we still don't have a thumbnail, check for it directly
                if not result["thumbnailUrl"]:
                    print("No thumbnail in response, checking S3 directly")
                    
                    try:
                        # Look for a thumbnail with matching timestamp
                        mosaic_filename = result["mosaicKey"].split('/')[-1]
                        timestamp = mosaic_filename.split('mosaic_')[1].split('.')[0] if 'mosaic_' in mosaic_filename else None
                        
                        if timestamp:
                            thumbnail_prefix = f"mosaics/{session_id}/mosaic_thumbnail_{timestamp}"
                            
                            # List objects with this prefix
                            thumb_response = s3_client.list_objects_v2(
                                Bucket=S3_BUCKET,
                                Prefix=thumbnail_prefix
                            )
                            
                            if 'Contents' in thumb_response and thumb_response['Contents']:
                                result["thumbnailKey"] = thumb_response['Contents'][0]['Key']
                                result["thumbnailUrl"] = s3_client.generate_presigned_url(
                                    'get_object',
                                    Params={
                                        'Bucket': S3_BUCKET,
                                        'Key': result["thumbnailKey"]
                                    },
                                    ExpiresIn=604800  # URL valid for 7 days
                                )
                                print(f"Found matching thumbnail directly in S3: {result['thumbnailKey']}")
                    except Exception as e:
                        print(f"Error looking for thumbnail directly: {str(e)}")
                
                return jsonify({
                    "status": "success",
                    "result": result
                })
            
        except Exception as e:
            print(f"Error creating mosaic: {str(e)}")
            traceback.print_exc()
            
            # Final check if a mosaic exists despite the error
            try:
                # Call check_mosaic_exists to check if a mosaic was created anyway
                with app.test_request_context(
                    '/check_mosaic_exists',
                    method='POST',
                    json={"sessionId": session_id}
                ):
                    check_result = check_mosaic_exists()
                    
                    # Parse the check result
                    if hasattr(check_result, 'get_json') and callable(check_result.get_json):
                        check_data = check_result.get_json()
                    else:
                        check_data = json.loads(check_result.data.decode('utf-8'))
                    
                    if check_data.get("status") == "success" and check_data.get("result"):
                        print("Found mosaic despite creation error, using it")
                        return jsonify({
                            "status": "success",
                            "result": check_data["result"]
                        })
            except Exception as check_e:
                print(f"Error during final mosaic check: {str(check_e)}")
            
            # Use existing mosaic as fallback if available
            if existing_mosaic:
                print("Using existing mosaic as fallback after all errors")
                try:
                    # Call check_mosaic_exists to get the full mosaic data
                    with app.test_request_context(
                        '/check_mosaic_exists',
                        method='POST',
                        json={"sessionId": session_id}
                    ):
                        fallback_result = check_mosaic_exists()
                        
                        # Parse the fallback result
                        if hasattr(fallback_result, 'get_json') and callable(fallback_result.get_json):
                            fallback_data = fallback_result.get_json()
                        else:
                            fallback_data = json.loads(fallback_result.data.decode('utf-8'))
                        
                        if fallback_data.get("status") == "success" and fallback_data.get("result"):
                            return jsonify({
                                "status": "success",
                                "result": fallback_data["result"]
                            })
                except Exception as fallback_e:
                    print(f"Error getting fallback mosaic: {str(fallback_e)}")
            
            return jsonify({
                "status": "error",
                "message": f"Error creating mosaic: {str(e)}"
            }), 500
    
    except Exception as e:
        print(f"Error in trigger_mosaic_creation: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/list_mosaics', methods=['POST'])
def list_mosaics():
    """
    Endpoint to list all mosaics for a specific session.
    Receives the session ID and returns a list of mosaic objects with keys and signed URLs.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        session_id = data.get('sessionId')
        if not session_id:
            return jsonify({"status": "error", "message": "Missing sessionId"}), 400
        
        prefix = f"mosaics/{session_id}/"
        params = {
            'Bucket': S3_BUCKET,
            'Prefix': prefix
        }
        
        print(f"Listing mosaics for session {session_id} with prefix {prefix}")
        response = s3_client.list_objects_v2(**params)
        
        if 'Contents' not in response or not response['Contents']:
            return jsonify({"status": "success", "mosaics": []})
        
        # Sort by last modified to get the most recent first
        sorted_contents = sorted(
            response['Contents'],
            key=lambda x: x['LastModified'],
            reverse=True
        )
        
        # Separate full mosaics from thumbnails
        full_mosaics = [item for item in sorted_contents 
                        if 'thumbnail' not in item['Key'] and 'mosaic_' in item['Key']]
        
        # Separate JPEG thumbnails
        jpeg_thumbnails = [item for item in sorted_contents 
                          if 'thumbnail' in item['Key'] and item['Key'].endswith('.jpg')]
        
        # Also look for WebP thumbnails
        webp_thumbnails = [item for item in sorted_contents 
                          if 'thumbnail' in item['Key'] and item['Key'].endswith('.webp')]
        
        print(f'Found mosaics: {len(full_mosaics)}')
        print(f'Found JPEG thumbnails: {len(jpeg_thumbnails)}')
        print(f'Found WebP thumbnails: {len(webp_thumbnails)}')
        
        # Process each mosaic
        mosaic_results = []
        for mosaic_obj in full_mosaics:
            # Generate signed URL for viewing
            view_url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': S3_BUCKET,
                    'Key': mosaic_obj['Key']
                },
                ExpiresIn=604800  # URL valid for 7 days
            )
            
            # Look for a matching thumbnail
            filename_root = mosaic_obj['Key'].split('mosaic_')[1].split('.')[0] if 'mosaic_' in mosaic_obj['Key'] else None
            print(f'Looking for thumbnail with base: {filename_root}')
            
            # First try to find JPEG thumbnail
            matching_thumbnail = None
            if filename_root:
                for thumb in jpeg_thumbnails:
                    if filename_root in thumb['Key']:
                        matching_thumbnail = thumb
                        break
            
            # If no exact match found, just use the most recent JPEG thumbnail
            if not matching_thumbnail and jpeg_thumbnails:
                print('No exact JPEG thumbnail match found, using most recent')
                matching_thumbnail = jpeg_thumbnails[0]
            
            thumbnail_url = None
            webp_thumbnail_url = None
            
            if matching_thumbnail:
                # Generate signed URL for thumbnail
                thumbnail_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET,
                        'Key': matching_thumbnail['Key']
                    },
                    ExpiresIn=604800  # URL valid for 7 days
                )
                print(f'Using JPEG thumbnail: {matching_thumbnail["Key"]}')
                
                # Check if there's a matching WebP version
                jpeg_timestamp = matching_thumbnail['Key'].split('thumbnail_')[1].split('.')[0] if 'thumbnail_' in matching_thumbnail['Key'] else None
                if jpeg_timestamp and webp_thumbnails:
                    matching_webp = None
                    for webp in webp_thumbnails:
                        if jpeg_timestamp in webp['Key']:
                            matching_webp = webp
                            break
                    
                    if matching_webp:
                        # Generate signed URL for WebP thumbnail
                        webp_thumbnail_url = s3_client.generate_presigned_url(
                            'get_object',
                            Params={
                                'Bucket': S3_BUCKET,
                                'Key': matching_webp['Key']
                            },
                            ExpiresIn=604800  # URL valid for 7 days
                        )
                        print(f'Found matching WebP thumbnail: {matching_webp["Key"]}')
            else:
                print(f'No thumbnail found for: {mosaic_obj["Key"]}')
            
            # Add this mosaic to the results
            mosaic_results.append({
                'key': mosaic_obj['Key'],
                'url': view_url,
                'thumbnailUrl': thumbnail_url,
                'webpThumbnailUrl': webp_thumbnail_url,
                'lastModified': mosaic_obj['LastModified'].isoformat()  # Convert datetime to string
            })
        
        return jsonify({
            "status": "success",
            "mosaics": mosaic_results
        })
        
    except Exception as e:
        print(f"Error listing mosaics: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# End of Refactoring AWS Service

# Refactoring DynamoDB Service

@app.route('/save_user', methods=['POST'])
def save_user():
    """
    Endpoint to save a user's information to DynamoDB.
    Receives the user ID, email, and selected image key.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        user_id = data.get('userId')
        email = data.get('email')
        selected_image_key = data.get('selectedImageKey')
        
        if not user_id:
            return jsonify({"status": "error", "message": "Missing userId"}), 400
        
        if not email:
            return jsonify({"status": "error", "message": "Missing email"}), 400
        
        # selectedImageKey is optional, but if provided, it should be a string
        if selected_image_key is not None and not isinstance(selected_image_key, str):
            return jsonify({"status": "error", "message": "Invalid selectedImageKey format"}), 400
        
        # Prepare the item for DynamoDB
        item = {
            'user_id': {'S': user_id},
            'email': {'S': email},
            'created_at': {'S': datetime.now().isoformat()},
            'updated_at': {'S': datetime.now().isoformat()}
        }
        
        # Only add selected_image_key if it exists
        if selected_image_key:
            item['selected_image_key'] = {'S': selected_image_key}
        
        # Save to DynamoDB
        response = dynamodb_client.put_item(
            TableName=DYNAMODB_TABLE,
            Item=item
        )
        
        print(f"Saved user to DynamoDB: {user_id}, {email}")
        
        # Return the created/updated user record
        return jsonify({
            "status": "success",
            "user": {
                "user_id": user_id,
                "email": email,
                "selected_image_key": selected_image_key
            }
        })
        
    except Exception as e:
        print(f"Error saving user to DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/get_user', methods=['POST'])
def get_user():
    """
    Endpoint to get a user's information from DynamoDB.
    Receives the user ID and returns the user's information.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        user_id = data.get('userId')
        if not user_id:
            return jsonify({"status": "error", "message": "Missing userId"}), 400
        
        # Get user from DynamoDB
        response = dynamodb_client.get_item(
            TableName=DYNAMODB_TABLE,
            Key={
                'user_id': {'S': user_id}
            }
        )
        
        # Check if user exists
        if 'Item' not in response:
            return jsonify({"status": "success", "user": None})
        
        # Convert DynamoDB format to regular JSON
        user_item = response['Item']
        user = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in user_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                user[key] = value['S']
            elif 'N' in value:  # Number
                user[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                user[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                user[key] = None
            elif 'L' in value:  # List
                user[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                user[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Retrieved user from DynamoDB: {user_id}")
        
        return jsonify({
            "status": "success",
            "user": user
        })
        
    except Exception as e:
        print(f"Error getting user from DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/get_user_by_email', methods=['POST'])
def get_user_by_email():
    """
    Endpoint to get a user's information from DynamoDB by email.
    Receives the email and returns the user's information.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        email = data.get('email')
        if not email:
            return jsonify({"status": "error", "message": "Missing email"}), 400
        
        # In production, you would use a GSI (Global Secondary Index) for better performance
        # For now, we'll use scan with a filter expression
        response = dynamodb_client.scan(
            TableName=DYNAMODB_TABLE,
            FilterExpression="email = :email",
            ExpressionAttributeValues={
                ":email": {"S": email}
            }
        )
        
        # Check if user exists
        if 'Items' not in response or not response['Items']:
            return jsonify({"status": "success", "user": None})
        
        # Take the first matching user
        user_item = response['Items'][0]
        user = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in user_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                user[key] = value['S']
            elif 'N' in value:  # Number
                user[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                user[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                user[key] = None
            elif 'L' in value:  # List
                user[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                user[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Retrieved user from DynamoDB by email: {email}")
        
        return jsonify({
            "status": "success",
            "user": user
        })
        
    except Exception as e:
        print(f"Error getting user by email from DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    


@app.route('/update_mosaic_key', methods=['POST'])
def update_mosaic_key():
    """
    Endpoint to update the mosaic key for a user.
    Receives the user ID and mosaic key.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        user_id = data.get('userId')
        mosaic_key = data.get('mosaicKey')
        
        if not user_id:
            return jsonify({"status": "error", "message": "Missing userId"}), 400
        
        if not mosaic_key:
            return jsonify({"status": "error", "message": "Missing mosaicKey"}), 400
        
        # Update the user record in DynamoDB
        response = dynamodb_client.update_item(
            TableName=DYNAMODB_TABLE,
            Key={
                'user_id': {'S': user_id}
            },
            UpdateExpression="SET mosaic_key = :mosaicKey, updated_at = :updatedAt",
            ExpressionAttributeValues={
                ':mosaicKey': {'S': mosaic_key},
                ':updatedAt': {'S': datetime.now().isoformat()}
            },
            ReturnValues="ALL_NEW"  # Return the updated item
        )
        
        # Check if update was successful
        if 'Attributes' not in response:
            return jsonify({
                "status": "error",
                "message": "Failed to update user, no attributes returned"
            }), 500
        
        # Convert DynamoDB format to regular JSON
        user_item = response['Attributes']
        user = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in user_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                user[key] = value['S']
            elif 'N' in value:  # Number
                user[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                user[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                user[key] = None
            elif 'L' in value:  # List
                user[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                user[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Updated mosaic key for user {user_id} to {mosaic_key}")
        
        return jsonify({
            "status": "success",
            "user": user
        })
        
    except Exception as e:
        print(f"Error updating mosaic key in DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    



@app.route('/save_blend', methods=['POST'])
def save_blend():
    """
    Endpoint to save a blend's information to DynamoDB.
    Receives the blend data object with blend ID, email, etc.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        blend_data = data.get('blendData')
        if not blend_data:
            return jsonify({"status": "error", "message": "Missing blendData"}), 400
        
        # Ensure required fields are present
        if 'blend_id' not in blend_data:
            return jsonify({"status": "error", "message": "Missing blend_id in blendData"}), 400
        
        # Add timestamps
        current_time = datetime.now().isoformat()
        if 'created_at' not in blend_data:
            blend_data['created_at'] = current_time
        blend_data['updated_at'] = current_time
        
        # Convert the blend data to DynamoDB format
        item = {}
        for key, value in blend_data.items():
            if value is None:
                item[key] = {'NULL': True}
            elif isinstance(value, str):
                item[key] = {'S': value}
            elif isinstance(value, (int, float)):
                item[key] = {'N': str(value)}
            elif isinstance(value, bool):
                item[key] = {'BOOL': value}
            elif isinstance(value, list):
                # For simplicity, assuming list of strings
                item[key] = {'L': [{'S': item} for item in value]}
            elif isinstance(value, dict):
                # For simplicity, assuming dict with string values
                item[key] = {'M': {k: {'S': v} if isinstance(v, str) else {'N': str(v)} for k, v in value.items()}}
            else:
                item[key] = {'S': str(value)}  # Default to string for complex types
        
        # Save to DynamoDB
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.put_item(
            TableName=blends_table_name,
            Item=item
        )
        
        print(f"Saved blend to DynamoDB: {blend_data.get('blend_id')}")
        
        # Return the created blend record
        return jsonify({
            "status": "success",
            "blend": blend_data
        })
        
    except Exception as e:
        print(f"Error saving blend to DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get_blend', methods=['POST'])
def get_blend():
    """
    Endpoint to get a blend's information from DynamoDB.
    Receives the blend ID and returns the blend data.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        blend_id = data.get('blendId')
        if not blend_id:
            return jsonify({"status": "error", "message": "Missing blendId"}), 400
        
        # Get the blend from DynamoDB
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.get_item(
            TableName=blends_table_name,
            Key={
                'blend_id': {'S': blend_id}
            }
        )
        
        # Check if blend exists
        if 'Item' not in response:
            return jsonify({"status": "success", "blend": None})
        
        # Convert DynamoDB format to regular JSON
        blend_item = response['Item']
        blend = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in blend_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                blend[key] = value['S']
            elif 'N' in value:  # Number
                blend[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                blend[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                blend[key] = None
            elif 'L' in value:  # List
                blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Retrieved blend from DynamoDB: {blend_id}")
        
        return jsonify({
            "status": "success",
            "blend": blend
        })
        
    except Exception as e:
        print(f"Error getting blend from DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
@app.route('/update_blend_status', methods=['POST'])
def update_blend_status():
    """
    Endpoint to update the status of a blend in DynamoDB.
    Receives the blend ID and status.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Extract all possible parameters
        blend_id = data.get('blendId')
        status_value = data.get('status')
        blend_url = data.get('blendUrl')
        error_msg = data.get('errorMsg')
        flora_video_url = data.get('floraVideoUrl')
        s3_key = data.get('s3Key')
        
        if not blend_id:
            return jsonify({"status": "error", "message": "Missing blendId"}), 400
        
        if status_value is None:
            return jsonify({"status": "error", "message": "Missing status"}), 400
        
        # Start building the update expression
        update_expression = "SET #statusAttr = :status"
        expression_attribute_values = {
            ':status': {'S': status_value}
        }
        
        # Create expression attribute names
        expression_attribute_names = {
            '#statusAttr': 'status'
        }
        
        # Add blendUrl attribute if provided
        if blend_url is not None:
            if isinstance(blend_url, dict):
                # Handle object with URLs
                for key, url in blend_url.items():
                    if url is not None:
                        # Handle reserved keywords in attribute names if needed
                        if key == 'status':
                            # 'status' is already handled as a reserved keyword
                            expression_attribute_values[':statusValue'] = {'S': url}
                            update_expression += ', #statusAttr = :statusValue'
                        else:
                            # Create a placeholder for each attribute name
                            attr_name_placeholder = f"#{key.replace('.', '_').replace('-', '_')}"
                            attr_value_placeholder = f":{key.replace('.', '_').replace('-', '_')}"
                            
                            expression_attribute_names[attr_name_placeholder] = key
                            
                            # Determine the type for the value
                            if isinstance(url, (int, float)):
                                expression_attribute_values[attr_value_placeholder] = {'N': str(url)}
                            elif isinstance(url, bool):
                                expression_attribute_values[attr_value_placeholder] = {'BOOL': url}
                            else:
                                expression_attribute_values[attr_value_placeholder] = {'S': str(url)}
                            
                            update_expression += f', {attr_name_placeholder} = {attr_value_placeholder}'
            else:
                # Handle string URL
                update_expression += ', blend_url = :blendUrl'
                expression_attribute_values[':blendUrl'] = {'S': str(blend_url)}
        
        # Add error_msg attribute if provided
        if error_msg is not None:
            update_expression += ', error_message = :errorMsg'
            expression_attribute_values[':errorMsg'] = {'S': str(error_msg)}
        
        # Add flora_video_url attribute if provided
        if flora_video_url is not None:
            update_expression += ', flora_video_url = :floraVideoUrl'
            expression_attribute_values[':floraVideoUrl'] = {'S': str(flora_video_url)}
        
        # Add s3_key attribute if provided
        if s3_key is not None:
            update_expression += ', s3_video_key = :s3Key'
            expression_attribute_values[':s3Key'] = {'S': str(s3_key)}
        
        # Add updated_at timestamp
        update_expression += ', updated_at = :updatedAt'
        expression_attribute_values[':updatedAt'] = {'S': datetime.now().isoformat()}
        
        # Log the update details for debugging
        print(f"Update expression: {update_expression}")
        print(f"Expression attribute values: {expression_attribute_values}")
        print(f"Expression attribute names: {expression_attribute_names}")
        
        # Update the blend in DynamoDB
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.update_item(
            TableName=blends_table_name,
            Key={
                'blend_id': {'S': blend_id}
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames=expression_attribute_names,
            ReturnValues="ALL_NEW"  # Return the updated item
        )
        
        # Check if update was successful
        if 'Attributes' not in response:
            return jsonify({
                "status": "error",
                "message": "Failed to update blend, no attributes returned"
            }), 500
        
        # Convert DynamoDB format to regular JSON
        blend_item = response['Attributes']
        blend = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in blend_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                blend[key] = value['S']
            elif 'N' in value:  # Number
                blend[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                blend[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                blend[key] = None
            elif 'L' in value:  # List
                blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Updated blend status in DynamoDB: {blend_id} to {status_value}")
        
        return jsonify({
            "status": "success",
            "blend": blend
        })
        
    except Exception as e:
        print(f"Error updating blend status in DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/get_all_blends', methods=['POST'])
def get_all_blends():
    """
    Endpoint to get all blends from DynamoDB.
    Returns a list of blend objects sorted by creation date (newest first).
    """
    try:
        # Get all blends from DynamoDB using scan
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.scan(
            TableName=blends_table_name
        )
        
        # Check if we found any blends
        if 'Items' not in response or not response['Items']:
            return jsonify({
                "status": "success",
                "blends": []
            })
        
        # Convert DynamoDB format to regular JSON
        blends = []
        for blend_item in response['Items']:
            blend = {}
            
            # Process each attribute based on its DynamoDB type
            for key, value in blend_item.items():
                # Handle different DynamoDB types
                if 'S' in value:  # String
                    blend[key] = value['S']
                elif 'N' in value:  # Number
                    blend[key] = float(value['N'])
                elif 'BOOL' in value:  # Boolean
                    blend[key] = value['BOOL']
                elif 'NULL' in value:  # Null
                    blend[key] = None
                elif 'L' in value:  # List
                    blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
                elif 'M' in value:  # Map
                    # For simplicity, we're just extracting string values
                    blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
                # Add more type handlers as needed
            
            blends.append(blend)
        
        # Sort by created_at, newest first
        blends.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        print(f"Retrieved {len(blends)} blends from DynamoDB")
        
        return jsonify({
            "status": "success",
            "blends": blends
        })
        
    except Exception as e:
        print(f"Error getting all blends from DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    

@app.route('/get_blends_by_email', methods=['POST'])
def get_blends_by_email():
    """
    Endpoint to get a blend by email from DynamoDB.
    Receives the email and returns a blend object.
    """
    try:
        # Get the email from the request body
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({
                "status": "error",
                "message": "Email is required"
            }), 400
        
        # Get blends from DynamoDB using scan with filter
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.scan(
            TableName=blends_table_name,
            FilterExpression='email = :email',
            ExpressionAttributeValues={
                ':email': {'S': email}
            }
        )
        
        # Check if we found any blends
        if 'Items' not in response or not response['Items']:
            return jsonify({
                "status": "success",
                "blends": []
            })
        
        # Convert DynamoDB format to regular JSON
        blends = []
        for blend_item in response['Items']:
            blend = {}
            
            # Process each attribute based on its DynamoDB type
            for key, value in blend_item.items():
                # Handle different DynamoDB types
                if 'S' in value:  # String
                    blend[key] = value['S']
                elif 'N' in value:  # Number
                    blend[key] = float(value['N'])
                elif 'BOOL' in value:  # Boolean
                    blend[key] = value['BOOL']
                elif 'NULL' in value:  # Null
                    blend[key] = None
                elif 'L' in value:  # List
                    blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
                elif 'M' in value:  # Map
                    # For simplicity, we're just extracting string values
                    blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
                # Add more type handlers as needed
            
            blends.append(blend)
        
        # Sort by created_at, newest first
        blends.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        print(f"Retrieved {len(blends)} blends for email {email} from DynamoDB")
        
        return jsonify({
            "status": "success",
            "blends": blends
        })
        
    except Exception as e:
        print(f"Error getting blends by email from DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/update_blend_source_image_key', methods=['POST'])
def update_blend_source_image_key():
    """
    Endpoint to update the source image key for a blend.
    Receives the blend ID and source image key.
    """
    try:
        # Get data from the request body
        data = request.json
        blend_id = data.get('blendId')
        source_image_key = data.get('sourceImageKey')
        
        if not blend_id or not source_image_key:
            return jsonify({
                "status": "error",
                "message": "Blend ID and source image key are required"
            }), 400
        
        # Update blend in DynamoDB
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.update_item(
            TableName=blends_table_name,
            Key={
                'blend_id': {'S': blend_id}
            },
            UpdateExpression='SET source_image_key = :sourceImageKey, updated_at = :updatedAt',
            ExpressionAttributeValues={
                ':sourceImageKey': {'S': source_image_key},
                ':updatedAt': {'S': datetime.now().isoformat()}
            },
            ReturnValues="ALL_NEW"
        )
        
        # Check if update was successful
        if 'Attributes' not in response:
            return jsonify({
                "status": "error",
                "message": "Failed to update blend source image key, no attributes returned"
            }), 500
        
        # Convert DynamoDB format to regular JSON
        blend_item = response['Attributes']
        blend = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in blend_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                blend[key] = value['S']
            elif 'N' in value:  # Number
                blend[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                blend[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                blend[key] = None
            elif 'L' in value:  # List
                blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Updated blend source image key in DynamoDB: {blend_id} to {source_image_key}")
        
        return jsonify({
            "status": "success",
            "blend": blend
        })
        
    except Exception as e:
        print(f"Error updating blend source image key in DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/update_blend_random_image_key', methods=['POST'])
def update_blend_random_image_key():
    """
    Endpoint to update the random image key for a blend.
    Receives the blend ID and random image key.
    """
    try:
        # Get data from the request body
        data = request.json
        blend_id = data.get('blendId')
        random_image_key = data.get('randomImageKey')
        
        if not blend_id or not random_image_key:
            return jsonify({
                "status": "error",
                "message": "Blend ID and random image key are required"
            }), 400
        
        # Update blend in DynamoDB
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        response = dynamodb_client.update_item(
            TableName=blends_table_name,
            Key={
                'blend_id': {'S': blend_id}
            },
            UpdateExpression='SET random_image_key = :randomImageKey, updated_at = :updatedAt',
            ExpressionAttributeValues={
                ':randomImageKey': {'S': random_image_key},
                ':updatedAt': {'S': datetime.now().isoformat()}
            },
            ReturnValues="ALL_NEW"
        )
        
        # Check if update was successful
        if 'Attributes' not in response:
            return jsonify({
                "status": "error",
                "message": "Failed to update blend random image key, no attributes returned"
            }), 500
        
        # Convert DynamoDB format to regular JSON
        blend_item = response['Attributes']
        blend = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in blend_item.items():
            # Handle different DynamoDB types
            if 'S' in value:  # String
                blend[key] = value['S']
            elif 'N' in value:  # Number
                blend[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                blend[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                blend[key] = None
            elif 'L' in value:  # List
                blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                # For simplicity, we're just extracting string values
                blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        print(f"Updated blend random image key in DynamoDB: {blend_id} to {random_image_key}")
        
        return jsonify({
            "status": "success",
            "blend": blend
        })
        
    except Exception as e:
        print(f"Error updating blend random image key in DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/update_blend_keys', methods=['POST'])
def update_blend_keys():
    """
    Endpoint to update the keys for a blend.
    Receives the blend ID and keys.
    """
    try:
        # Get data from the request body
        data = request.json
        blend_id = data.get('blendId')
        keys = data.get('keys')
        
        if not blend_id or not keys:
            return jsonify({
                "status": "error",
                "message": "Blend ID and keys are required"
            }), 400
        
        # First get the current blend
        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        get_response = dynamodb_client.get_item(
            TableName=blends_table_name,
            Key={
                'blend_id': {'S': blend_id}
            }
        )
        
        if 'Item' not in get_response:
            return jsonify({
                "status": "error",
                "message": f"Blend not found: {blend_id}"
            }), 404
        
        # Convert DynamoDB format to regular format
        current_blend_item = get_response['Item']
        current_blend = {}
        
        # Process each attribute based on its DynamoDB type
        for key, value in current_blend_item.items():
            if 'S' in value:  # String
                current_blend[key] = value['S']
            elif 'N' in value:  # Number
                current_blend[key] = float(value['N'])
            elif 'BOOL' in value:  # Boolean
                current_blend[key] = value['BOOL']
            elif 'NULL' in value:  # Null
                current_blend[key] = None
            elif 'L' in value:  # List
                current_blend[key] = [item.get('S', item.get('N', item.get('BOOL', None))) for item in value['L']]
            elif 'M' in value:  # Map
                current_blend[key] = {k: v.get('S', v.get('N', v.get('BOOL', None))) for k, v in value['M'].items()}
            # Add more type handlers as needed
        
        # Create an updated blend with the new keys
        updated_blend = {**current_blend, **keys, "updated_at": datetime.now().isoformat()}
        
        # Convert regular Python types to DynamoDB format
        dynamodb_item = {}
        for key, value in updated_blend.items():
            if isinstance(value, str):
                dynamodb_item[key] = {'S': value}
            elif isinstance(value, (int, float)):
                dynamodb_item[key] = {'N': str(value)}
            elif isinstance(value, bool):
                dynamodb_item[key] = {'BOOL': value}
            elif value is None:
                dynamodb_item[key] = {'NULL': True}
            elif isinstance(value, list):
                dynamodb_item[key] = {'L': [{'S': item} if isinstance(item, str) else {'N': str(item)} if isinstance(item, (int, float)) else {'BOOL': item} if isinstance(item, bool) else {'NULL': True} for item in value]}
            elif isinstance(value, dict):
                dynamodb_item[key] = {'M': {k: {'S': v} if isinstance(v, str) else {'N': str(v)} if isinstance(v, (int, float)) else {'BOOL': v} if isinstance(v, bool) else {'NULL': True} for k, v in value.items()}}
            # Add more type handlers as needed
        
        # Store the updated blend in DynamoDB
        put_response = dynamodb_client.put_item(
            TableName=blends_table_name,
            Item=dynamodb_item
        )
        
        print(f"Successfully updated blend keys in DynamoDB for blend {blend_id}")
        
        return jsonify({
            "status": "success",
            "blend": updated_blend
        })
        
    except Exception as e:
        print(f"Error updating blend keys in DynamoDB: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# End of Refactoring DynamoDB Service



# Start of Refactoring Flora Service

@app.route('/get_signed_s3_url', methods=['POST'])
def get_signed_s3_url():
    """
    Endpoint to get a signed URL for an S3 object.
    Receives the S3 object key.
    """
    try:
        # Get the request data
        data = request.get_json()
        if not data or 'key' not in data:
            return jsonify({'status': 'error', 'message': 'Missing required parameter: key'}), 400
        
        key = data['key']
        if not key:
            return jsonify({'status': 'error', 'message': 'Key cannot be empty'}), 400
            
        # Clean up the key if necessary
        clean_key = key[1:] if key.startswith('/') else key
        
        # Remove bucket name if present in the key
        if clean_key.startswith(f"{S3_BUCKET}/"):
            clean_key = clean_key[len(S3_BUCKET) + 1:]
            
        # Generate a pre-signed URL using the existing s3_client
        expires_in = 604800  # URL valid for 7 days (maximum allowed by AWS SigV4)
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': clean_key
            },
            ExpiresIn=expires_in
        )
        
        return jsonify({'status': 'success', 'signedUrl': signed_url})
        
    except Exception as e:
        print(f"Error generating signed URL: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate signed URL: {str(e)}'
        }), 500



@app.route('/upload_video_to_s3', methods=['POST'])
def upload_video_to_s3():
    """
    Endpoint to upload a video to S3 and return a signed URL.
    Receives the video data as binary and the key information.
    """
    try:
        # Check if we received multipart/form-data
        if 'video' not in request.files:
            video_data = request.get_data()
            s3_file_name = request.headers.get('s3-file-name')
            content_type = request.headers.get('Content-Type', 'video/mp4')
        else:
            video_file = request.files['video']
            video_data = video_file.read()
            s3_file_name = request.form.get('s3_file_name')
            content_type = video_file.content_type or 'video/mp4'
            
        # Check required parameters
        if not video_data:
            return jsonify({
                'status': 'error',
                'message': 'No video data provided'
            }), 400
            
        if not s3_file_name:
            return jsonify({
                'status': 'error',
                'message': 'No S3 file name provided'
            }), 400
            
        # Create S3 key
        s3_key = f'blends-video/{s3_file_name}'
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=video_data,
            ContentType=content_type
        )
        
        # Generate a pre-signed URL
        expires_in = 604800  # 7 days
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': s3_key
            },
            ExpiresIn=expires_in
        )
        
        return jsonify({
            'status': 'success',
            'videoUrl': signed_url,
            's3Key': s3_key
        })
        
    except Exception as e:
        print(f"Error uploading video to S3: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Failed to upload video: {str(e)}'
        }), 500






# End of Refactoring Flora Service

@app.route('/get_new_blends', methods=['POST'])
def get_new_blends():
    """
    Endpoint to get blends updated since a specific timestamp.
    Requires a 'since_timestamp' (ISO 8601 format) in the JSON body.
    Uses the GSI-UpdatedAt index.
    """
    print(f"--- Request received for /get_new_blends via {request.method} ---") # Keep or remove debug print
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        since_timestamp = data.get('since_timestamp')
        if not since_timestamp:
            return jsonify({"status": "error", "message": "Missing 'since_timestamp'"}), 400
        
        # Validate timestamp format (basic check)
        try:
            datetime.fromisoformat(since_timestamp.replace('Z', '+00:00')) 
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid 'since_timestamp' format. Use ISO 8601."}), 400

        blends_table_name = os.getenv('REACT_APP_DYNAMODB_BLENDS_TABLE', 'eternity-mirror-blends')
        gsi_name = 'GSI-UpdatedAt' 
        gsi_partition_key_attribute = 'gsi_pk' # Attribute name for GSI partition key
        gsi_sort_key_attribute = 'updated_at'  # Attribute name for GSI sort key
        gsi_partition_key_value = 'ALL_BLENDS'

        print(f"Querying GSI '{gsi_name}' for new blends since: {since_timestamp}")

        # --- CORRECTED QUERY USING LOW-LEVEL CLIENT SYNTAX ---
        response = dynamodb_client.query(
            TableName=blends_table_name,
            IndexName=gsi_name,
            # KeyConditionExpression uses literal placeholders #pk and #sk
            KeyConditionExpression="#pk = :pk_val AND #sk > :sk_val",
            # ExpressionAttributeNames maps the placeholders to actual attribute names
            ExpressionAttributeNames={
                "#pk": gsi_partition_key_attribute, # Map '#pk' placeholder to 'gsi_pk'
                "#sk": gsi_sort_key_attribute     # Map '#sk' placeholder to 'updated_at'
            },
            # ExpressionAttributeValues maps placeholders to actual values with types
            ExpressionAttributeValues={
                ":pk_val": {"S": gsi_partition_key_value}, # Map ':pk_val' placeholder
                ":sk_val": {"S": since_timestamp}         # Map ':sk_val' placeholder
            }
            # ScanIndexForward=False # Optional
        )
        # --- END OF CORRECTION ---

        new_blends = []
        if 'Items' in response:
            # Reuse your existing logic to convert DynamoDB items to JSON
            for blend_item in response['Items']:
                blend = {}
                for key, value in blend_item.items():
                    if 'S' in value: blend[key] = value['S']
                    elif 'N' in value: blend[key] = float(value['N']) # Or int
                    elif 'BOOL' in value: blend[key] = value['BOOL']
                    elif 'NULL' in value: blend[key] = None
                    elif 'L' in value: 
                         blend[key] = [item.get('S') for item in value['L'] if 'S' in item]
                    elif 'M' in value: 
                         blend[key] = {k: v.get('S') for k, v in value['M'].items() if 'S' in v}
                new_blends.append(blend)
        
        new_blends.sort(key=lambda x: x.get('updated_at', ''), reverse=True)

        print(f"Found {len(new_blends)} new or updated blends since {since_timestamp}.")

        return jsonify({
            "status": "success",
            "blends": new_blends
        })

    except dynamodb_client.exceptions.ResourceNotFoundException:
         print(f"Error: Table '{blends_table_name}' or Index '{gsi_name}' not found.")
         return jsonify({
            "status": "error",
            "message": f"Server configuration error: Table or Index not found."
         }), 500
    except Exception as e:
        print(f"Error getting new blends: {str(e)}")
        traceback.print_exc()
        # Send back the specific error message for easier debugging on the client
        return jsonify({
            "status": "error",
            "message": f"An error occurred: {str(e)}" 
        }), 500


#--- end of endpoint ---





if __name__ == '__main__':
    # Start the USB cleanup thread
    start_usb_cleanup_thread()
    
    # Ensure DynamoDB tables exist
    ensure_blend_table_exists()
    ensure_interactions_table_exists()
    
    # Add a cleanup handler for when the server is shutting down
    @atexit.register
    def cleanup_on_exit():
        print("Server shutting down, cleaning up resources...")
        # Stop the USB cleanup thread
        stop_usb_cleanup_thread()
        
        # Clean up any camera resources
        global camera_control
        if camera_control:
            try:
                camera_control.cleanup()
            except:
                pass
            camera_control = None
        
        # Final USB cleanup
        try:
            cleanup_usb_resources()
        except:
            pass
        
        print("Cleanup complete, server shutting down")
    
    # Start the Flask application
    app.run(
        port=5001, 
        debug=True,  
        threaded=True,  # Enable threading for multiple concurrent requests
        # Increase the timeout for long-running operations like mosaic creation
        host='0.0.0.0'  # Make the server accessible from any IP (not just localhost)
    )  # Changed from 5000 to 5001