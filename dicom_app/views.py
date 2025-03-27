import json
import numpy as np
import os
import pydicom
import uuid
import logging
import sys
import subprocess
import shutil
from .forms import DicomUploadForm
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image, ImageDraw
import traceback


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('dicom_app')
logger.setLevel(logging.DEBUG)

# Define any missing UIDs
if not hasattr(pydicom.uid, 'JPEGLS'):
    logger.info("Adding missing JPEG-LS UIDs to pydicom.uid")
    pydicom.uid.JPEGLS = pydicom.uid.UID('1.2.840.10008.1.2.4.80')
    pydicom.uid.JPEGLSLossless = pydicom.uid.UID('1.2.840.10008.1.2.4.80')
    pydicom.uid.JPEGLSNearLossless = pydicom.uid.UID('1.2.840.10008.1.2.4.81')

# Configure pydicom to use all available handlers
from pydicom import config

# Try to import all available handlers
logger.info("Attempting to import and configure DICOM pixel data handlers...")
available_handlers = {}

try:
    from pydicom.pixel_data_handlers import numpy_handler
    available_handlers['numpy'] = numpy_handler
    logger.info("Numpy handler imported successfully")
except ImportError:
    logger.warning("Numpy handler import failed")

try:
    from pydicom.pixel_data_handlers import pillow_handler
    available_handlers['pillow'] = pillow_handler
    logger.info("Pillow handler imported successfully")
except ImportError:
    logger.warning("Pillow handler import failed")

try:
    from pydicom.pixel_data_handlers import pylibjpeg_handler
    if pylibjpeg_handler.is_available():
        available_handlers['pylibjpeg'] = pylibjpeg_handler
        logger.info("pylibjpeg handler imported and available")
    else:
        logger.warning("pylibjpeg handler imported but dependencies missing")
except ImportError:
    logger.warning("pylibjpeg handler import failed")

try:
    from pydicom.pixel_data_handlers import jpeg_ls_handler
    if jpeg_ls_handler.is_available():
        available_handlers['jpeg_ls'] = jpeg_ls_handler
        logger.info("JPEG-LS handler imported and available")
    else:
        logger.warning("JPEG-LS handler imported but dependencies missing")
except ImportError:
    logger.warning("JPEG-LS handler import failed")

try:
    from pydicom.pixel_data_handlers import gdcm_handler
    if gdcm_handler.is_available():
        available_handlers['gdcm'] = gdcm_handler
        logger.info("GDCM handler imported and available")
    else:
        logger.warning("GDCM handler imported but dependencies missing")
except ImportError:
    logger.warning("GDCM handler import failed")

# Enable all available handlers in preferred order
handlers = []
for handler_name in ['gdcm', 'pylibjpeg', 'jpeg_ls', 'pillow', 'numpy']:
    if handler_name in available_handlers:
        handlers.append(available_handlers[handler_name])

if handlers:
    config.pixel_data_handlers = handlers
    logger.info(f"Enabled pixel data handlers (in order): {[h.__name__ for h in handlers]}")
else:
    logger.warning("No handlers were successfully configured!")

# Set additional configs to help with problematic files
config.APPLY_J2K_CORRECTIONS = True
if 'gdcm' in available_handlers:
    config.GDCM_HANDLER_IGNORE_ERRORS = True

def get_dicom_info(ds):
    """Get detailed information about a DICOM dataset for logging purposes"""
    info = {
        "TransferSyntaxUID": str(ds.file_meta.TransferSyntaxUID),
        "TransferSyntaxName": ds.file_meta.TransferSyntaxUID.name,
        "IsCompressed": ds.file_meta.TransferSyntaxUID.is_compressed,
        "Rows": getattr(ds, "Rows", "N/A"),
        "Columns": getattr(ds, "Columns", "N/A"),
        "BitsAllocated": getattr(ds, "BitsAllocated", "N/A"),
        "BitsStored": getattr(ds, "BitsStored", "N/A"),
        "SamplesPerPixel": getattr(ds, "SamplesPerPixel", "N/A"),
        "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", "N/A"),
        "PixelRepresentation": getattr(ds, "PixelRepresentation", "N/A"),
        "FileSize": f"{os.path.getsize(ds.filename) / (1024*1024):.2f} MB" if hasattr(ds, 'filename') and ds.filename else "N/A"
    }
    return info

# Check if dcmtk tools are available
def find_dcmtk_tool(tool_name):
    """Find path to a DCMTK tool if installed"""
    # Common paths where DCMTK tools might be installed
    paths_to_check = [
        '/usr/bin',
        '/usr/local/bin',
        '/opt/local/bin',
        '/data/data/com.termux/files/usr/bin'
    ]
    
    for path in paths_to_check:
        tool_path = os.path.join(path, tool_name)
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            return tool_path
    
    return None

# Use DCMTK to preserve compression if possible
def preserve_compression_with_dcmtk(original_path, uncompressed_path, output_path):
    """
    Use DCMTK tools to attempt to preserve the original compression
    Returns True if successful, False otherwise
    """
    try:
        # Read the original DICOM to determine compression type
        original_ds = pydicom.dcmread(original_path)
        transfer_syntax = original_ds.file_meta.TransferSyntaxUID
        logger.info(f"Original transfer syntax: {transfer_syntax.name}")
        
        # Log file sizes for comparison
        original_size_mb = os.path.getsize(original_path) / (1024*1024)
        uncompressed_size_mb = os.path.getsize(uncompressed_path) / (1024*1024)
        logger.info(f"Original size: {original_size_mb:.2f} MB, Uncompressed size: {uncompressed_size_mb:.2f} MB")
        
        # JPEG Baseline/Lossless compression
        if str(transfer_syntax).startswith('1.2.840.10008.1.2.4.5') or \
           str(transfer_syntax).startswith('1.2.840.10008.1.2.4.7'):
            dcmcjpeg_path = find_dcmtk_tool('dcmcjpeg')
            if dcmcjpeg_path:
                # Determine quality option based on transfer syntax
                quality_option = ""
                if transfer_syntax in [pydicom.uid.JPEGBaseline]:
                    quality_option = "--quality 90"  # Medium quality for baseline
                elif transfer_syntax in [pydicom.uid.JPEGExtended]:
                    quality_option = "--quality 75"  # Lower quality for extended
                elif transfer_syntax in [pydicom.uid.JPEGLossless, pydicom.uid.JPEGLosslessSV1]:
                    quality_option = "--lossless"   # Lossless compression
                
                # Run dcmcjpeg with appropriate options
                cmd = f"{dcmcjpeg_path} {quality_option} {uncompressed_path} {output_path}"
                logger.info(f"Running: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    compressed_size_mb = os.path.getsize(output_path) / (1024*1024)
                    logger.info(f"Compression successful! New size: {compressed_size_mb:.2f} MB")
                    logger.info(f"Compression ratio: {uncompressed_size_mb/compressed_size_mb:.2f}x")
                    return True
                else:
                    logger.error(f"dcmcjpeg failed: {result.stderr}")
            else:
                logger.warning("dcmcjpeg not found, cannot compress JPEG")
        
        # JPEG 2000 compression
        elif str(transfer_syntax).startswith('1.2.840.10008.1.2.4.9'):
            dcmcjp2k_path = find_dcmtk_tool('dcmcjp2k')
            if dcmcjp2k_path:
                # Determine whether to use lossless mode
                lossless_option = ""
                if transfer_syntax == pydicom.uid.JPEG2000Lossless:
                    lossless_option = "--lossless"
                
                # Run dcmcjp2k
                cmd = f"{dcmcjp2k_path} {lossless_option} {uncompressed_path} {output_path}"
                logger.info(f"Running: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    compressed_size_mb = os.path.getsize(output_path) / (1024*1024)
                    logger.info(f"Compression successful! New size: {compressed_size_mb:.2f} MB")
                    logger.info(f"Compression ratio: {uncompressed_size_mb/compressed_size_mb:.2f}x")
                    return True
                else:
                    logger.error(f"dcmcjp2k failed: {result.stderr}")
            else:
                logger.warning("dcmcjp2k not found, cannot compress JPEG 2000")
        
        # RLE compression
        elif str(transfer_syntax) == str(pydicom.uid.RLELossless):
            dcmcrle_path = find_dcmtk_tool('dcmcrle')
            if dcmcrle_path:
                # Run dcmcrle
                cmd = f"{dcmcrle_path} {uncompressed_path} {output_path}"
                logger.info(f"Running: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    compressed_size_mb = os.path.getsize(output_path) / (1024*1024)
                    logger.info(f"Compression successful! New size: {compressed_size_mb:.2f} MB")
                    logger.info(f"Compression ratio: {uncompressed_size_mb/compressed_size_mb:.2f}x")
                    return True
                else:
                    logger.error(f"dcmcrle failed: {result.stderr}")
            else:
                logger.warning("dcmcrle not found, cannot compress RLE")
        
        # JPEG-LS compression - try alternative approach for Termux
        elif str(transfer_syntax).startswith('1.2.840.10008.1.2.4.8'):
            # For JPEG-LS, we try a different approach using standard DCMTK tools
            # Since we don't have direct JPEG-LS support, try to use JPEG Lossless instead
            dcmcjpeg_path = find_dcmtk_tool('dcmcjpeg')
            if dcmcjpeg_path:
                # Try to use JPEG Lossless instead of JPEG-LS
                cmd = f"{dcmcjpeg_path} --lossless {uncompressed_path} {output_path}"
                logger.info(f"JPEG-LS not directly supported. Trying JPEG Lossless instead: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    compressed_size_mb = os.path.getsize(output_path) / (1024*1024)
                    logger.info(f"Alternative compression successful! New size: {compressed_size_mb:.2f} MB")
                    logger.info(f"Compression ratio: {uncompressed_size_mb/compressed_size_mb:.2f}x")
                    return True
                else:
                    logger.error(f"Alternative compression failed: {result.stderr}")
            else:
                logger.warning("dcmcjpeg not found, cannot use alternative compression")
        
        # No matching compression type or tools
        logger.warning(f"No compression tool available for {transfer_syntax.name}")
        return False
        
    except Exception as e:
        logger.error(f"Error in preserve_compression_with_dcmtk: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Utility function to convert compressed DICOM to uncompressed
def decompress_dicom(input_path, output_path=None):
    """
    Attempt to decompress a DICOM file
    Returns the path to the decompressed file
    """
    try:
        # If no output path specified, create a temporary one
        if output_path is None:
            output_path = input_path + '.uncompressed.dcm'
            
        logger.info(f"Attempting to decompress DICOM from {input_path} to {output_path}")
        
        # First try DCMTK if available
        dcmdjpeg_path = find_dcmtk_tool('dcmdjpeg')
        if dcmdjpeg_path:
            try:
                logger.info("Attempting decompression with DCMTK dcmdjpeg")
                cmd = f"{dcmdjpeg_path} {input_path} {output_path}"
                logger.info(f"Running: {cmd}")
                result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    logger.info("Successfully decompressed with DCMTK")
                    return output_path
                else:
                    logger.warning(f"DCMTK decompression failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"Error using DCMTK for decompression: {str(e)}")
        
        # Fallback to pydicom decompression
        logger.info("Attempting decompression with pydicom")
        ds = pydicom.dcmread(input_path)
        dicom_info = get_dicom_info(ds)
        logger.info(f"Original DICOM info: {dicom_info}")
        
        # Try to access pixel data to force decompression
        pixel_array = ds.pixel_array
        logger.info(f"Successfully read pixel array: shape={pixel_array.shape}, dtype={pixel_array.dtype}")
        
        # Save as uncompressed
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(output_path)
        logger.info(f"Successfully decompressed DICOM to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Decompression failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_polygon_mask(width, height, points):
    """
    Generate binary mask from polygon points
    
    Args:
        width: Image width
        height: Image height
        points: List of points defining the polygon
        
    Returns:
        NumPy array containing the mask (1 inside polygon, 0 outside)
    """
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    polygon_points = [(p['x'], p['y']) for p in points]
    draw.polygon(polygon_points, outline=1, fill=1)
    return np.array(img)

def index(request):
    return render(request, 'dicom_app/index.html')

def upload_dicom(request):
    if request.method == 'POST':
        form = DicomUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dicom_file = form.cleaned_data['dicom_file']

            # Define the correct upload path within MEDIA_ROOT
            upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_path, exist_ok=True)  # Ensure the directory exists

            # Save the uploaded file
            file_path = os.path.join(upload_path, 'temp.dcm')
            with open(file_path, 'wb+') as destination:
                for chunk in dicom_file.chunks():
                    destination.write(chunk)

            # Read the DICOM file using pydicom
            try:
                ds = pydicom.dcmread(file_path)
                logger.info(f"Uploaded DICOM: {get_dicom_info(ds)}")
                if hasattr(ds, 'PatientName'):
                    logger.info(f"Patient Name: {ds.PatientName}")
            except Exception as e:
                logger.warning(f"Could not read DICOM meta {str(e)}")

            # Return a URL that Django will serve dynamically
            dicom_url = f"/dicom/temp.dcm"
            return render(request, 'dicom_app/display.html', {'dicom_file_url': dicom_url})
    else:
        form = DicomUploadForm()
    return render(request, 'dicom_app/upload.html', {'form': form})

def serve_dicom_file(request, filename):
    # Check if download is requested
    is_download = request.GET.get('download', False)
    
    for folder in ['uploads', 'outputs']:
        file_path = os.path.join(settings.MEDIA_ROOT, folder, filename)
        if os.path.exists(file_path):
            try:
                # Log info about the file being served
                try:
                    ds = pydicom.dcmread(file_path)
                    logger.info(f"Serving DICOM file: {filename}")
                    logger.info(f"DICOM info: {get_dicom_info(ds)}")
                except Exception as read_error:
                    logger.info(f"Serving DICOM file (couldn't read metadata): {filename}")
                    logger.debug(f"Metadata read error: {str(read_error)}")
                    
                with open(file_path, "rb") as f:
                    response = HttpResponse(f.read(), content_type="application/dicom")
                    # Set attachment for downloads, inline for viewing
                    disposition = 'attachment' if is_download else 'inline'
                    response["Content-Disposition"] = f'{disposition}; filename="{filename}"'
                    return response
            except Exception as e:
                logger.error(f"Error serving file {filename}: {str(e)}")
                return HttpResponse(f"Error serving file: {str(e)}", status=500)
    
    logger.warning(f"File not found: {filename}")
    return HttpResponse("File not found", status=404)

@csrf_exempt
def crop_dicom(request):
    if request.method == 'POST':
        try:
            logger.info("Received crop request")
            data = json.loads(request.body)
            logger.debug(f"Request  {data}")
            
            # Get mask color from request (default to black if not specified)
            mask_color = data.get('maskColor', 'black')
            logger.info(f"Using mask color: {mask_color}")
            
            # Check if it's the new format (with 'type' field)
            if 'type' in data and data['type'] == 'free':
                selection_type = 'free'
            else:
                # Fallback to rectangle
                selection_type = 'rect'
            
            logger.info(f"Selection type: {selection_type}")

            # Load original DICOM with enhanced error handling
            original_dicom_path = os.path.join(settings.MEDIA_ROOT, 'uploads/temp.dcm')
            logger.info(f"Loading DICOM from: {original_dicom_path}")
            
            # First try to load the file directly
            try:
                ds = pydicom.dcmread(original_dicom_path)
                dicom_info = get_dicom_info(ds)
                # begin center tryout
                pixel_array = ds.pixel_array
                original_rows = ds.Rows
                original_columns = ds.Columns
                # end center tryout
                logger.info(f"Original DICOM info: {dicom_info}")
                
                # Try to access pixel data
                pixel_array = ds.pixel_array
                logger.info(f"Pixel array loaded successfully: shape={pixel_array.shape}, dtype={pixel_array.dtype}")
                
                # Store which pixel handler was used
                if hasattr(ds, '_pixel_array_handler'):
                    logger.info(f"Pixel handler used: {ds._pixel_array_handler.__name__}")
                
                dicom_path = original_dicom_path
                using_original = True
                is_compressed = ds.file_meta.TransferSyntaxUID.is_compressed
                logger.info(f"Successfully loaded original DICOM directly (compressed: {is_compressed})")
                
            except Exception as direct_load_error:
                logger.error(f"Could not load original DICOM directly: {str(direct_load_error)}")
                logger.error(traceback.format_exc())
                logger.info("Attempting to decompress DICOM...")
                
                # Attempt to decompress
                decomp_path = decompress_dicom(original_dicom_path)
                if decomp_path:
                    try:
                        ds = pydicom.dcmread(decomp_path)
                        dicom_info = get_dicom_info(ds)
                        logger.info(f"Decompressed DICOM info: {dicom_info}")
                        
                        pixel_array = ds.pixel_array
                        logger.info(f"Decompressed pixel array loaded: shape={pixel_array.shape}, dtype={pixel_array.dtype}")
                        
                        dicom_path = decomp_path
                        using_original = False
                        is_compressed = False
                        logger.info("Successfully loaded decompressed DICOM")
                    except Exception as decomp_load_error:
                        error_msg = f"Failed to load even the decompressed DICOM: {str(decomp_load_error)}"
                        logger.error(error_msg)
                        logger.error(traceback.format_exc())
                        return JsonResponse({'error': error_msg}, status=500)
                else:
                    error_msg = "Could not decompress DICOM file"
                    logger.error(error_msg)
                    return JsonResponse({'error': error_msg}, status=500)
            
            # Print additional helpful info
            logger.info(f"DICOM dimensions: {ds.Rows}x{ds.Columns}")
            logger.info(f"Pixel data type: {pixel_array.dtype}")
            pixel_min, pixel_max = np.min(pixel_array), np.max(pixel_array)
            logger.info(f"Pixel value range: {pixel_min} to {pixel_max}")
            
            mask_value = np.max(pixel_array)
            
            if selection_type == 'rect':
                # Handle rectangle
                if 'x1' not in data or 'y1' not in data or 'x2' not in data or 'y2' not in data:
                    logger.error("Missing rectangle coordinates in request")
                    return JsonResponse({'error': 'Missing rectangle coordinates'}, status=400)
                
                x1 = max(0, int(data['x1']))
                y1 = max(0, int(data['y1']))
                x2 = min(int(data['x2']), ds.Columns)
                y2 = min(int(data['y2']), ds.Rows)

                # begin center tryout
                cropped_region = pixel_array[y1:y2, x1:x2]
                cropped_height, cropped_width = cropped_region.shape

                # Create centered array with original dimensions
                centered_array = np.zeros((original_rows, original_columns), dtype=pixel_array.dtype)
                start_y = (original_rows - cropped_height) // 2
                start_x = (original_columns - cropped_width) // 2
                end_y = start_y + cropped_height
                end_x = start_x + cropped_width

                # Ensure bounds
                start_y = max(0, start_y)
                start_x = max(0, start_x)
                end_y = min(original_rows, end_y)
                end_x = min(original_columns, end_x)

                centered_array[start_y:end_y, start_x:end_x] = cropped_region
                ds.PixelData = centered_array.tobytes()

                # Maintain original dimensions
                ds.Rows = original_rows
                ds.Columns = original_columns
                # end center tryout

                logger.info(f"Rectangle coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Create a copy of the pixel array to modify
                masked_array = pixel_array.copy()
                logger.info(f"Created copy of pixel array for masking")
                
                # Apply mask to the OUTSIDE of the rectangle
                if mask_color == 'black':
                    logger.info("Applying black mask outside rectangle")
                    mask_value = 0                    
                else:
                    logger.info("Applying white mask outside rectangle")
                    # Set everything outside the rectangle to white (max value)
                
                # Top portion
                if y1 > 0:
                    masked_array[0:y1, :] = mask_value
                # Bottom portion
                if y2 < ds.Rows:
                    masked_array[y2:, :] = mask_value
                # Left portion (within the height of the rectangle)
                if x1 > 0:
                    masked_array[y1:y2, 0:x1] = mask_value
                # Right portion (within the height of the rectangle)
                if x2 < ds.Columns:
                    masked_array[y1:y2, x2:] = mask_value
                
            else:
                # Handle polygon
                if 'points' not in data or len(data['points']) < 3:
                    logger.error("Invalid polygon: needs at least 3 points")
                    return JsonResponse({'error': 'Polygon needs at least 3 points'}, status=400)
                
                # Validate points
                for p in data['points']:
                    if not (0 <= p['x'] < ds.Columns and 0 <= p['y'] < ds.Rows):
                        logger.error(f"Point out of bounds: ({p['x']}, {p['y']})")
                        return JsonResponse({'error': 'Point out of bounds'}, status=400)
                
                logger.info(f"Creating polygon mask with {len(data['points'])} points")
                # Create polygon mask (1 inside polygon, 0 outside)
                mask = create_polygon_mask(ds.Columns, ds.Rows, data['points'])
                
                # For polygon, we want to keep the inside and mask the outside
                if mask_color == 'black':
                    logger.info("Applying black mask outside polygon")
                    # Black mask: set outside to black (0)
                    masked_array = np.where(mask == 1, pixel_array, 0)
                else:
                    logger.info("Applying white mask outside polygon")
                    # White mask: set outside to white (max value)
                    masked_array = np.where(mask == 1, pixel_array, max_pixel_value)
            
            # Save the result, preserving compression if possible
            try:
                # Preserve original pixel representation
                masked_array = masked_array.astype(pixel_array.dtype)
                logger.info(f"Masked array created with shape={masked_array.shape}, dtype={masked_array.dtype}")
                
                # Define output paths
                output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                cropped_filename = f'cropped_{uuid.uuid4().hex}.dcm'
                cropped_path = os.path.join(output_dir, cropped_filename)
                uncompressed_path = cropped_path + '.uncompressed.dcm'
                
                # First save an uncompressed version
                logger.info(f"Saving uncompressed version to {uncompressed_path}")
                output_ds = ds.copy()
                output_ds.PixelData = masked_array.tobytes()
                output_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                output_ds.is_little_endian = True
                output_ds.is_implicit_VR = False
                output_ds.save_as(uncompressed_path)
                
                # Calculate uncompressed file size
                uncompressed_size_mb = os.path.getsize(uncompressed_path) / (1024*1024)
                logger.info(f"Uncompressed file size: {uncompressed_size_mb:.2f} MB")
                
                # Try to preserve compression if original was compressed
                compression_preserved = False
                if using_original and is_compressed:
                    logger.info(f"Attempting to preserve original compression")
                    compression_preserved = preserve_compression_with_dcmtk(
                        original_dicom_path, uncompressed_path, cropped_path)
                
                # If compression failed or wasn't needed, use the uncompressed version
                if not compression_preserved:
                    logger.info("Using uncompressed version for output")
                    shutil.move(uncompressed_path, cropped_path)
                elif os.path.exists(uncompressed_path):
                    # Clean up temporary file if compression worked
                    os.remove(uncompressed_path)
                    logger.info("Removed temporary uncompressed file")
                
                # Get final file size and info
                final_size_mb = os.path.getsize(cropped_path) / (1024*1024)
                logger.info(f"Final file size: {final_size_mb:.2f} MB")
                
                # Verify the saved file
                try:
                    verify_ds = pydicom.dcmread(cropped_path)
                    verify_info = get_dicom_info(verify_ds)
                    logger.info(f"Output file verification: {verify_info}")
                    
                    # Double-check compression status
                    actual_compressed = verify_ds.file_meta.TransferSyntaxUID.is_compressed
                    if compression_preserved and not actual_compressed:
                        logger.warning("Expected compressed output but got uncompressed")
                        compression_preserved = False
                    
                    # Test pixel data access
                    verify_pixels = verify_ds.pixel_array
                    logger.info(f"Output pixel verification: shape={verify_pixels.shape}")
                except Exception as verify_error:
                    logger.warning(f"Output verification issue: {str(verify_error)}")
                
                # Return success response
                return JsonResponse({
                    'cropped_url': f'/media/outputs/{cropped_filename}',
                    'mask_color': mask_color,
                    'compression_preserved': compression_preserved,
                    'original_size_mb': f"{os.path.getsize(original_dicom_path) / (1024*1024):.2f}",
                    'final_size_mb': f"{final_size_mb:.2f}",
                    'compression_ratio': f"{(uncompressed_size_mb/final_size_mb):.2f}x" if compression_preserved else "1.0x"
                })
                
            except Exception as save_error:
                logger.error(f"Failed to save masked DICOM: {str(save_error)}")
                logger.error(traceback.format_exc())
                
                # Emergency fallback save
                try:
                    logger.info("Attempting emergency fallback save")
                    emergency_path = os.path.join(output_dir, f'emergency_{uuid.uuid4().hex}.dcm')
                    
                    # Create minimal viable dataset
                    fallback_ds = pydicom.Dataset()
                    fallback_ds.file_meta = pydicom.Dataset()
                    fallback_ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture
                    fallback_ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                    fallback_ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                    fallback_ds.is_little_endian = True
                    fallback_ds.is_implicit_VR = False
                    
                    # Required elements for a valid DICOM
                    fallback_ds.SOPClassUID = fallback_ds.file_meta.MediaStorageSOPClassUID
                    fallback_ds.SOPInstanceUID = fallback_ds.file_meta.MediaStorageSOPInstanceUID
                    fallback_ds.StudyInstanceUID = pydicom.uid.generate_uid()
                    fallback_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                    fallback_ds.PatientName = getattr(ds, 'PatientName', 'ANONYMOUS')
                    fallback_ds.PatientID = getattr(ds, 'PatientID', '0000000')
                    fallback_ds.Modality = 'OT'  # Other
                    
                    # Add pixel data
                    fallback_ds.Rows = masked_array.shape[0]
                    fallback_ds.Columns = masked_array.shape[1]
                    fallback_ds.SamplesPerPixel = 1
                    fallback_ds.PhotometricInterpretation = 'MONOCHROME2'
                    fallback_ds.BitsAllocated = 16
                    fallback_ds.BitsStored = 16
                    fallback_ds.HighBit = 15
                    fallback_ds.PixelRepresentation = 0
                    fallback_ds.PixelData = masked_array.tobytes()
                    
                    # Save emergency file
                    fallback_ds.save_as(emergency_path)
                    logger.info(f"Emergency save successful to {emergency_path}")
                    
                    return JsonResponse({
                        'cropped_url': f'/media/outputs/{os.path.basename(emergency_path)}',
                        'mask_color': mask_color,
                        'compression_preserved': False,
                        'note': 'Emergency fallback save used - original metadata lost'
                    })
                    
                except Exception as fallback_error:
                    logger.error(f"Emergency fallback also failed: {str(fallback_error)}")
                    logger.error(traceback.format_exc())
                    return JsonResponse({'error': 'Failed to save masked DICOM'}, status=500)

        except Exception as e:
            logger.error(f"Error during cropping: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)

def clean_outputs_directory(keep_file=None):
    """
    Clean the outputs directory by deleting all files except the one specified.
    
    Args:
        keep_file (str, optional): Filename to keep (without path). If None, deletes all files.
    
    Returns:
        int: Number of files deleted
    """
    output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')
    if not os.path.exists(output_dir):
        return 0
        
    count = 0
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        # Skip directories and the file to keep
        if os.path.isdir(file_path) or (keep_file and filename == keep_file):
            continue
            
        try:
            os.remove(file_path)
            count += 1
            logger.info(f"Deleted file from outputs directory: {filename}")
        except Exception as e:
            logger.error(f"Failed to delete file {filename}: {str(e)}")
            
    logger.info(f"Cleaned outputs directory: {count} files deleted")
    return count

@csrf_exempt
def clear_outputs(request):
    """Clean all files in the outputs directory"""
    if request.method == 'POST':
        try:
            count = clean_outputs_directory()
            return JsonResponse({
                'success': True,
                'files_deleted': count,
                'message': f"Successfully deleted {count} files from outputs directory"
            })
        except Exception as e:
            logger.error(f"Error clearing outputs directory: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

import os
from django.shortcuts import render
from .forms import DicomUploadForm, DicomFolderForm

def dicom_upload_view(request):
    dicom_files = []
    dicom_form = DicomUploadForm()
    folder_form = DicomFolderForm()

    if request.method == 'POST':
        # Handle single file upload
        if 'upload_single' in request.POST:
            dicom_form = DicomUploadForm(request.POST, request.FILES)
            if dicom_form.is_valid():
                # Process the single DICOM file
                dicom_file = dicom_form.cleaned_data['dicom_file']
                dicom_files = [dicom_file.name]
        
        # Handle folder upload
        elif 'upload_folder' in request.POST:
            folder_form = DicomFolderForm(request.POST)
            if folder_form.is_valid():
                # Process the selected folder
                folder_path = folder_form.cleaned_data['folder_path']
                dicom_files = find_dicom_files(folder_path)

    return render(request, 'upload.html', {
        'dicom_form': dicom_form,
        'folder_form': folder_form,
        'dicom_files': dicom_files,
    })

def find_dicom_files(directory):
    """ Recursively find all DICOM files in the selected directory. """
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files
