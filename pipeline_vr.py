import sys
import os
from pathlib import Path

# Ensure bundled spatial-media package is importable (Streamlit Cloud friendly)
# This points to the repo folder `spatial-media/` which contains the `spatialmedia` package
spatial_media_repo = Path(__file__).parent / "spatial-media"
if spatial_media_repo.exists():
    sys.path.insert(0, str(spatial_media_repo))
    print(f"[OK] Added to sys.path: {spatial_media_repo}")
else:
    # Fallback: import from a zip archive containing the 'spatialmedia' package at its root
    spatial_media_zip = Path(__file__).parent / "spatialmedia.zip"
    if spatial_media_zip.exists():
        sys.path.insert(0, str(spatial_media_zip))
        print(f"[OK] Added zip to sys.path: {spatial_media_zip}")
    else:
        print("[WARNING] spatial-media folder or spatialmedia.zip not found; metadata injection may fail.")
#===============================
# STEP 1: Video to Frames (memory-safe)
#===============================

import subprocess
import cv2
import numpy as np
import os

def video_to_frames_folder(video_path, output_dir):
    """
    Extract frames from video and save to disk only.
    Does NOT keep all frames in memory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path
    ]
    width, height = map(int, subprocess.check_output(probe_cmd).decode().strip().split("x"))

    # FFmpeg command → raw frames
    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    frame_size = width * height * 3
    frame_idx = 0

    while True:
        raw_frame = pipe.stdout.read(frame_size)
        if not raw_frame:
            break

        # Convert raw bytes → numpy image
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

        # Save to disk only
        filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(filename, frame)

        if frame_idx % 50 == 0:
            print(f"Saved frame {frame_idx}")

        frame_idx += 1

    pipe.stdout.close()
    pipe.wait()

    print(f"[OK] Extracted {frame_idx} frames")
    print(f"[OK] Saved to folder: {output_dir}")


# # Example usage
# if __name__ == "__main__":
#     video_to_frames_folder(
#         r"D:\AIDS\cv_job_assignment\vr_180_round2\input\output_clip.mp4",
#         r"D:\AIDS\cv_job_assignment\vr_180_round3\frames"
#     )




#===============================
# STEP 2: depth map
#===============================



import cv2
import torch
import numpy as np
import os
import glob
from enum import Enum
from PIL import Image
import torch.nn.functional as F

class ModelType(Enum):
    DPT_LARGE = "DPT_Large"
    DPT_HYBRID = "DPT_Hybrid"
    MIDAS_SMALL = "MiDaS_small"

class AdvancedMidas:
    def __init__(self, modelType: ModelType = ModelType.MIDAS_SMALL, device=None):
        self.modelType = modelType
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        print(f"Loading model: {modelType.value}")
        print(f"Using device: {self.device}")
        
        # Load the model
        if modelType == ModelType.DPT_LARGE:
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif modelType == ModelType.DPT_HYBRID:
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        else:
            # Prefer local checkpoint for MiDaS small
            local_small_ckpt = "checkpoints/midas_v21_small_256.pt"
            if os.path.exists(local_small_ckpt):
                print(f"Found local MiDaS small checkpoint: {local_small_ckpt}")
                # Construct the architecture without downloading pretrained weights
                try:
                    self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)
                except TypeError:
                    # Fallback in case hub signature doesn't accept pretrained flag
                    self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                # Load weights from local file
                try:
                    state = torch.load(local_small_ckpt, map_location=self.device)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    load_result = self.midas.load_state_dict(state, strict=False)
                    print(f"Loaded local weights with result: {load_result}")
                except Exception as e:
                    print(f"Warning: Failed to load local MiDaS small weights: {e}")
                    print("Falling back to hub pretrained weights (cache or download)...")
                    self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            else:
                print("Local MiDaS small checkpoint not found; using hub pretrained weights (cache or download)...")
                self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        if self.modelType.value in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def enhance_image_quality(self, image):
        """Advanced image enhancement for better depth estimation"""
        img_float = image.astype(np.float32) / 255.0
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_float, -1, kernel)
        sharpened = np.clip(sharpened, 0, 1)
        gamma = 0.8
        enhanced = np.power(sharpened, gamma)
        enhanced = (enhanced * 255).astype(np.uint8)
        return enhanced

    def multi_scale_prediction(self, image):
        """Predict depth at multiple scales and combine for better quality"""
        h, w = image.shape[:2]
        input_tensor = self.transform(image).to(self.device)
        with torch.no_grad():
            depth_original = self.midas(input_tensor)
        
        try:
            scale_factor = 1.2
            h_scaled = int(h * scale_factor)
            w_scaled = int(w * scale_factor)
            image_scaled = cv2.resize(image, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
            
            input_tensor_scaled = self.transform(image_scaled).to(self.device)
            with torch.no_grad():
                depth_scaled = self.midas(input_tensor_scaled)
                depth_scaled = F.interpolate(
                    depth_scaled.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            depth_combined = 0.6 * depth_original.squeeze() + 0.4 * depth_scaled
        except:
            depth_combined = depth_original.squeeze()
        
        return depth_combined

    def advanced_depth_refinement(self, depth_map, original_image):
        """Advanced depth map refinement using edge information"""
        if torch.is_tensor(depth_map):
            depth_np = depth_map.cpu().numpy()
        else:
            depth_np = depth_map
        
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        guide = gray.astype(np.float32) / 255.0
        depth_filtered = self.guided_filter(guide, depth_np.astype(np.float32), radius=8, epsilon=0.01)
        depth_enhanced = self.enhance_depth_edges(depth_filtered, edges)
        return depth_enhanced

    def guided_filter(self, guide, input_image, radius, epsilon):
        mean_guide = cv2.blur(guide, (radius, radius))
        mean_input = cv2.blur(input_image, (radius, radius))
        mean_guide_input = cv2.blur(guide * input_image, (radius, radius))
        cov_guide_input = mean_guide_input - mean_guide * mean_input
        var_guide = cv2.blur(guide * guide, (radius, radius)) - mean_guide * mean_guide
        a = cov_guide_input / (var_guide + epsilon)
        b = mean_input - a * mean_guide
        mean_a = cv2.blur(a, (radius, radius))
        mean_b = cv2.blur(b, (radius, radius))
        output = mean_a * guide + mean_b
        return output

    def enhance_depth_edges(self, depth_map, edges):
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edges_dilated.astype(np.float32) / 255.0
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_enhanced = gradient_magnitude * (1 + edge_mask * 0.5)
        blur = cv2.GaussianBlur(depth_map, (3, 3), 1.0)
        enhanced = depth_map + 0.3 * (depth_map - blur)
        return enhanced

    def advanced_normalization(self, depth_map):
        p1, p99 = np.percentile(depth_map, [1, 99])
        depth_clipped = np.clip(depth_map, p1, p99)
        depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
        depth_equalized = clahe.apply(depth_normalized)
        return depth_equalized

    def predict_frame(self, frame, save_path=None, apply_colormap=True, edge_enhancement_strength=2.0, final_sharpening_strength=1.5):
        """Advanced depth prediction with multiple techniques
        
        Returns both float depth (0-1) for VR180 and colormapped visualization.
        """
        img_enhanced = self.enhance_image_quality(frame)
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        print("Performing multi-scale depth prediction...")
        depth_tensor = self.multi_scale_prediction(img_rgb)
        original_shape = frame.shape[:2]
        depth_resized = F.interpolate(
            depth_tensor.unsqueeze(0).unsqueeze(0),
            size=original_shape,
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze()
        print("Applying advanced depth refinement...")
        depth_refined = self.advanced_depth_refinement(depth_resized, img_rgb)
        
        # Float normalized depth for VR180
        depth_float = depth_refined.astype(np.float32)
        depth_float -= depth_float.min()
        depth_float /= (depth_float.max() + 1e-8)
        
        # Visualization
        depth_final = self.advanced_normalization(depth_refined)
        kernel_sharpen = np.array([
            [0, -final_sharpening_strength, 0],
            [-final_sharpening_strength, 1 + 4 * final_sharpening_strength, -final_sharpening_strength],
            [0, -final_sharpening_strength, 0]
        ])
        depth_sharpened = cv2.filter2D(depth_final, -1, kernel_sharpen)
        depth_sharpened = np.clip(depth_sharpened, 0, 255).astype(np.uint8)
        depth_colored = None
        if apply_colormap:
            depth_colored = cv2.applyColorMap(depth_sharpened, cv2.COLORMAP_MAGMA)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if apply_colormap:
                # cv2.imwrite(save_path, depth_colored, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                pass
            # np.save(save_path.replace('.png','_float.npy'), depth_float)
            raw_path = save_path.replace('.png', '_raw.png')
            cv2.imwrite(raw_path, depth_sharpened, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"[OK] Depth map saved to {save_path}")
            print(f"[OK] Raw depth saved to {raw_path}")
            print(f"[OK] Float depth saved to {save_path.replace('.png','_float.npy')}")
        
        return depth_float, depth_colored

    
        
        return results




#============================================
#Step 3: stereo pairs
#============================================


import cv2
import numpy as np
import os
from glob import glob
from numba import njit, prange

@njit(parallel=True)
def generate_stereo_pair_numba(rgb_image, depth_gray, baseline=15):
    """
    Numba-accelerated stereo pair generation using grayscale depth.

    Args:
        rgb_image: Original RGB image (H x W x 3)
        depth_gray: Grayscale depth image (H x W), 0=near, 255=far
        baseline: Maximum pixel shift for stereo effect (in pixels)
    Returns:
        left_img, right_img: Stereo pair images
    """
    h, w = depth_gray.shape
    left_img = np.zeros_like(rgb_image)
    right_img = np.zeros_like(rgb_image)

    for y in prange(h):
        for x in range(w):
            depth_norm = depth_gray[y, x] / 255.0
            shift = int((1.0 - depth_norm) * baseline)

            left_x = min(max(x - shift, 0), w - 1)
            right_x = min(max(x + shift, 0), w - 1)

            left_img[y, x] = rgb_image[y, left_x]
            right_img[y, x] = rgb_image[y, right_x]

    return left_img, right_img

def process_folder_numba_separate(rgb_folder, depth_folder, left_out, right_out, baseline=15):
    # Make separate output folders
    os.makedirs(left_out, exist_ok=True)
    os.makedirs(right_out, exist_ok=True)

    rgb_files = sorted(glob(os.path.join(rgb_folder, "*.png")))
    depth_files = sorted(glob(os.path.join(depth_folder, "*.png")))

    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        left, right = generate_stereo_pair_numba(rgb, depth, baseline)

        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        cv2.imwrite(os.path.join(left_out, f"{base_name}_left.png"), left)
        cv2.imwrite(os.path.join(right_out, f"{base_name}_right.png"), right)
        print(f"Processed {base_name}")


#============================================
#Step 4: fish eye projection
#============================================

import numpy as np
import cv2
import math
import os
from glob import glob


# ============================
# Projection Utilities
# ============================
def apply_compression(x_norm, y_norm, r_out, compression_strength):
    """Radial compression (fisheye-like)."""
    compression_factor = 1.0 - compression_strength * (r_out**2)
    compression_factor = np.clip(compression_factor, 0.1, 1.0)
    return x_norm * compression_factor, y_norm * compression_factor

def apply_projection(x_c, y_c, panini_weight, stereo_weight):
    """Panini + Stereographic + standard blend."""
    phi = np.arcsin(np.clip(y_c, -1, 1))
    theta = x_c * (np.pi * 0.5)

    # Panini
    panini_x = np.tan(theta) / (np.cos(theta) + 1e-10)
    panini_y = np.tan(phi) / (np.cos(theta) + 1e-10)

    # Stereographic
    k = 2 / (1 + np.cos(phi) * np.cos(theta))
    stereo_x = k * np.cos(phi) * np.sin(theta)
    stereo_y = k * np.sin(phi)

    # Blend with standard spherical
    x_proj = (1 - panini_weight - stereo_weight) * (np.cos(phi) * np.sin(theta)) \
             + panini_weight * panini_x \
             + stereo_weight * stereo_x
    y_proj = (1 - panini_weight - stereo_weight) * (np.sin(phi)) \
             + panini_weight * panini_y \
             + stereo_weight * stereo_y
    z_proj = np.cos(phi) * np.cos(theta)

    return x_proj, y_proj, z_proj

def apply_camera_offset(x_proj, y_proj, z_proj, camera_offset):
    """Apply Z-offset to simulate seat position."""
    return x_proj, y_proj, z_proj + camera_offset

def apply_perspective(x_proj, y_proj, z_proj, w, h, focal_length):
    """Convert 3D projection into 2D image coords."""
    x_in = (x_proj / (z_proj + 1e-10)) * focal_length + w // 2
    y_in = (y_proj / (z_proj + 1e-10)) * focal_length + h // 2
    return x_in, y_in

def apply_distortion(x_in, y_in, w, h):
    """Apply barrel distortion."""
    center_x, center_y = w // 2, h // 2
    dx = x_in - center_x
    dy = y_in - center_y
    r_pixel = np.sqrt(dx**2 + dy**2)
    max_r = min(w, h) // 2
    r_normalized = np.clip(r_pixel / max_r, 0, 1)

    # Barrel coefficients
    k1, k2, k3 = 0.15, 0.08, 0.02
    distortion = 1 + k1*r_normalized**2 + k2*r_normalized**4 + k3*r_normalized**6

    return center_x + dx * distortion, center_y + dy * distortion

def build_corner_fill(x_norm, y_norm, w, h, focal_length):
    """Compute extended corner mapping (for pixels outside circle)."""
    angle = np.arctan2(y_norm, x_norm)
    x_circle = np.cos(angle)
    y_circle = np.sin(angle)
    x_in_corner = (x_circle / (1 + 1e-10)) * focal_length + w // 2
    y_in_corner = (y_circle / (1 + 1e-10)) * focal_length + h // 2
    return x_in_corner, y_in_corner

# ============================
# Blur Utilities
# ============================
def build_blur_mask(output_size, blur_offset=3, blur_mode="corner"):
    """Create feather mask for blur (0=sharp, 1=blur)."""
    cx, cy = output_size // 2, output_size // 2
    Y, X = np.ogrid[:output_size, :output_size]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Circle radius
    r_max = output_size // 2
    blur_start = r_max - blur_offset

    if blur_mode == "circle":
        blur_end = r_max
    elif blur_mode == "edge":
        blur_end = max(cx, cy)
    else:  # "corner"
        blur_end = math.sqrt(cx**2 + cy**2)

    mask = np.clip((dist - blur_start) / (blur_end - blur_start), 0, 1).astype(np.float32)
    return mask

def apply_feather_blur(image, mask, blur_strength=15):
    """Blend sharp and blurred image with feather mask."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_strength, sigmaY=blur_strength)
    mask_3c = cv2.merge([mask, mask, mask])
    return (image * (1 - mask_3c) + blurred * mask_3c).astype(np.uint8)

# ============================
# Main VR180 Builder
# ============================
def create_vr180_projection_square(input_image,
                                   output_size=2048,
                                   compression_strength=0.3,
                                   camera_offset=0.0,
                                   panini_weight=0.7,
                                   stereo_weight=0.2,
                                   corner_fill="extend",
                                   blur_offset=3,
                                   blur_mode="corner",
                                   blur_strength=15):
    """
    Full VR180 projection pipeline with Panini/Stereo blend, distortion,
    corner fill, and feather blur.
    """
    h, w, c = input_image.shape

    # Output grid
    x_out, y_out = np.meshgrid(np.arange(output_size), np.arange(output_size))
    x_norm = (x_out / (output_size - 1)) * 2 - 1
    y_norm = (y_out / (output_size - 1)) * 2 - 1
    r_out = np.sqrt(x_norm**2 + y_norm**2)
    circular_mask = r_out <= 1.0

    # Compression
    x_c, y_c = apply_compression(x_norm, y_norm, r_out, compression_strength)

    # Projection
    x_proj, y_proj, z_proj = apply_projection(x_c, y_c, panini_weight, stereo_weight)
    x_proj, y_proj, z_proj = apply_camera_offset(x_proj, y_proj, z_proj, camera_offset)

    # Perspective
    focal_length = min(w, h) * 0.4
    x_in, y_in = apply_perspective(x_proj, y_proj, z_proj, w, h, focal_length)

    # Distortion
    x_in, y_in = apply_distortion(x_in, y_in, w, h)

    # Valid mask
    valid_mask = (x_in >= 0) & (x_in < w) & (y_in >= 0) & (y_in < h) & (z_proj > 0) & circular_mask

    # Corner fill
    if corner_fill == "extend":
        x_in_corner, y_in_corner = build_corner_fill(x_norm, y_norm, w, h, focal_length)
        valid_corner = (x_in_corner >= 0) & (x_in_corner < w) & (y_in_corner >= 0) & (y_in_corner < h)
        x_in[~valid_mask & valid_corner] = x_in_corner[~valid_mask & valid_corner]
        y_in[~valid_mask & valid_corner] = y_in_corner[~valid_mask & valid_corner]
        valid_mask = valid_mask | (~valid_mask & valid_corner)

    # Remap
    map_x = x_in.astype(np.float32)
    map_y = y_in.astype(np.float32)
    map_x[~valid_mask] = -1
    map_y[~valid_mask] = -1
    vr_image = cv2.remap(input_image, map_x, map_y,
                         cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)

    # Blur mask + apply
    mask = build_blur_mask(output_size, blur_offset=blur_offset, blur_mode=blur_mode)
    vr_image = apply_feather_blur(vr_image, mask, blur_strength=blur_strength)

    # Debug save
    # cv2.imwrite("debug_mask.jpg", (mask * 255).astype(np.uint8))

    return vr_image

def process_folder(input_folder, output_folder):
    image_paths = sorted(glob(os.path.join(input_folder, "*.png")) + glob(os.path.join(input_folder, "*.jpg")))
    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {path}")
            continue
        
        vr_img = create_vr180_projection_square(
            img,
            output_size=3084,
            compression_strength=0.3,
            camera_offset=0.6,
            panini_weight=0.1,
            stereo_weight=0.1,
            corner_fill="extend",
            blur_offset=100,
            blur_mode="edge",    # "circle", "edge", "corner"
            blur_strength=50
        )
        
        cv2.imwrite(os.path.join(output_folder, filename), vr_img)
        print(f"[OK] Processed {filename}")


#============================================
#Step 5: stereo stitching
#============================================

import cv2
import numpy as np
import os
from tqdm import tqdm

def stitch_frame(lf, rf, left_folder, right_folder, output_folder):
    """Read left/right frames, resize, stitch side-by-side, save output"""
    left_path  = os.path.join(left_folder, lf)
    right_path = os.path.join(right_folder, rf)

    left_img  = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print(f"⚠️ Skipping {lf}, {rf} (cannot read)")
        return

    # Resize to match
    h = min(left_img.shape[0], right_img.shape[0])
    w = min(left_img.shape[1], right_img.shape[1])
    left_img  = cv2.resize(left_img, (w, h))
    right_img = cv2.resize(right_img, (w, h))

    stereo = np.concatenate((left_img, right_img), axis=1)

    out_path = os.path.join(output_folder, lf)  # save with left filename
    cv2.imwrite(out_path, stereo)

def create_stereo_vr180(left_folder, right_folder, output_folder):
    """Folder-to-folder stereo VR180 stitching (single-threaded)"""
    os.makedirs(output_folder, exist_ok=True)

    left_files  = sorted([f for f in os.listdir(left_folder) if f.lower().endswith((".png", ".jpg"))])
    right_files = sorted([f for f in os.listdir(right_folder) if f.lower().endswith((".png", ".jpg"))])

    if len(left_files) != len(right_files):
        print("[WARN] Left and right folder have different number of frames!")

    for lf, rf in tqdm(zip(left_files, right_files), total=len(left_files), desc="Stitching frames"):
        stitch_frame(lf, rf, left_folder, right_folder, output_folder)

    print(f"[OK] Stereo stitching complete! Saved to {output_folder}")


#============================================
#Step 6: convert to video
#============================================


import cv2
import os
from tqdm import tqdm
import subprocess

import subprocess
from pathlib import Path

def frames_to_video_with_audio_safe_ffmpeg(frames_folder, output_video_path, original_video_path, fps=30):
    """
    Convert frames to video using FFmpeg (H.264 + yuv420p) and optionally merge audio.
    Fully compatible with normal players.
    """
    frames_folder = Path(frames_folder)
    output_video_path = Path(output_video_path)

    # Ensure output directory exists
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover frames (PNG, JPG, JPEG)
    png_files = sorted(frames_folder.glob("*.png"))
    jpg_files = sorted(frames_folder.glob("*.jpg"))
    jpeg_files = sorted(frames_folder.glob("*.jpeg"))
    all_frames = sorted(png_files + jpg_files + jpeg_files)
    if not all_frames:
        print("[WARN] No frames (*.png/*.jpg/*.jpeg) found to encode.")
        return

    # Prefer numeric sequence pattern if present; else create one (for builds without glob support)
    numbered_example = frames_folder / "frame_00000.png"

    # Temporary video without audio
    temp_video_path = output_video_path.with_name(output_video_path.stem + "_noaudio.mp4")

    # Encode frames to H.264 with yuv420p
    temp_seq_dir = None
    try:
        if numbered_example.exists() and len(jpg_files) == 0 and len(jpeg_files) == 0:
            # Use numeric sequence in-place
            frame_pattern = str(frames_folder / "frame_%05d.png")
        else:
            # Create a temporary numbered sequence (for builds without glob support)
            import shutil
            temp_seq_dir = frames_folder / "_seq_tmp"
            # Recreate folder
            if temp_seq_dir.exists():
                for p in temp_seq_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            else:
                temp_seq_dir.mkdir(parents=True, exist_ok=True)

            for idx, src in enumerate(all_frames):
                dst = temp_seq_dir / f"frame_{idx:05d}.png"
                shutil.copy2(src, dst)
            frame_pattern = str(temp_seq_dir / "frame_%05d.png")

        cmd_encode = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(temp_video_path)
        ]
        subprocess.run(cmd_encode, check=True)
    finally:
        # Clean temp sequence directory if created
        if temp_seq_dir and temp_seq_dir.exists():
            for p in temp_seq_dir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                temp_seq_dir.rmdir()
            except Exception:
                pass
    print(f"[OK] Video without audio saved: {temp_video_path}")

    # Check if original video has audio
    cmd_probe = ["ffprobe", "-i", str(original_video_path), "-show_streams", "-select_streams", "a", "-loglevel", "error"]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    has_audio = "codec_type=audio" in result.stdout

    if has_audio:
        # Merge audio
        cmd_merge = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-i", str(original_video_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video_path)
        ]
        subprocess.run(cmd_merge, check=True)
        print(f"[OK] Final video with audio saved: {output_video_path}")
        temp_video_path.unlink()
    else:
        # On Windows, rename will fail if destination exists. Remove existing file first.
        try:
            Path(output_video_path).unlink(missing_ok=True)
        except Exception:
            pass
        temp_video_path.replace(output_video_path)  # atomic replace where supported
        print(f"[WARN] No audio found. Video saved without audio: {output_video_path}")

#============================================
#Step 7: inject metadata
#============================================


def inject_vr180_metadata(input_file, output_file, stereo_mode="left-right", projection="fisheye", use_v2=False):
    """Inject VR180 spatial metadata using the bundled spatialmedia package.

    Works on Streamlit Cloud when only spatialmedia.zip is present by:
    1) Calling spatialmedia.__main__.main() directly (preferred)
    2) Falling back to subprocess with sys.executable and PYTHONPATH set
    """
    try:
        # Preferred: in-process call
        from spatialmedia.__main__ import main as spatialmedia_main
        args = ["-i"]
        if use_v2:
            args.append("--v2")
        args += [f"--stereo={stereo_mode}", f"--projection={projection}", input_file, output_file]
        spatialmedia_main(args)
        print(f"[OK] VR180 video with metadata saved to {output_file}")
        return True
    except Exception as e:
        print(f"[WARN] Programmatic spatialmedia injection failed: {e}")
        print("[INFO] Falling back to subprocess invocation...")
        try:
            env = os.environ.copy()
            # Ensure the current project path and the spatialmedia.zip are on PYTHONPATH
            project_dir = str(Path(__file__).parent)
            zip_path = str((Path(__file__).parent / "spatialmedia.zip").resolve())
            py_paths = [project_dir]
            if os.path.exists(zip_path):
                py_paths.insert(0, zip_path)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(py_paths + ([existing] if existing else []))
            cmd = [
                sys.executable, "-m", "spatialmedia", "-i",
                f"--stereo={stereo_mode}", f"--projection={projection}",
                input_file, output_file
            ]
            subprocess.run(cmd, check=True, env=env)
            print(f"[OK] VR180 video with metadata saved to {output_file}")
            return True
        except subprocess.CalledProcessError as sub_e:
            print(f"[WARN] Metadata injection failed with code {sub_e.returncode}. Command: {' '.join(cmd)}")
            return False

# import subprocess

# input_file = "D:/AIDS/cv_job_assignment/vr_180_round3/stereo_video_final.mp4"
# output_file = "D:/AIDS/cv_job_assignment/vr_180_round3/stereo_video_final_meta_fisheye.mp4"

# # Build the command as a list
# cmd = [
#     "python", "-m", "spatialmedia", "-i",
#     "--stereo=left-right",
#     "--projection=fisheye",
#     input_file,
#     output_file
# ]

# # Run the command
# result = subprocess.run(cmd, capture_output=True, text=True)

# # Print stdout and stderr
# print("STDOUT:\n", result.stdout)
# print("STDERR:\n", result.stderr)
# #====================================================================

import os
import tempfile
import shutil
from pathlib import Path

def process_video_to_vr180(input_video, output_video, fps=18,
                            midas_model="MIDAS_SMALL", baseline=15,
                            vr_output_size=2048, compression_strength=0.3,
                            camera_offset=0.3, panini_weight=0.1, stereo_weight=0.1,
                            blur_offset=50, blur_mode="corner", blur_strength=15,
                            progress_callback=None, status_callback=None):
    """
    Full VR180 pipeline in temporary folders.
    """
    # Create a root temp folder
    root_temp = Path(__file__).parent / "vr180_temp"
    root_temp.mkdir(exist_ok=True)
    print(f"Using temp root folder: {root_temp}")

    # Step 1: frames
    frames_folder = root_temp / "frames"
    frames_folder.mkdir(exist_ok=True)
    if status_callback:
        status_callback("STEP 1: Extracting frames...")
    if progress_callback:
        progress_callback(5)   # show immediate activity
    video_to_frames_folder(input_video, str(frames_folder))
    if progress_callback:
        progress_callback(20)  # Frames done: 20%

    # Step 2: depth
    depth_folder = root_temp / "depth_maps"
    depth_folder.mkdir(exist_ok=True)
    if status_callback:
        status_callback("STEP 2: Predicting depth maps...")
    midas = AdvancedMidas(ModelType[midas_model])
    import glob, cv2
    img_paths = sorted(glob.glob(str(frames_folder / "*.*")))
    total_depth = max(1, len(img_paths))
    for i, img_path in enumerate(img_paths):
        frame = cv2.imread(img_path)
        out_path = depth_folder / Path(img_path).name
        midas.predict_frame(frame, save_path=str(out_path), apply_colormap=True)
        if progress_callback:
            # Depth progresses from 20% to 60%
            frac = (i + 1) / total_depth
            pct = 20 + int(frac * (60 - 20))
            progress_callback(min(pct, 60))

    # Step 3: stereo
    left_folder = root_temp / "stereo_left"
    right_folder = root_temp / "stereo_right"
    left_folder.mkdir(exist_ok=True)
    right_folder.mkdir(exist_ok=True)
    if status_callback:
        status_callback("STEP 3: Generating stereo pairs...")
    process_folder_numba_separate(str(frames_folder), str(depth_folder),
                                  str(left_folder), str(right_folder), baseline=baseline)
    if progress_callback:
        progress_callback(70)

    # Step 4: VR180 projection
    vr_left_folder = root_temp / "vr180_left"
    vr_right_folder = root_temp / "vr180_right"
    vr_left_folder.mkdir(exist_ok=True)
    vr_right_folder.mkdir(exist_ok=True)
    if status_callback:
        status_callback("STEP 4: Applying fisheye/VR180 projection...")
    process_folder(str(left_folder), str(vr_left_folder))
    if progress_callback:
        progress_callback(75)
    process_folder(str(right_folder), str(vr_right_folder))
    if progress_callback:
        progress_callback(85)

    # Step 5: stereo stitching
    stereo_output_folder = root_temp / "vr180_stereo"
    stereo_output_folder.mkdir(exist_ok=True)
    if status_callback:
        status_callback("STEP 5: Stitching stereo frames...")
    if progress_callback:
        progress_callback(88)
    create_stereo_vr180(str(vr_left_folder), str(vr_right_folder), str(stereo_output_folder))
    if progress_callback:
        progress_callback(90)

    # Step 6: convert to video
    if status_callback:
        status_callback("STEP 6: Converting frames to video...")
    frames_to_video_with_audio_safe_ffmpeg(str(stereo_output_folder), output_video, input_video, fps=fps)
    if progress_callback:
        progress_callback(95)

    # Step 7: inject metadata
    if status_callback:
        status_callback("STEP 7: Injecting VR180 stereo metadata...")

    meta_output = str(Path(output_video).with_name(Path(output_video).stem + "_meta.mp4"))
    # Ensure we can overwrite an existing meta file on repeated runs
    try:
        Path(meta_output).unlink(missing_ok=True)
    except Exception:
        pass

    success = inject_vr180_metadata(output_video, meta_output)
    if not success:
        # Fall back to returning the non-metadata video if injection failed
        meta_output = output_video
    if progress_callback:
        progress_callback(100)

    # Optional: cleanup temp
    shutil.rmtree(root_temp)
    print("Temporary files cleaned up.")

    return meta_output

# Example usage
if __name__ == "__main__":
    
    input_vid = r"D:\AIDS\vr_180_round2\input\inception_short.mp4"
    output_vid = str((Path(__file__).parent / "output_compression_strength_04_vr180.mp4").resolve())
    process_video_to_vr180(input_vid, output_vid, fps=15,
                        midas_model="MIDAS_SMALL", baseline=15,
                        vr_output_size=1440, compression_strength=0.3,
                        camera_offset=0, panini_weight=0, stereo_weight=0,
                        blur_offset=50, blur_mode="edge", blur_strength=50)


