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

    print(f"✅ Extracted {frame_idx} frames")
    print(f"📂 Saved to folder: {output_dir}")


# Example usage
if __name__ == "__main__":
    video_to_frames_folder(
        r"D:\AIDS\cv_job_assignment\vr_180_round2\input\output_clip.mp4",
        r"D:\AIDS\cv_job_assignment\vr_180_round3\frames"
    )




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
            local_small_ckpt = r"D:\AIDS\cv_job_assignment\vr_180_round3\app\checkpoints\midas_v21_small_256.pt"
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
            print(f"✅ Depth map saved to {save_path}")
            print(f"✅ Raw depth saved to {raw_path}")
            print(f"✅ Float depth saved to {save_path.replace('.png','_float.npy')}")
        
        return depth_float, depth_colored

    
        
        return results

if __name__ == "__main__":
    # midas = AdvancedMidas(ModelType.MIDAS_SMALL)
    # frame_paths = [r"D:\AIDS\cv_job_assignment\vr_180_round3\frames\frame_00003.png"]
    # out_dir = r"D:\AIDS\cv_job_assignment\vr_180_round3\app"
    # os.makedirs(out_dir, exist_ok=True)
    
    # print("Processing with MAXIMUM edge enhancement...")
    # results = midas.predict_frame(
    #     frame_paths, 
    #     out_dir, 
    #     apply_colormap=True,
    #     edge_enhancement_strength=4.0,
    #     final_sharpening_strength=3.0
    # )
    # Paths
    input_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\frames"
    output_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\depth_maps"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize model
    midas = AdvancedMidas(modelType=ModelType.MIDAS_SMALL)

    # Collect all image files (PNG, JPG, etc.)
    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.*")))

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        # Read frame
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Failed to read {img_path}, skipping...")
            continue
        
        # Create output path
        base_name = os.path.basename(img_path)
        out_path = os.path.join(output_folder, base_name)
        
        # Predict depth
        depth_float, depth_colored = midas.predict_frame(
            frame,
            save_path=out_path,
            apply_colormap=True
        )
        
        print(f"✅ Done {img_path}")

    print("All images processed!")





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

if __name__ == "__main__":
    rgb_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\frames"
    depth_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\depth_maps"
    left_out = r"D:\AIDS\cv_job_assignment\vr_180_round3\stereo_left"
    right_out = r"D:\AIDS\cv_job_assignment\vr_180_round3\stereo_right"
    os.makedirs(left_out, exist_ok=True)
    os.makedirs(right_out, exist_ok=True)

    process_folder_numba_separate(rgb_folder, depth_folder, left_out, right_out, baseline=15)
    print("✅ All frames processed into left/right stereo folders!")


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
    cv2.imwrite("debug_mask.jpg", (mask * 255).astype(np.uint8))

    return vr_image


# # ============================
# # Example usage
# # ============================
# if __name__ == "__main__":
#     input_image = cv2.imread(r"D:\AIDS\cv_job_assignment\vr_180_round3\frame\right_frame_0033.png")

#     vr_img = create_vr180_projection_square(
#         input_image,
#         output_size=1024,
#         compression_strength=0.3,
#         camera_offset=0,
#         panini_weight=0.1,
#         stereo_weight=0.1,
#         corner_fill="extend",
#         blur_offset=100,
#         blur_mode="edge",    # "circle", "edge", "corner"
#         blur_strength=50       # now configurable!
#     )

#     cv2.imwrite(r"D:\AIDS\cv_job_assignment\vr_180_round3\vr180_final.jpg", vr_img)
#     print("✅ Saved VR180 with modular pipeline + blur strength control")




# Make sure create_vr180_projection_square is already defined in this script

# ============================
# Paths
# ============================
left_input_folder = r"stereo_left"
right_input_folder = r"stereo_right"

left_output_folder = r"vr180_left"
right_output_folder = r"vr180_right"

os.makedirs(left_output_folder, exist_ok=True)
os.makedirs(right_output_folder, exist_ok=True)

# ============================
# Helper to process folder
# ============================
def process_folder(input_folder, output_folder):
    image_paths = sorted(glob(os.path.join(input_folder, "*.png")) + glob(os.path.join(input_folder, "*.jpg")))
    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Could not read {path}")
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
        print(f"✅ Processed {filename}")

# ============================
# Process both folders
# ============================
print("Processing LEFT folder...")
process_folder(left_input_folder, left_output_folder)

print("Processing RIGHT folder...")
process_folder(right_input_folder, right_output_folder)

print("🎉 VR180 projection + blur done for all images!")





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
        print("⚠️ Warning: Left and right folder have different number of frames!")

    for lf, rf in tqdm(zip(left_files, right_files), total=len(left_files), desc="Stitching frames"):
        stitch_frame(lf, rf, left_folder, right_folder, output_folder)

    print(f"✅ Stereo stitching complete! Saved to {output_folder}")

# ---------------- Example usage ----------------
if __name__ == "__main__":
    left_folder  = r"D:\AIDS\cv_job_assignment\vr_180_round3\vr180_left"
    right_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\vr180_right"
    output_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\vr180_stereo"
    os.makedirs(output_folder, exist_ok=True)
    create_stereo_vr180(left_folder, right_folder, output_folder)





#============================================
#Step 6: convert to video
#============================================


import cv2
import os
from tqdm import tqdm
import subprocess

def frames_to_video_with_audio_safe(frames_folder, output_video_path, original_video_path, fps=30):
    """
    Convert frames to video and optionally merge audio if original video has audio,
    trimming audio to match video length.
    """
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frame_files:
        print("⚠️ No frames found!")
        return

    # Read first frame to get size
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Calculate video duration
    video_duration = len(frame_files) / fps
    print(f"Video duration: {video_duration:.2f}s")

    # Temporary video without audio
    temp_video_path = output_video_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Write frames
    for f in tqdm(frame_files, desc="Writing frames"):
        frame_path = os.path.join(frames_folder, f)
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()
    print(f"✅ Video without audio saved: {temp_video_path}")

    # Check if original video has audio
    cmd_probe = ["ffprobe", "-i", original_video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"]
    result = subprocess.run(cmd_probe, capture_output=True, text=True)
    has_audio = "codec_type=audio" in result.stdout

    # Merge audio only if exists
    if has_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_video_path,
            "-i", original_video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_duration),  # trim audio to video length
            output_video_path
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Final video with audio saved: {output_video_path}")
            os.remove(temp_video_path)
        except subprocess.CalledProcessError as e:
            print("⚠️ Audio merge failed:", e)
    else:
        # No audio, rename temp video
        os.rename(temp_video_path, output_video_path)
        print("⚠️ Original video has no audio. Saved video without audio.")


# ---------------- Example usage ----------------
if __name__ == "__main__":
    frames_folder = r"vr180_stereo"
    output_video_path = r"D:\AIDS\cv_job_assignment\vr_180_round3\stereo_video_final.mp4"
    original_video_path = r"D:\AIDS\cv_job_assignment\vr_180_round2\input\Inception 4K HDR _ Hallway Fight Scene(4K_HD).webm"
    fps = 15  # 5 frames → 5 second video

    frames_to_video_with_audio_safe(frames_folder, output_video_path, original_video_path, fps=fps)


#============================================
#Step 7: inject metadata
#============================================


import subprocess

input_file = "D:/AIDS/cv_job_assignment/vr_180_round3/stereo_video_final.mp4"
output_file = "D:/AIDS/cv_job_assignment/vr_180_round3/stereo_video_final_meta_fisheye.mp4"

# Build the command as a list
cmd = [
    "python", "-m", "spatialmedia", "-i",
    "--stereo=left-right",
    "--projection=fisheye",
    input_file,
    output_file
]

# Run the command
result = subprocess.run(cmd, capture_output=True, text=True)

# Print stdout and stderr
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)
