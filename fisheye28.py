import numpy as np
import cv2
import math


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
                                   compression_strength=0.2,
                                   camera_offset=0.0,
                                   panini_weight=0,
                                   stereo_weight=0,
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


# ============================
# Example usage
# ============================
import cv2
import os

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"vr180_{file_name}")

            input_image = cv2.imread(input_path)
            if input_image is None:
                print(f"❌ Failed to load {input_path}")
                continue

            vr_img = create_vr180_projection_square(
                input_image,
                output_size=3084,
                compression_strength=0.3,
                camera_offset=0.3,
                panini_weight=0,
                stereo_weight=0,
                corner_fill="extend",
                blur_offset=100,
                blur_mode="edge",
                blur_strength=50
            )

            cv2.imwrite(output_path, vr_img)
            print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    input_folder = r"D:\AIDS\cv_job_assignment\vr_180_round2\output\video_full\8k_right_stereo"
    output_folder = r"D:\AIDS\cv_job_assignment\vr_180_round3\8k_right_stereo"

    process_folder(input_folder, output_folder)
    print("🎉 All images processed!")
