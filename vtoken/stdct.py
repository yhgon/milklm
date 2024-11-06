import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct

# Pre-configured min and max values for specific patch sizes
pre_config_min_max = {
    "4x4": {
        (0, 0): [0, 1024], 
        (3, 3): [-512, 512],
    },    
    "8x8": {
        (0, 0): [0, 2048],
        (3, 3): [-1024, 1024],
        (7, 7): [-512, 512],
    },
    "16x16": {
        (0, 0): [0, 4096],
        (3, 3): [-2048, 2048],
        (7, 7): [-1024, 1024],
        (15, 15): [-512, 512],
    },
    "32x32": {
        (0, 0): [0, 8192],
        (3, 3): [-4096, 4096],
        (7, 7): [-2048, 2048],
        (15, 15): [-1024, 1024],
        (31, 31): [-512, 512],
    },
    "64x64": {
        (0, 0): [0, 16384],
        (3, 3): [-8192, 8192],
        (7, 7): [-4096, 4096],
        (15, 15): [-2048, 2048],
        (31, 31): [-1024, 1024],
        (63, 63): [-512, 512],
    },
    "128x128": {
        (0, 0): [0, 32768],        
        (3, 3): [-16384, 16384],
        (7, 7): [ -8192, 8192],
        (15, 15): [ -4096, 4096],
        (31, 31): [ -2048, 2048],
        (63, 63): [ -1024, 1024],
        (127, 127): [-512, 512],
    }
}

def load_image(path, mode='RGB'):
    image = Image.open(path).convert(mode)
    return np.array(image)

def pad_image(image, block_size=8):
    h, w = image.shape[:2]
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant') if image.ndim == 3 else np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
    return padded_image, (h, w)

def stdct(block, norm='ortho'):
    return dct(dct(block.T, norm=norm).T, norm=norm)

def istdct(dct_block, norm='ortho'):
    return idct(idct(dct_block.T, norm=norm).T, norm=norm)

def image_to_dct_with_overlap(image, block_size=8, hop_size=4):
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    dct_height = (h - block_size) // hop_size + 1
    dct_width = (w - block_size) // hop_size + 1
    dct_latent = np.zeros((dct_height, dct_width, c, block_size, block_size), dtype=np.float32)

    for i in range(dct_height):
        for j in range(dct_width):
            for k in range(c):
                row_start = i * hop_size
                col_start = j * hop_size
                block = image[row_start:row_start + block_size, col_start:col_start + block_size] if c == 1 else image[row_start:row_start + block_size, col_start:col_start + block_size, k]
                dct_latent[i, j, k] = stdct(block)

    return dct_latent

def quantize_nbits(array, min_val, max_val, nbits=8):
    scale = (2 ** nbits) - 1
    return np.round((array - min_val) / (max_val - min_val) * scale).astype(np.int32)

def dequantize_nbits(array, min_val, max_val, nbits=8):
    scale = (2 ** nbits) - 1
    return array / scale * (max_val - min_val) + min_val

def get_interpolated_min_max(size_key, i, j):
    size_config = pre_config_min_max[size_key]
    nearest_keys = sorted(size_config.keys())
    
    lower_key = max([key for key in nearest_keys if key[0] <= i and key[1] <= j], default=(0, 0))
    upper_key = min([key for key in nearest_keys if key[0] >= i and key[1] >= j], default=nearest_keys[-1])

    if lower_key == upper_key:
        min_val, max_val = size_config[lower_key]
    else:
        min_val_lower, max_val_lower = size_config[lower_key]
        min_val_upper, max_val_upper = size_config[upper_key]
        ratio = max(i / upper_key[0], j / upper_key[1]) if upper_key[0] and upper_key[1] else 1
        min_val = min_val_lower + (min_val_upper - min_val_lower) * ratio
        max_val = max_val_lower + (max_val_upper - max_val_lower) * ratio

    return min_val, max_val

def progressive_dct_normalize(dct_latent, patch_size):
    size_key = f"{patch_size}x{patch_size}"
    for i in range(patch_size):
        for j in range(patch_size):
            min_val, max_val = get_interpolated_min_max(size_key, i, j)
            component = dct_latent[..., i, j]
            dct_latent[..., i, j] = (component - min_val) / (max_val - min_val) if max_val != min_val else 0
    return dct_latent

def progressive_dct_denormalize(dct_latent, patch_size):
    size_key = f"{patch_size}x{patch_size}"
    for i in range(patch_size):
        for j in range(patch_size):
            min_val, max_val = get_interpolated_min_max(size_key, i, j)
            component = dct_latent[..., i, j]
            dct_latent[..., i, j] = component * (max_val - min_val) + min_val
    return dct_latent

def truncate_digits(array, precision=2):
    return np.round(array, decimals=precision)

def dct_to_image_with_overlap(dct_latent, original_size, block_size=8, hop_size=4):
    h, w = original_size
    c = dct_latent.shape[2]
    padded_h = h + (block_size - h % block_size) % block_size
    padded_w = w + (block_size - w % block_size) % block_size
    reconstructed_image = np.zeros((padded_h, padded_w, c), dtype=np.float32)
    weight_matrix = np.zeros((padded_h, padded_w, c), dtype=np.float32)
    dct_height, dct_width = dct_latent.shape[:2]

    for i in range(dct_height):
        for j in range(dct_width):
            for k in range(c):
                row_start = i * hop_size
                col_start = j * hop_size
                ifft_block = istdct(dct_latent[i, j, k])
                reconstructed_image[row_start:row_start + block_size, col_start:col_start + block_size, k] += ifft_block
                weight_matrix[row_start:row_start + block_size, col_start:col_start + block_size, k] += 1

    reconstructed_image = np.divide(reconstructed_image, weight_matrix, where=(weight_matrix != 0))
    return np.clip(reconstructed_image[:h, :w], 0, 255).astype(np.uint8)

# Main pipeline
filename = "dog.png"
mode = 'L'  # Use 'L' for grayscale, 'RGB' for color, and 'RGBA' for images with alpha channel
patch_size = 16
hop_size = 8
truncate = 3
nbits = 10

image = load_image(filename, mode=mode)
padded_image, original_size = pad_image(image, block_size=patch_size)
print("Image shape:", image.shape)

dct_latent = image_to_dct_with_overlap(padded_image, block_size=patch_size, hop_size=hop_size)
dct_latent_norm = progressive_dct_normalize(dct_latent, patch_size)
dct_latent_norm_truncate = truncate_digits(dct_latent_norm, precision=truncate)
dct_latent_norm_truncated_quant = quantize_nbits(dct_latent_norm_truncate, min_val=-1, max_val=1, nbits=nbits)
dct_latent_norm_truncated_dequant = dequantize_nbits(dct_latent_norm_truncated_quant, min_val=-1, max_val=1, nbits=nbits)

dct_latent_denorm = progressive_dct_denormalize(dct_latent_norm_truncated_dequant, patch_size)
print("dct_latent shape:", dct_latent.shape)
reconstructed_image = dct_to_image_with_overlap(dct_latent_denorm, original_size, block_size=patch_size, hop_size=hop_size)

plt.figure(figsize=(36, 12))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray" if mode == 'L' else None)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image, cmap="gray" if mode == 'L' else None)
plt.title("Reconstructed Image")

plt.subplot(1, 3, 3)
# Determine if the image has multiple channels
# Ensure reconstructed_image has the same shape as the original image
if reconstructed_image.shape != image.shape:
    if reconstructed_image.shape[-1] == 1:
        reconstructed_image = reconstructed_image[..., 0]  # Remove the extra channel dimension for grayscale

# Determine if the image has multiple channels
is_color = image.ndim == 3 and image.shape[2] > 1

# Calculate delta image based on whether the image is grayscale or color
delta_image = np.sqrt(np.sum((image - reconstructed_image) ** 2, axis=2)) if is_color else np.abs(image - reconstructed_image)
fixed_max = np.max(delta_image)  

# Plotting
img = plt.imshow(delta_image, cmap="viridis", vmax=5)
plt.title("Difference Image")
cbar = plt.colorbar(img, fraction=0.046, pad=0.04)


plt.tight_layout()
plt.show()
