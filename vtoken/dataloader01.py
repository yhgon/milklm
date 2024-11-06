import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
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
        (7, 7): [-8192, 8192],
        (15, 15): [-4096, 4096],
        (31, 31): [-2048, 2048],
        (63, 63): [-1024, 1024],
        (127, 127): [-512, 512],
    }
}

class VisionDCTLoader(Dataset):
    def __init__(self, image_paths, config, min_max_values=pre_config_min_max):
        self.image_paths = image_paths
        self.config = config
        self.min_max_values = min_max_values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.image_paths[idx], self.config['mode'])
        padded_image, original_size = self.pad_image(image, self.config['patch_size'])
        dct_latent = self.image_to_dct_with_overlap(padded_image, block_size=self.config['patch_size'], hop_size=self.config['hop_size'])
        dct_latent_norm = self.progressive_dct_normalize(dct_latent, self.config['patch_size'])
        dct_latent_norm_truncate = self.truncate_digits(dct_latent_norm, precision=self.config['truncation_precision'])
        dct_latent_quant = self.quantize_nbits(dct_latent_norm_truncate, min_val=-1, max_val=1, nbits=self.config['quantization_nbits'])

        # Metadata to pass alongside DCT latent representation
        metadata = {
            "mode": self.config['mode'],
            "size": original_size,
            "patch_size": self.config['patch_size'],
            "hop_size": self.config['hop_size'],
            "truncation_precision": self.config['truncation_precision'],
            "quantization_nbits": self.config['quantization_nbits']
        }

        return dct_latent_quant, metadata

    @staticmethod
    def load_image(path, mode='RGB'):
        image = Image.open(path).convert(mode)
        return np.array(image)

    @staticmethod
    def pad_image(image, block_size=8):
        h, w = image.shape[:2]
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant') if image.ndim == 3 else np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded_image, (h, w)

    @staticmethod
    def stdct(block, norm='ortho'):
        return dct(dct(block.T, norm=norm).T, norm=norm)

    @staticmethod
    def istdct(dct_block, norm='ortho'):
        return idct(idct(dct_block.T, norm=norm).T, norm=norm)

    def image_to_dct_with_overlap(self, image, block_size=8, hop_size=4):
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
                    dct_latent[i, j, k] = self.stdct(block)

        return dct_latent

    @staticmethod
    def truncate_digits(array, precision=2):
        return np.round(array, decimals=precision)

    @staticmethod
    def quantize_nbits(array, min_val, max_val, nbits=8):
        scale = (2 ** nbits) - 1
        return np.round((array - min_val) / (max_val - min_val) * scale).astype(np.int32)

    def progressive_dct_normalize(self, dct_latent, patch_size):
        size_key = f"{patch_size}x{patch_size}"
        for i in range(patch_size):
            for j in range(patch_size):
                min_val, max_val = self.get_interpolated_min_max(size_key, i, j)
                component = dct_latent[..., i, j]
                dct_latent[..., i, j] = (component - min_val) / (max_val - min_val) if max_val != min_val else 0
        return dct_latent

    def get_interpolated_min_max(self, size_key, i, j):
        size_config = self.min_max_values[size_key]
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

# Configuration and DataLoader Setup
config = {
    "mode": "L",  # Grayscale
    "patch_size": 16,
    "hop_size": 8,
    "truncation_precision": 3,
    "quantization_nbits": 10
}

image_paths = ["dog.png", "cat.png"]  # List of image paths
dataset = VisionDCTLoader(image_paths, config)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loading data in batch
for batch in dataloader:
    input_latents, meta_data = batch
    print("DCT Latents:", input_latents.shape)
    print("Metadata:", meta_data)
