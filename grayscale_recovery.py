"""
Idea: We want to store a full-size copy of the original RGB image as grayscale inside 
the different channels
"""
from help_functions import *
import cv2
import numpy as np


def watermarking(img, seed):
    size_x, size_y = img.shape[:2]

    # Get RNG
    rng = np.random.default_rng(seed=seed)

    # Generate authentication mask
    auth = rng.integers(low=0, high=4, size=(size_x * size_y), dtype=np.uint8)

    # Generate grayscale image
    img_gs = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    img_gs_qua = (img_gs & 0b11110000) >> 4
    gs = img_gs_qua
    
    gs0 = gs & 0b0011
    gs1 = (gs & 0b1100) >> 2
    
    gs0_array = gs0.reshape(-1)
    gs1_array = gs1.reshape(-1)

    # Recovery index array
    rec_indices = np.arange(size_x * size_y)
    # Shuffle
    rng.shuffle(rec_indices)

    gs0_array = gs0_array[rec_indices]
    gs1_array = gs1_array[rec_indices]

    # Arrange channel shuffling
    base_indices = np.arange(3)
    channel_indices = np.tile(base_indices, (size_x, size_y, 1))
    channel_indices = rng.permuted(channel_indices, axis=2)

    auth_indices = (channel_indices == 0)
    gs0_indices = (channel_indices == 1)
    gs1_indices = (channel_indices == 2)

    base_array = np.zeros((size_x, size_y, 3), dtype=np.uint8)
    base_array[auth_indices] = auth
    base_array[gs0_indices] = gs0_array
    base_array[gs1_indices] = gs1_array
    
    watermarked = (img & 0b11111100) | base_array

    return watermarked


def tamper_detection(img, seed):
    size_x, size_y = img.shape[:2]

    # Get RNG
    rng = np.random.default_rng(seed=seed)

    # Generate authentication mask
    auth_exp = rng.integers(low=0, high=4, size=(size_x * size_y), dtype=np.uint8)
    auth_rec = np.zeros((size_x * size_y), dtype=np.uint8)

    # Recovery index array
    rec_indices = np.arange(size_x * size_y)
    # Shuffle
    rng.shuffle(rec_indices)

    
    base_indices = np.arange(3)
    channel_indices = np.tile(base_indices, (size_x, size_y, 1))
    channel_indices = rng.permuted(channel_indices, axis=2)
    
    auth_indices = (channel_indices == 0)
    gs0_indices = (channel_indices == 1)
    gs1_indices = (channel_indices == 2)

    # Extracted base_array
    base_array = img & 3

    auth_rec = base_array[auth_indices]
    gs0_array = base_array[gs0_indices]
    gs1_array = base_array[gs1_indices]
    
    gs0_unshuffled = np.empty((size_x * size_y), dtype=np.uint8)
    gs1_unshuffled = np.empty((size_x * size_y), dtype=np.uint8)

    gs0_unshuffled[rec_indices] = gs0_array
    gs1_unshuffled[rec_indices] = gs1_array

    gs0 = gs0_unshuffled.reshape((size_x, size_y))
    gs1 = gs1_unshuffled.reshape((size_x, size_y))
    
    gs = (gs1 << 2) | (gs0)
    recovery_watermark = gs * 16

    # Detect tampering
    auth_diff = np.clip(np.abs((auth_exp.astype(int) - auth_rec.astype(int))), a_min=0, a_max=1).astype(np.uint8).reshape((size_x, size_y))
    
    # Smooth out tamper mask
    tamper_mask = grow_tamper_pixels_single_channel(tamper_mask=auth_diff, step=2)

    return tamper_mask, rec_indices, recovery_watermark


def __tamper_detection_for_test(img, seed):
    size_x, size_y = img.shape[:2]

    # Get RNG
    rng = np.random.default_rng(seed=seed)

    # Generate authentication mask
    auth_exp = rng.integers(low=0, high=4, size=(size_x * size_y), dtype=np.uint8)
    auth_rec = np.zeros((size_x * size_y), dtype=np.uint8)

    # Recovery index array
    rec_indices = np.arange(size_x * size_y)
    # Shuffle
    rng.shuffle(rec_indices)

    
    base_indices = np.arange(3)
    channel_indices = np.tile(base_indices, (size_x, size_y, 1))
    channel_indices = rng.permuted(channel_indices, axis=2)
    
    auth_indices = (channel_indices == 0)
    gs0_indices = (channel_indices == 1)
    gs1_indices = (channel_indices == 2)

    # Extracted base_array
    base_array = img & 3

    auth_rec = base_array[auth_indices]
    gs0_array = base_array[gs0_indices]
    gs1_array = base_array[gs1_indices]
    
    gs0_unshuffled = np.empty((size_x * size_y), dtype=np.uint8)
    gs1_unshuffled = np.empty((size_x * size_y), dtype=np.uint8)

    gs0_unshuffled[rec_indices] = gs0_array
    gs1_unshuffled[rec_indices] = gs1_array

    gs0 = gs0_unshuffled.reshape((size_x, size_y))
    gs1 = gs1_unshuffled.reshape((size_x, size_y))
    
    gs = (gs1 << 2) | (gs0)
    recovery_watermark = gs * 16

    # Detect tampering
    auth_diff = np.clip(np.abs((auth_exp.astype(int) - auth_rec.astype(int))), a_min=0, a_max=1).astype(np.uint8).reshape((size_x, size_y))
    
    # Smooth out tamper mask
    tamper_mask = grow_tamper_pixels_single_channel(tamper_mask=auth_diff, step=2)

    return tamper_mask, auth_diff


def recovery(img, tamper_mask, recovery_indices, recovery_watermark):
    rec_indices = recovery_indices
    size_x, size_y = img.shape[:2]

    # Detect voids
    auth_diff_array = tamper_mask.reshape(-1)
    voids = np.zeros(size_x * size_y, dtype=np.uint8)
    voids[rec_indices] = auth_diff_array
    voids = voids.reshape((size_x, size_y))
        
    # Restore all void recovery pixels that fall within a tampered zone:
    rec_w_fixed, voids_rem = fix_voids_using_interpolation(voids, rec_watermark=recovery_watermark, size_x=size_x, size_y=size_y)

    # Replace tampered area with recovery watermark
    tamper_mask_3d = np.repeat(tamper_mask[:, :, np.newaxis], 3, axis=2)

    to_keep = (tamper_mask_3d < 1).astype(np.uint8)
    to_replace = tamper_mask
    inpainted =  (img * to_keep) | np.repeat((rec_w_fixed * to_replace)[:, :, np.newaxis], 3, axis=2)

    # if include_tamper_mask:
    #     return inpainted, tamper_mask
    return inpainted
