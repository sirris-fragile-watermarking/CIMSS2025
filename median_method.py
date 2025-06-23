"""
This code implements the median fragile watermarking approach proposed in:
Image tamper detection and self-recovery using multiple median watermarking
by Vishal Rajput, Irshad Ahmad Ansari
https://doi.org/10.1007/s11042-019-07971-w
"""

import numpy as np
import cv2
from skimage.metrics import mean_squared_error
from help_functions import display_images
from generic_attacks import Text_addition




def post_processing(image):
    # Median filter
    image = cv2.medianBlur(image, ksize=3)
    
    # Sharpening
    laplacian_mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel=laplacian_mask)
    
    # Averaging (= Blurring = Low-pass filter)
    # image = cv2.blur(image, (3,3))
    return image

def do_self_embedding_rgb(original_image, seed, verbose=False):
    
    img_size_x, img_size_y, _ = original_image.shape
    
    rng = np.random.default_rng(seed=seed)

    # Step 1: 
    # Split image in 4 equal parts
    x_step_size = img_size_x // 2
    y_step_size = img_size_y // 2
    parts = [
        original_image[:x_step_size, :y_step_size, :] & 0b11110000,
        original_image[:x_step_size, y_step_size:, :] & 0b11110000,
        original_image[x_step_size:, :y_step_size, :] & 0b11110000,
        original_image[x_step_size:, y_step_size:, :] & 0b11110000,
        ]

    if verbose:
        print("Subimage sizes", [a.shape for a in parts])
        display_images(parts, title="Image divided in equal parts")

    # Step 2: 
    # Downscale image
    target_shape = img_size_x // 2, img_size_y // 2, 3
    target_shape_T = img_size_y // 2, img_size_x // 2, 3
    small_image = cv2.resize(original_image, target_shape_T[:2])
    # display_images([small_image], title="Resized image")
    
    # Step 3:
    # Extract MSB:
    # Do ravel() to extract a contiguous 1D array (not a copy), as the shuffle() function only shuffles the first axis of an array
    # (we will use ravel in reconstruction, use flatten for now here as we are making copies anyway)
    # So we transform the image into a 1D array to ensure thorough shuffling
    base_watermark = ((small_image & 0b11110000) >> 4).flatten()
    # display_images([base_watermark], title="Watermark")

    # Create 4 random permutations of the smaller image
    watermark_permutations = []

    for _ in range(4):
        shuffle_order = np.arange(len(base_watermark))
        rng.shuffle(shuffle_order)
        shuffled_watermark = np.copy(base_watermark)[shuffle_order]
        watermark_permutations.append(shuffled_watermark.reshape(target_shape))
        

    if verbose:
        display_images([perm << 4 for perm in watermark_permutations], "Permutations after shuffling")

    # Recombine parts and watermarks to create watermarked image
    new_image = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)

    new_image[:x_step_size,:y_step_size, :] = parts[0] | watermark_permutations[0]
    new_image[:x_step_size,y_step_size:, :] = parts[1] | watermark_permutations[1]
    new_image[x_step_size:,:y_step_size, :] = parts[2] | watermark_permutations[2]
    new_image[x_step_size:,y_step_size:, :] = parts[3] | watermark_permutations[3]
    
    if verbose:
        display_images([new_image], title="Watermarked image")
    
    return new_image

def tamper_detection(image, seed, do_post_processing=False):
    # Extract LSBs
    rng = np.random.default_rng(seed=seed)
    extracted_images = (image.copy() & 0b00001111) << 4
    img_size_x, img_size_y, _ = extracted_images.shape
    x_step_size = img_size_x // 2
    y_step_size = img_size_y // 2
    
    # Use rng to reshuffle indices to know and address which pixels need to go where 
    indices = []
    for _ in range(4):
        idx = np.arange(x_step_size * y_step_size * 3)
        rng.shuffle(idx)
        indices.append(idx)

    # Split image in 4 equal parts and undo shuffling effect
    parts = []

    for i in range(2):
        for j in range(2):
            part = np.zeros((x_step_size, y_step_size, 3), dtype=np.uint8).ravel()

            idx = indices[2*i + j]
            part[idx] = extracted_images[i * x_step_size:(i+1) * x_step_size, j * y_step_size:(j+1) * y_step_size, :].ravel()
            part = part.reshape((x_step_size, y_step_size, 3))
            parts.append(part)

    # Determine the median image
    median_image = np.median(parts, axis=0).astype(np.uint8)
    # Optional: Do post-processing
    if do_post_processing:
        median_image = post_processing(median_image)

    # Scale back up to original size
    # TWO OPTIONS: Either we scale down the tampered image to compare against the median
    # or we scale up the median to compare against the tampered
    # The first seems more faithful, as only then we are sure that unchanged pixels are exactly the same
    
    tampered_image_small = cv2.resize(image, (y_step_size, x_step_size))

    # Tamper localization
    # First, remove the watermark from the tampered image
    tampered_image_small = tampered_image_small & 0b11110000
    # Then, create a binary mask which indicates where the image was modified
    # Current comparison uses subtraction, are there better methods?
    tamper_mask = np.abs(tampered_image_small.astype(float) - median_image.astype(float))
    # Tamper threshold
    tt = 5
    tamper_mask = (tamper_mask >= tt).astype(np.uint8)

    # In the case of RGB: A tampered image should have all three channels modified (own addition)
    # Go from 3 channels to grayscale
    tamper_mask = np.min(tamper_mask, axis=2).astype(np.uint8)
    tamper_mask = np.repeat(tamper_mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    # Now reconstruct the original image based on the median image and the tamper mask
    # First, determine which sub-image to use for recovery
    masked_median = median_image * tamper_mask
    best_image = 0
    best_score = 0
    multiple_candidates = False
    for i in range(len(parts)):
        part = parts[i]
        masked_part = part * tamper_mask

        mse = np.round(mean_squared_error(image0=masked_median, image1=masked_part), decimals=2)
        if mse < best_score:
            best_score = mse
            best_image = i
        elif mse == best_score:
            multiple_candidates = True

    # Then, scale-up the recovery and the mask
    if multiple_candidates:
        recovery_image = cv2.resize(median_image, (img_size_y, img_size_x))
    else:
        recovery_image = cv2.resize(parts[best_image], (img_size_y, img_size_x))
    recovery_mask = cv2.resize(tamper_mask, (img_size_y, img_size_x))

    # In the case that multiple candidates are equally optimal, choose the median instead
    untamper_mask = np.logical_not(recovery_mask).astype(np.uint8)
    untampered_image = image * untamper_mask
    
    reconstruction_portion = recovery_image * recovery_mask
    recovered_image = untampered_image | reconstruction_portion

    recovery_mask_2d = recovery_mask[...,0] | recovery_mask[...,1] | recovery_mask[...,2]

    return recovery_mask_2d, recovered_image

def recovery(image, recovered_image):
    return recovered_image
