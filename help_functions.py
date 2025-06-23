"""
Module implementing help functions for the rest of the project
"""

import matplotlib.pyplot as plt
import numpy as np
from enum import Enum, auto
import cv2
import copy
# from cv2 import resize


class BinaryClassification:

    def __init__(self, y_true, y_pred):
        numel_true = len(y_true)
        numel_pred = len(y_pred)

        assert numel_pred == numel_true, "True and prediction lists have a different number of elements"
        
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        for i in range(numel_true):
            y_t = y_true[i]
            y_p = y_pred[i]
            if y_t == 0:
                if y_p == 0:
                    self.tn += 1
                else:
                    self.fp += 1
            else:
                if y_p == 0:
                    self.fn += 1
                else:
                    self.tp += 1

    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    def precision(self):
        if (self.tp + self.fp) == 0:
            return 0
        return self.tp / (self.tp + self.fp)
    
    def f1(self):
        p = self.precision()
        r = self.recall()

        return 2 * p * r / (p + r)
    
    def fpr(self):
        return self.fp / (self.fp + self.tn)
    
    def fnr(self):
        return self.fn / (self.fn + self.tp)
    


def flip_binary_array(array):
    """
    For an input array with only 1s and 0s, turn every 1 into a 0 and 
    every 0 into a 1 (binary NOT operation)
    """
    og_dtype = array.dtype

    return (array.astype(float) * (-1) + 1).astype(og_dtype)


def grow_tamper_pixels_single_channel(tamper_mask, step=1):
    size_x, size_y = tamper_mask.shape[:2]
    tamper_mask_updated = tamper_mask.copy()

    for x in range(size_x):
        for y in range(size_y):
            if tamper_mask[x,y] == 1:
                # Get neighbouring pixel range
                min_x = max(0, x - step)
                max_x = min(size_x, x + step)
                min_y = max(0, y - step)
                max_y = min(size_y, y + step)
                tamper_mask_updated[min_x:max_x, min_y:max_y] = 1

    return tamper_mask_updated

def fix_voids_using_interpolation(void_watermark, rec_watermark, size_x, size_y):
    changes=1
    rec_w = copy.deepcopy(rec_watermark.copy())
    voids_w = copy.deepcopy(void_watermark.copy())
    # voids_w = void_watermark.copy()
    n_its = 0

    # print("Start interpolation with {} voids remaining".format(np.sum(voids_w)))
    # Keep repeating as long as there are void pixels and as there are changes
    # Changes are added to avoid infinite loops
    while np.sum(voids_w) > 0 and changes > 0:
        changes = 0
        for x in range(size_x):
            for y in range(size_y):
                if voids_w[x, y]:
                    # Determine neighbour borders (account for edges)
                    x_min = max(0, x - 1)
                    x_max = min(size_x, x + 2)
                    y_min = max(0, y - 1)
                    y_max = min(size_y, y + 2)
                    # Count void neighbours
                    n_voids = np.sum(voids_w[x_min:x_max, y_min:y_max]) - 1
                    n_pixels = (x_max - x_min) * (y_max - y_min) - 1
                    # If less than 6 of the neighbouring pixels are void
                    if n_voids < 0.75*n_pixels:
                        # Use the valid neighbours to update the pixel
                        # We use the average, seems most appropriate to account for sudden changes as opposed to median
                        # Use np.zeros for intitialization to account for 3-channel (RGB) inputs as well as grayscale
                        pixel_sum = np.zeros(rec_w[x, y].shape, dtype=int)
                        n_pixels = 0
                        for i in range(x_min, x_max):
                            for j in range(y_min, y_max):
                                if voids_w[i, j] == 0:
                                    pixel_sum += rec_w[i, j].astype(int)
                                    n_pixels += 1
                        rec_w[x, y] = (pixel_sum / n_pixels).astype(np.uint8)
                        voids_w[x, y] = 0
                        changes += 1
        n_its += 1
    # print("Stopped because", np.sum(voids_w), changes)
    print("Pixel interpolation needed {} iterations".format(n_its))
    # print("Finished interpolation with {} voids remaining".format(np.sum(voids_w)))
    return rec_w, voids_w


def fix_voids_using_interpolation_(void_watermark, rec_watermark, size_x, size_y):
    changes=1
    rec_w = copy.deepcopy(rec_watermark.copy())
    voids_w = copy.deepcopy(void_watermark.copy())
    # voids_w = void_watermark.copy()
    n_its = 0

    # First iteration: Use to build a list of unresolved voids
    void_coords_list = [] # Unresolved voids

    for x in range(size_x):
        for y in range(size_y):
            if voids_w[x, y]:
                # Determine neighbour borders (account for edges)
                x_min = max(0, x - 1)
                x_max = min(size_x, x + 2)
                y_min = max(0, y - 1)
                y_max = min(size_y, y + 2)
                # Count void neighbours
                n_voids = np.sum(voids_w[x_min:x_max, y_min:y_max]) - 1
                n_pixels = (x_max - x_min) * (y_max - y_min) - 1
                # If less than 6 of the neighbouring pixels are void
                if n_voids < 0.75*n_pixels:
                    # Use the valid neighbours to update the pixel
                    # We use the average, seems most appropriate to account for sudden changes as opposed to median
                    # Use np.zeros for intitialization to account for 3-channel (RGB) inputs as well as grayscale
                    pixel_sum = np.zeros(rec_w[x, y].shape, dtype=int)
                    n_pixels = 0
                    for i in range(x_min, x_max):
                        for j in range(y_min, y_max):
                            if voids_w[i, j] == 0:
                                pixel_sum += rec_w[i, j].astype(int)
                                n_pixels += 1
                    rec_w[x, y] = (pixel_sum / n_pixels).astype(np.uint8)
                    voids_w[x, y] = 0
                    changes += 1
                else:
                    void_coords_list.append((x, y))

    n_its += 1

    # Keep repeating as long as there are void pixels and as there are changes
    # Changes are added to avoid infinite loops
    new_void_coords_list = []
    while np.sum(voids_w) > 0 and changes > 0:
        changes = 0
        for x,y in void_coords_list:
            if voids_w[x, y]:
                # Determine neighbour borders (account for edges)
                x_min = max(0, x - 1)
                x_max = min(size_x, x + 2)
                y_min = max(0, y - 1)
                y_max = min(size_y, y + 2)
                # Count void neighbours
                n_voids = np.sum(voids_w[x_min:x_max, y_min:y_max]) - 1
                n_pixels = (x_max - x_min) * (y_max - y_min) - 1
                # If less than 6 of the neighbouring pixels are void
                if n_voids < 0.75*n_pixels:
                    # Use the valid neighbours to update the pixel
                    # We use the average, seems most appropriate to account for sudden changes as opposed to median
                    # Use np.zeros for intitialization to account for 3-channel (RGB) inputs as well as grayscale
                    pixel_sum = np.zeros(rec_w[x, y].shape, dtype=int)
                    n_pixels = 0
                    for i in range(x_min, x_max):
                        for j in range(y_min, y_max):
                            if voids_w[i, j] == 0:
                                pixel_sum += rec_w[i, j].astype(int)
                                n_pixels += 1
                    rec_w[x, y] = (pixel_sum / n_pixels).astype(np.uint8)
                    voids_w[x, y] = 0
                    changes += 1
                else:
                    new_void_coords_list.append((x, y))
        
        n_its += 1
        void_coords_list = new_void_coords_list.copy()
        new_void_coords_list = []
    # print("Stopped because", np.sum(voids_w), changes)
    # print("Pixel interpolation needed {} iterations".format(n_its))
    # print("Finished interpolation with {} voids remaining".format(np.sum(voids_w)))
    return rec_w, voids_w


def fix_voids_using_interpolation_fast(void_watermark, rec_watermark, size_x, size_y, min_valid=4):
    import copy
    changes=1
    rec_w = copy.deepcopy(rec_watermark.copy())
    voids_w = copy.deepcopy(void_watermark.copy())
    # voids_w = void_watermark.copy()
    n_its = 1

    # np.seterr(all='warn')

    void_coords_list = []
    # First pass:
    for x in range(size_x):
        for y in range(size_y):
            if voids_w[x, y]:
                # Determine neighbour borders (account for edges)
                x_min = max(0, x - 1)
                x_max = min(size_x, x + 2)
                y_min = max(0, y - 1)
                y_max = min(size_y, y + 2)
                # Count void neighbours
                n_voids = np.sum(voids_w[x_min:x_max, y_min:y_max]) - 1
                n_pixels = (x_max - x_min) * (y_max - y_min) - 1
                # If less than 6 of the neighbouring pixels are void
                pixel_sum = np.zeros(rec_w[x, y].shape, dtype=int)
                # n_pixels = 0
                if (n_pixels - n_voids) >= min_valid:
                    for i in range(x_min, x_max):
                        for j in range(y_min, y_max):
                            if voids_w[i, j] == 0:
                                pixel_sum += rec_w[i, j].astype(int)
                                n_pixels += 1
                    rec_w[x, y] = (pixel_sum / n_pixels).astype(np.uint8)
                    voids_w[x, y] = 0
                else:
                    void_coords_list.append((x, y))

    # For the next passess, only consider the remaining voids (to avoid looping over the entire array the entire time)
    changes = 1
    rem_void_coords_list = []
    while len(void_coords_list) > 0 and changes > 0:
        changes = 0
        n_its += 1
        for x, y in void_coords_list:
            # Determine neighbour borders (account for edges)
            x_min = max(0, x - 1)
            x_max = min(size_x, x + 2)
            y_min = max(0, y - 1)
            y_max = min(size_y, y + 2)
            # Count void neighbours
            n_voids = np.sum(voids_w[x_min:x_max, y_min:y_max]) - 1
            n_pixels = (x_max - x_min) * (y_max - y_min) - 1

            pixel_sum = np.zeros(rec_w[x, y].shape, dtype=int)
            
            if (n_pixels - n_voids) >= min_valid:
                for i in range(x_min, x_max):
                    for j in range(y_min, y_max):
                        if voids_w[i, j] == 0:
                            pixel_sum += rec_w[i, j].astype(int)
                            n_pixels += 1
                rec_w[x, y] = (pixel_sum / n_pixels).astype(np.uint8)
                voids_w[x, y] = 0
                changes += 1
            else:
                rem_void_coords_list.append((x, y))
        
        void_coords_list = rem_void_coords_list.copy()
        rem_void_coords_list = []
    
    print("Stopped because", len(void_coords_list), changes)
    print("Pixel interpolation needed {} iterations".format(n_its))

    return rec_w, voids_w


def reduce_img_size(input_image, **kwargs):
    new_size_x = kwargs.get("new_size_x", input_image.shape[0] // 2)
    new_size_y = kwargs.get("new_size_y", input_image.shape[1] // 2)
    return cv2.resize(input_image, (new_size_y, new_size_x))

def do_inpainting(attacked_image, tamper_mask, recovery_watermark, do_sharpening=True):
    full_size_x, full_size_y = attacked_image.shape[:2]
    tamper_mask_full_size = (reduce_img_size(input_image=tamper_mask * 255, 
                                             new_size_x=full_size_x, 
                                             new_size_y=full_size_y) > 0).astype(np.uint8)
    
    recovery_watermark_full_size = reduce_img_size(input_image=recovery_watermark, new_size_x=full_size_x, new_size_y=full_size_y).astype(np.uint8)
    
    if do_sharpening:
        recovery_watermark_full_size = sharpen(recovery_watermark_full_size)
    
    to_keep = (tamper_mask_full_size < 1).astype(np.uint8)
    to_replace = tamper_mask_full_size
    return (attacked_image* to_keep) | (recovery_watermark_full_size * to_replace), tamper_mask_full_size


def sharpen(recovery_watermark):
    derivative = cv2.Laplacian(recovery_watermark, ddepth=cv2.CV_8U)
    sharpened = recovery_watermark + derivative

    return sharpened

def maxpool2d(a, size_x, size_y, k=2):
    """
    Take the maxpool of a 2D matrix using a k x k kernel.
    Only works if no padding is required (e.g. if size_x % k == 0 and size_y % k == 0)
    """
    xk = size_x // k
    yk = size_y // k
    return a[:xk*k, :yk*k].reshape(xk, k, yk, k).max(axis=(1,3))

def display_images(images, title=""):
    print("Submitted {} images".format(len(images)))
    
    # Find cmap information for each image
    cmaps = []
    for image in images:
        if len(image.shape) == 3:
            cmaps.append(None)
        else:
            cmaps.append("gray")
    
    n_images = len(images)
    x_images = max(n_images // 2, 1)
    y_images = max(int(np.ceil(n_images / 2)), 1)


    if n_images == 1:
        fig, axes = plt.subplots(x_images, y_images)
        axes.imshow(images[0], cmap=cmaps[0])
    elif n_images == 2:
        fig, axes = plt.subplots(1, 2)
        for i in range(2):
            axes[i].imshow(images[i], cmap=cmaps[i])
    elif n_images == 3:
        fig, axes = plt.subplots(1, 3)
        for i in range(3):
            axes[i].imshow(images[i], cmap=cmaps[i])
    else:
        fig, axes = plt.subplots(x_images, y_images)
        for x in range(x_images):
            for y in range(y_images):
                axes[x, y].imshow(images[x * 2 + y], cmap=cmaps[x * 2 + y])
    fig.suptitle(t=title)
    plt.show()