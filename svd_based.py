"""
Based on paper:
A. Shehab, M. Elhoseny, K. Muhammad, A. K. Sangaiah, P. Yang,
H. Huang, and G. Hou, “Secure and robust fragile watermarking scheme
for medical images,” IEEE Access, vol. 6, pp. 10269–10278, 2018.
"""

import numpy as np

# Arnold transform
# Is actually called Arnold's cat map
# https://en.wikipedia.org/wiki/Arnold%27s_cat_map

def arnold_transform_scramble(image, iterations, dtype=np.uint8):
    #image=np.array(image)
    
    size_x, size_y = image.shape[:2]
    scrambled = np.zeros((image.shape), dtype=dtype)
    for x in range(size_x):
        for y in range(size_y):
            new_x = (2 * x + y) % size_x
            new_y = (x + y) % size_y
            scrambled[new_x, new_y] = image[x, y]
    if iterations > 1:
        return arnold_transform_scramble(scrambled, iterations - 1, dtype=dtype)
    else:
        return scrambled

def arnold_transform_unscramble(image, iterations, dtype=np.uint8):
    #image=np.array(image)
    size_x, size_y = image.shape[:2]
    unscrambled = np.zeros((image.shape), dtype=dtype)
    for x in range(size_x):
        for y in range(size_y):
            new_x = (2 * x + y) % size_x
            new_y = (x + y) % size_y
            unscrambled[x, y]= image[new_x, new_y]
    if iterations > 1:
        return arnold_transform_unscramble(unscrambled, iterations - 1, dtype=dtype)
    else:
        return unscrambled


def write_32bits_to_4x4_lsbs(block, integer):
    for x in range(4):
        for y in range(4):
            offset = (x * 4 + y) * 2
            and_mask = (2 ** offset) + (2 ** (offset + 1))
            lsbs_to_embed = np.astype((integer & and_mask) >> offset, np.uint8)
            # print("{} & {} >> {}".format(integer, and_mask, offset))
            block[x, y] |= lsbs_to_embed

    # print(block & 3)
    return block


def read_32bits_from_4x4_lsbs(block, channels=3):
    # print(block & 3)
    integer = np.zeros(channels, dtype=np.uint)
    for x in range(4):
        for y in range(4):
            offset = (x * 4 + y) * 2
            # We need to do this seperate step to allow for the astype
            # As otherwise the read value from block[x,y] will be uint8, which cannot be left shifted for more than 7
            # Resulting in an integer with only 8 relevant values
            auth_rec_bits = (block[x, y] & 3).astype(np.uint)
            integer |= (auth_rec_bits << offset)

    return integer


def do_svd_and_block_based_watermarking(input_img, key):
    
    block_size = 4

    base_x, base_y = input_img.shape[:2]

    # First, clear 2 LSBs
    without_lsb = input_img & 0b11111100

    # Also retrieve scrambled image
    x_blocks = base_x // block_size
    y_blocks = base_y // block_size
    
    block_map_base = np.arange(x_blocks * y_blocks, dtype=int).reshape((x_blocks, y_blocks))
    block_map_scrambled = arnold_transform_scramble(block_map_base, iterations=key, dtype=int)
    # print(block_map_scrambled.dtype)
    # sys.exit(0)
    pixels_scrambled = np.empty(input_img.shape, dtype=int)

    # Scramble pixels according to scrambled block map
    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            dst_x = block_x * block_size # Pixel x destination coord
            dst_y = block_y * block_size # Pixel y destination coord
            og_block_x = block_map_scrambled[block_x, block_y] // y_blocks
            og_block_y = block_map_scrambled[block_x, block_y] % y_blocks
            og_x = og_block_x * block_size # Pixel x original coord
            og_y = og_block_y * block_size # Pixel y original coord

            pixels_scrambled[dst_x:dst_x + block_size, dst_y: dst_y + block_size] = \
                input_img[og_x:og_x + block_size, og_y:og_y + block_size]
            
    # Process every block
    for x in range(0, base_x, block_size):
        for y in range(0, base_y, block_size):
            # For every channel in an rgb image
            ban_values = []
            for c in range(3):
                # Do singular value decomposition
                S = np.linalg.svd(without_lsb[x:x+block_size, y:y+block_size, c], 
                                  full_matrices=True, compute_uv=False)
                
                # Compute Block Authentication Number
                # Compute trace
                # (a trace is the sum of all diagonal elements)
                trace = np.sum(S)
                # Now adapt the trace to fit a 12-bit value
                ban = np.clip(np.round(trace, decimals=0), a_min=0, a_max=4096).astype(np.uint)
                ban_values.append(ban)

            # Compute averages
            # [ 00 01]
            # [ 10 11]
            inspected_maj_block = pixels_scrambled[x:x+block_size, y:y+block_size]
            # dtype is np.uint instead of np.uint8, as these values will be used to construct a 32-bit unsigned integer
            # setting uint8 will cause errors when shifting these values beyond the 8 bits
            block00_avg = np.average(inspected_maj_block[:2, :2], axis=(0, 1)).astype(np.uint)
            block01_avg = np.average(inspected_maj_block[2:, :2], axis=(0, 1)).astype(np.uint)
            block10_avg = np.average(inspected_maj_block[:2, 2:], axis=(0, 1)).astype(np.uint)
            block11_avg = np.average(inspected_maj_block[2:, 2:], axis=(0, 1)).astype(np.uint)

            # Construct 32-bit integer
            ban = np.array(ban_values, dtype=np.uint32)
            integer = (ban & 4095) << 20

            integer |= (((block00_avg & 0b11111000) >> 3) << 15)
            integer |= (((block01_avg & 0b11111000) >> 3) << 10)
            integer |= (((block10_avg & 0b11111000) >> 3) << 5)
            integer |= ((block11_avg & 0b11111000) >> 3)

            without_lsb[x:x+block_size, y:y+block_size] = write_32bits_to_4x4_lsbs(
                block=without_lsb[x:x+block_size, y:y+block_size], 
                integer=integer)
            # sys.exit(0)
    # print(without_lsb & 3)
    return without_lsb

def get_neighbourhood_pixels(block_x, block_y, unscrambled_tamper_blocks, recovered_image, block_size=4):
    base_x, base_y = recovered_image.shape[:2]
    x_blocks, y_blocks = unscrambled_tamper_blocks.shape[:2]
    # Try to get 8-adjacent neighbours
    # but consider the edges of the image
    x_min = max(block_x - 1, 0)
    x_max = min(block_x + 1, base_x)
    y_min = max(block_y - 1, 0)
    y_max = min(block_y + 1, base_y)

    acc =np.zeros(3, dtype=int)
    n_valids = 0
    for neighbour_x in range(x_min, x_max):
        for neighbour_y in range(y_min, y_max):
            if unscrambled_tamper_blocks[neighbour_x, neighbour_y] == 0:
                n_valids += 1
                pixel_x = neighbour_x * block_size
                pixel_y = neighbour_y * block_size
                val = np.average(recovered_image[pixel_x:pixel_x + block_size, 
                                                 pixel_y:pixel_y + block_size],
                                                 axis=(0, 1)).astype(int)
                acc += val
    
    if n_valids > 0:
        return (acc / n_valids).astype(np.uint8)
    else:
        # Return 0 if no valid neighbours are present
        return 0

def tamper_detection(input_img, key):
    block_size = 4
    base_x, base_y = input_img.shape[:2]
    x_blocks = base_x // block_size
    y_blocks = base_y // block_size

    base_x, base_y = input_img.shape[:2]
    without_lsb = input_img & 0b11111100

    # Also retrieve scrambled image
    # scrambled = arnold_transform_scramble(image=input_img, iterations=key)
    extracted_recovery_pixels = np.zeros(input_img.shape, dtype=np.uint8)
    tampered_pixels = np.zeros(input_img.shape[:2], dtype=np.uint8)
    tampered_blocks = np.zeros((x_blocks, y_blocks), dtype=np.uint8)
    
    block_map_base = np.arange(x_blocks * y_blocks, dtype=int).reshape((x_blocks, y_blocks))
    block_map_scrambled = arnold_transform_scramble(block_map_base, iterations=key, dtype=int)
    pixels_unscrambled = np.empty(input_img.shape, dtype=int)

    # Scramble pixels according to scrambled block map
    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            dst_x = block_x * block_size # Pixel x destination coord
            dst_y = block_y * block_size # Pixel y destination coord
            og_block_x = block_map_scrambled[block_x, block_y] // y_blocks
            og_block_y = block_map_scrambled[block_x, block_y] % y_blocks
            og_x = og_block_x * block_size # Pixel x original coord
            og_y = og_block_y * block_size # Pixel y original coord

            pixels_unscrambled[og_x:og_x + block_size, og_y:og_y + block_size] = \
                input_img[dst_x:dst_x + block_size, dst_y: dst_y + block_size]

    # Process every block
    for x in range(0, base_x, block_size):
        for y in range(0, base_y, block_size):
            # For every channel in an rgb image
            ban_values = []
            for c in range(3):
                # Do singular value decomposition
                # U, S, Vh = np.linalg.svd(without_lsb[x:x+block_size, y:y+block_size, c], 
                #                         full_matrices=True)
                S = np.linalg.svd(without_lsb[x:x+block_size, y:y+block_size, c], 
                                  full_matrices=True, compute_uv=False)
                
                # Compute Block Authentication Number
                # Compute trace
                # (a trace is the sum of all diagonal elements)
                trace = np.sum(S)
                # Now adapt the trace to fit a 12-bit value
                ban = np.clip(np.round(trace, decimals=0), a_min=0, a_max=4096).astype(np.uint)
                ban_values.append(ban)
            
            # This first extraction gets the ban
            integer = read_32bits_from_4x4_lsbs(block=input_img[x:x+block_size, y:y+block_size],
                                                channels=3)
            # The second (unscrambled) one gets the recovery pixels
            integer2 = read_32bits_from_4x4_lsbs(block=pixels_unscrambled[x:x+block_size, y:y+block_size],
                                                channels=3)
            # print("{:d} = {:b}".format(integer[0], integer[0]))
            extracted_ban = np.array(ban_values)
            retrieved_ban = (integer >> 20) & 4095
            block00_avg = ((integer2 >> 12) & 0b11111000)
            block01_avg = ((integer2 >> 7) & 0b11111000)
            block10_avg = ((integer2 >> 2) & 0b11111000)
            block11_avg = ((integer2 << 3) & 0b11111000)

            # Store tamper detection results
            # tampered_blocks[x // block_size, y // block_size] = 0 if np.array_equal(extracted_ban, retrieved_ban) else 1
            if not np.array_equal(extracted_ban, retrieved_ban):
                tampered_pixels[x:x+block_size, y:y+block_size] = 1
                tampered_blocks[x // block_size, y // block_size] = 1
            
            # Store reconstruction pixels
            inspected_maj_block = extracted_recovery_pixels[x:x+block_size, y:y+block_size]
            inspected_maj_block[:2, :2] = block00_avg
            inspected_maj_block[2:, :2] = block01_avg
            inspected_maj_block[:2, 2:] = block10_avg
            inspected_maj_block[2:, 2:] = block11_avg

    # Scale up the tamper mask
    tamper_mask_fs = np.zeros((base_x, base_y), dtype=np.uint8)
    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            pixel_x = block_x * block_size
            pixel_y = block_y * block_size
            tamper_mask_fs[pixel_x:pixel_x + block_size, pixel_y:pixel_y + block_size] = tampered_blocks[block_x, block_y]

    return tamper_mask_fs, tampered_pixels, tampered_blocks, extracted_recovery_pixels

def recovery(input_img, key, tampered_pixels, tampered_blocks, extracted_recovery_pixels):
    block_size = 4
    base_x, base_y = input_img.shape[:2]
    x_blocks = base_x // block_size
    y_blocks = base_y // block_size

    if np.sum(tampered_pixels) > 1:
        # Unscramble the recovery pixels        
        untampered_pixels = 1 - tampered_pixels
        untampered_pixels = np.repeat(untampered_pixels[:, :, np.newaxis], 3, axis=2)
        tampered_pixels = np.repeat(tampered_pixels[:, :, np.newaxis], 3, axis=2)
        recovered_image = input_img * untampered_pixels + extracted_recovery_pixels * tampered_pixels
        # display_images([input_img, tampered_pixels * 255, unscrambled_recovery, recovered_image])

         # Unscramble tamper blocks to find tampered recovery blocks
        unscrambled_tamper_blocks = arnold_transform_unscramble(tampered_blocks, iterations=key)

        # Infer neighbourhood information for each tampered block
        for block_x in range(x_blocks):
            for block_y in range(y_blocks):
                # Only consider the tampered area
                if tampered_blocks[block_x, block_y] == 1:
                    # If recovery is affected
                    if unscrambled_tamper_blocks[block_x, block_y] == 1:
                        pixel_x = block_x * block_size
                        pixel_y = block_y * block_size
                        
                        recovered_image[pixel_x:pixel_x + block_size, 
                                        pixel_y:pixel_y + block_size] = \
                                        get_neighbourhood_pixels(block_x, block_y, 
                                                                 unscrambled_tamper_blocks,
                                                                 recovered_image, block_size=4)

        return recovered_image

    else:
        return input_img
