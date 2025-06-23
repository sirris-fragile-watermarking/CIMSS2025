"""
This code implements the paper:
Approximate regeneration of image using fragile watermarking for tamper detection and recovery in real time
by Varsha Sisaudia, Virendra P. Vishwakarma
https://doi.org/10.1007/s11042-024-18247-3
"""

import numpy as np

def extract_bits(start_x, start_y, input_array):
    # We have a 4x4 block for which we must calculate both the LBP as well as the averages of all sub-blocks.
    # First do the averaging
    recovery_values = []
    for i in range(2):
        for j in range(2):
            # Averaging using basic numpy operations
            input_block = input_array[start_x + i*2:start_x + (i+1)*2, start_y + j*2: start_y + (j+1)*2].astype(float)

            avg = np.average(input_block, axis=(0, 1)).astype(np.uint8)
            # We only keep the 5 MSB
            recovery_values.append((avg & 0b11111000) >> 3)


    # Now also calculate LBP
    middle_avg = np.average(input_array[start_x + 1:start_x + 3, start_y +1: start_y + 3], axis=(0, 1))
    lbp = (input_array[start_x:start_x + 4, start_y: start_y + 4] >= middle_avg).astype(np.uint8)
    # Note the order of the LBP bits
    # Should be → ↓ ← ↑ for a start at (0, 0), which is the top left corner
    lbp_values = [
        lbp[0,0], lbp[0,1], lbp[0,2], lbp[0,3], # →
        lbp[1,3], lbp[2,3], lbp[3,3],           # ↓
        lbp[3,2], lbp[3,1], lbp[3,0],           # ←
        lbp[2,0], lbp[1,0]                      # ↑
    ]

    watermark_values = [lbp_values[i] ^ lbp_values[i + 6] for i in range(6)]

    return watermark_values, recovery_values


def embed_bits(og_start_x, og_start_y, corr_start_x, corr_start_y, watermark_bits, recovery_bits, embedding):
    """
    og_start_x: Starting x coordinate for originating sub-block
    og_start_y: Starting y coordinate for originating sub-block
    corr_start_x: Starting x coordinate for originating sub-block that is paired with the original block
    corr_start_y: Starting y coordinate for originating sub-block that is paired with the original block
    """
    # As the embedding approach is fixed and lacks a logical structure,
    # we hardcode it for a fixed 16-pixel block
    embedding[og_start_x,     og_start_y] = (watermark_bits[0] << 1) | watermark_bits[1]
    embedding[og_start_x + 1, og_start_y] = (watermark_bits[2] << 1) | watermark_bits[3]
    embedding[og_start_x + 2, og_start_y] = (watermark_bits[4] << 1) | watermark_bits[5]
    embedding[corr_start_x + 3, corr_start_y] = embedding[og_start_x, og_start_y]
    embedding[corr_start_x, corr_start_y + 1] = embedding[og_start_x + 1, og_start_y]
    embedding[corr_start_x + 1, corr_start_y + 1] = embedding[og_start_x + 2, og_start_y]
    embedding[corr_start_x + 2, corr_start_y + 1] = (recovery_bits[0] & 0b00011000) >> 3
    embedding[corr_start_x + 3, corr_start_y + 1] = (recovery_bits[0] & 0b00000110) >> 1
    embedding[corr_start_x,     corr_start_y + 2] = ((recovery_bits[0] & 1) << 1) | ((recovery_bits[1] & 0b00010000) >> 4)
    embedding[corr_start_x + 1, corr_start_y + 2] = (recovery_bits[1] & 0b00001100) >> 2
    embedding[corr_start_x + 2, corr_start_y + 2] = recovery_bits[1] & 0b00000011
    embedding[corr_start_x + 3, corr_start_y + 2] = (recovery_bits[2] & 0b00011000) >> 3
    embedding[corr_start_x,     corr_start_y + 3] = (recovery_bits[2] & 0b00000110) >> 1
    embedding[corr_start_x + 1, corr_start_y + 3] = ((recovery_bits[2] & 1) << 1) | ((recovery_bits[3] & 0b00010000) >> 4)
    embedding[corr_start_x + 2, corr_start_y + 3] = (recovery_bits[3] & 0b00001100) >> 2
    embedding[corr_start_x + 3, corr_start_y + 3] =  recovery_bits[3] & 0b00000011


def get_block_recovery_bits(start_x, start_y, watermarked_image):
    # watermarked_img_msb = watermarked_image & 0b11111100
    watermarked_img_lsb = watermarked_image & 0b00000011

    # Extract the recovery bits
    recovery_bits = np.zeros((4, 3), dtype=np.uint8)

    recovery_bits[0] = (watermarked_img_lsb[start_x + 2, start_y + 1] << 3)
    recovery_bits[0] |= (watermarked_img_lsb[start_x + 3, start_y + 1] << 1)
    recovery_bits[0] |= ((watermarked_img_lsb[start_x, start_y + 2] & 0b00000010) >> 1)

    recovery_bits[1] = ((watermarked_img_lsb[start_x, start_y + 2] & 1) << 4)
    recovery_bits[1] |= (watermarked_img_lsb[start_x + 1, start_y + 2] << 2)
    recovery_bits[1] |= watermarked_img_lsb[start_x + 2, start_y + 2]

    recovery_bits[2] = (watermarked_img_lsb[start_x + 3, start_y + 2] << 3)
    recovery_bits[2] |= (watermarked_img_lsb[start_x, start_y + 3] << 1)
    recovery_bits[2] |= ((watermarked_img_lsb[start_x + 1, start_y + 3] & 0b00000010) >> 1)

    recovery_bits[3] = ((watermarked_img_lsb[start_x + 1, start_y + 3] & 1) << 4)
    recovery_bits[3] |= (watermarked_img_lsb[start_x + 2, start_y + 3] << 2)
    recovery_bits[3] |= watermarked_img_lsb[start_x + 3, start_y + 3]

    # Turn recovery bits into 8-bit numbers
    recovery_bits = recovery_bits << 3

    return recovery_bits

def get_block_auth_bits(start_x, start_y, watermarked):
    watermarked_msb = watermarked & 0b11111100

    # Retrieve embedded local auth bits
    l0l1 = watermarked[start_x, start_y] & 0b00000011
    l2l3 = watermarked[start_x + 1, start_y] & 0b00000011
    l4l5 = watermarked[start_x + 2, start_y] & 0b00000011
    local_bits = [l0l1 >> 1, l0l1 & 1, l2l3 >> 1, l2l3 & 1, l4l5 >> 1, l4l5 & 1]

    # Retrieve embedded global auth bits
    g0g1 = watermarked[start_x + 3, start_y] & 0b00000011
    g2g3 = watermarked[start_x, start_y + 1] & 0b00000011
    g4g5 = watermarked[start_x + 1, start_y + 1] & 0b00000011
    global_bits = [g0g1 >> 1, g0g1 & 1, g2g3 >> 1, g2g3 & 1, g4g5 >> 1, g4g5 & 1]

    # Verify local watermark bits
    # First compute LBP
    middle_avg = np.average(watermarked_msb[start_x + 1:start_x + 3, start_y +1: start_y + 3], axis=(0, 1))
    lbp = (watermarked_msb[start_x:start_x + 4, start_y: start_y + 4] >= middle_avg).astype(np.uint8)
    lbp_values = [
        lbp[0,0], lbp[0,1], lbp[0,2], lbp[0,3], # →
        lbp[1,3], lbp[2,3], lbp[3,3],           # ↓
        lbp[3,2], lbp[3,1], lbp[3,0],           # ←
        lbp[2,0], lbp[1,0]                      # ↑
    ]

    bits = []
    for i in range(6):
        watermark_value = lbp_values[i] ^ lbp_values[i + 6]
        bits.append(watermark_value)

    embedded_local_auth = local_bits
    embedded_global_auth = global_bits
    expected_local_auth = bits

    return embedded_local_auth, embedded_global_auth, expected_local_auth

def detect_tampering_bits(og_start_x, og_start_y, corr_start_x, corr_start_y, watermarked_image):
    """
    Return authenticity level
    0 = Original block has been tampered with
    1 = Original block is unmodified, but correlated block has been tampered with
    2 = Original and corresponding blocks are both unmodified
    """
    og_emb_local_auth, og_emb_global_auth, og_exp_local_auth = get_block_auth_bits(start_x=og_start_x, start_y=og_start_y, watermarked=watermarked_image)

    # Compare local expected and embedded auth bits
    og_lvl1 = np.array_equal(og_emb_local_auth, og_exp_local_auth)

    corr_emb_local_auth, corr_emb_global_auth, corr_exp_local_auth = get_block_auth_bits(start_x=corr_start_x, start_y=corr_start_y, watermarked=watermarked_image)
    corr_lvl1 = np.array_equal(corr_emb_local_auth, corr_exp_local_auth)

    lvl2_part1 = np.array_equal(og_emb_global_auth, corr_emb_local_auth)
    lvl2_part2 = np.array_equal(corr_emb_global_auth, og_emb_local_auth)
    lvl2 = lvl2_part1 and lvl2_part2

    if og_lvl1:
        if corr_lvl1:
            if lvl2:
                return 2
            else:
                return 1
        else:
            return 1
    else:
        return 0


def try_block_recovery(og_start_x, og_start_y, corr_start_x, corr_start_y, watermarked_image, recovered_img):
    recovery_bits = get_block_recovery_bits(start_x=corr_start_x, start_y=corr_start_y, watermarked_image=watermarked_image)

    # Replace the tampered bits:
    for i in range(2):
        for j in range(2):
            recovered_img[og_start_x + i*2:og_start_x + (i+1)*2, og_start_y + j*2: og_start_y + (j+1)*2] = recovery_bits[i * 2 + j]


def do_self_embedding_rgb(original_image, verbose=False):
    
    img_size_x, img_size_y, _ = original_image.shape

    # If the image sides are not divisible by 8, this embedding method will not work
    if img_size_x % 8 != 0:
        raise ValueError("Image with size {}x{} has an x-axis that is indivisible by 8".format(img_size_x, img_size_y))
    elif img_size_y % 8 != 0:
        raise ValueError("Image with size {}x{} has an y-axis that is indivisible by 8".format(img_size_x, img_size_y))

    # Embedding process
    # 1) Remove LSBs
    without_lsbs = original_image.copy() & 0b11111100

    # 2) Divide the image into two pairs:
    # Each sub-image has a size of (sub_size_x by sub_size_y)
    sub_size_x, sub_size_y = img_size_x // 2, img_size_y // 2

    # Determine starting coords for each of the sub images
    sub_image_coords = [(0, 0), (0, sub_size_y), (sub_size_x, 0), (sub_size_x, sub_size_y)]
    sub_image_corresp = [(sub_size_x, sub_size_y), (sub_size_x, 0), (0, sub_size_y), (0, 0)]
    # sub_image_corresp = [(0, sub_size_y), (sub_size_x, sub_size_y), (sub_size_x, 0), (0, 0)]

    # 3) Divide the image further into 4x4 blocks
    # Note: As each subimage should be divisible into 4x4 blocks, the overall image should be divisible by 8 on
    # both sides (to ensure that the number of 4x4 blocks is not odd)

    embeddings = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)

    # We will now iterate over each block to process them
    # First, select the sub image 
    for subimage_idx in range(4):
        # The, select the block in that image
        for x in range(sub_size_x // 4):
            for y in range(sub_size_y // 4):
                extract_start_x = sub_image_coords[subimage_idx][0] + x * 4
                extract_start_y = sub_image_coords[subimage_idx][1] + y * 4
                embed_start_x = sub_image_corresp[subimage_idx][0] + x * 4
                embed_start_y = sub_image_corresp[subimage_idx][1] + y * 4
                if verbose:
                    print("Square [({}x{}), ({}x{})] -> [({}x{}), ({}x{})]".format(
                        extract_start_x, extract_start_y, extract_start_x + 4, extract_start_y + 4,
                        embed_start_x, embed_start_y, embed_start_x + 4, embed_start_y + 4
                    ))
                watermark_bits, recovery_bits = extract_bits(start_x=extract_start_x, 
                                                             start_y=extract_start_y, 
                                                             input_array=without_lsbs)
                embed_bits(
                    og_start_x=extract_start_x,
                    og_start_y=extract_start_y, 
                    corr_start_x=embed_start_x, 
                    corr_start_y=embed_start_y, 
                    watermark_bits=watermark_bits, 
                    recovery_bits=recovery_bits, 
                    embedding=embeddings)

    return without_lsbs | embeddings


def tamper_detection(input_img):
    img_size_x, img_size_y, _ = input_img.shape
    sub_size_x, sub_size_y = img_size_x // 2, img_size_y // 2

    # Determine starting coords for each of the sub images
    sub_image_coords = [(0, 0), (0, sub_size_y), (sub_size_x, 0), (sub_size_x, sub_size_y)]
    sub_image_corresp = [(sub_size_x, sub_size_y), (sub_size_x, 0), (0, sub_size_y), (0, 0)]
    
    # Retrieve image without LSBs for tamper detection and recovery
    without_lsbs = input_img.copy() & 0b11111100
    tampering_mask = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)
    recovery_mask = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)
    # recovered_img = np.zeros((img_size_x, img_size_y, 3), dtype=np.uint8)
    recovered_img = input_img.copy()

    x_blocks = sub_size_x // 4
    y_blocks = sub_size_y // 4
    for subimage_idx in range(4):
        # The, select the block in that image
        for x in range(x_blocks):
            for y in range(y_blocks):
                og_start_x = sub_image_coords[subimage_idx][0] + x * 4
                og_start_y = sub_image_coords[subimage_idx][1] + y * 4
                corr_start_x = sub_image_corresp[subimage_idx][0] + x * 4
                corr_start_y = sub_image_corresp[subimage_idx][1] + y * 4

                auth_level = detect_tampering_bits(
                    og_start_x=og_start_x, 
                    og_start_y=og_start_y,
                    corr_start_x=corr_start_x,
                    corr_start_y=corr_start_y,
                    watermarked_image=input_img)
                # Determine which regions have been tampered with
                if auth_level == 0:
                    tampering_mask[og_start_x:og_start_x + 4, og_start_y:og_start_y+4] = 2
                elif auth_level == 1:
                    tampering_mask[og_start_x:og_start_x + 4, og_start_y:og_start_y+4] = 1

                # If a block shows signs of tampering
                if auth_level < 1:
                    try_block_recovery(og_start_x=og_start_x, 
                                                  og_start_y=og_start_y,
                                                  corr_start_x=corr_start_x,
                                                  corr_start_y=corr_start_y,
                                                  watermarked_image=recovered_img,
                                                  recovered_img=recovered_img)

    tampering_mask = np.clip(tampering_mask, a_min=0, a_max=1)
    tampering_mask = tampering_mask[..., 0] | tampering_mask[..., 1] | tampering_mask[..., 2]
    return tampering_mask, recovered_img

def recovery(input_image, recovered_img):
    return recovered_img