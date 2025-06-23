"""
Implementation based on paper:
J. Molina-Garcia, B. P. Garcia-Salgado, V. Ponomaryov, R. Reyes-Reyes,
S. Sadovnychiy, and C. Cruz-Ramos, “An effective fragile watermarking
scheme for color image tampering detection and self-recovery,” Signal Pro-
cessing: Image Communication, vol. 81, p. 115725, 2020.
"""

import numpy as np
import cv2

def watermark_embedding(img, key_r, key_g, key_b):
    bs = 4
    size_x, size_y = img.shape[:2]
    # Extract recovery and authentication bits
    W1, Wcr, Wcb = extract_recovery_watermark(img=img, bs=bs)
    rgb_auth_bits = extract_authentication_watermark(img=img, bs=bs)
        
    # Prepare embeddings per channel:
    keys = [key_r, key_g, key_b]
    embeddings = np.zeros((size_x, size_y, 3), dtype=np.uint8)

    for channel in range(3):
        # Each channel uses a different key for permutation
        # aka: Uses a different seed for shuffling
        rng = np.random.default_rng(seed=keys[channel])

        # Randomly shuffle luminance block indices:
        lum_indices = [(x, y) for x in range(size_x // bs) for y in range(size_y // bs)]
        rng.shuffle(lum_indices)

        # Randomly shuffle chrominance bits
        cr_indices = np.arange(len(Wcr))
        cb_indices = np.arange(len(Wcb))
        rng.shuffle(cr_indices)
        rng.shuffle(cb_indices)

        # Get appropriate authentication bits
        Wa = rgb_auth_bits[..., channel]

        # Iterate over all image blocks
        x_blocks = size_x // bs
        y_blocks = size_y // bs
        for i in range(x_blocks):
            for j in range(y_blocks):
                x = i * bs
                y = j * bs
                list_idx = (i * y_blocks + j)
                
                # Get luminance 4x4 bits
                lum_x, lum_y = lum_indices[list_idx]
                lum_block = W1[lum_x * bs:lum_x * bs + bs, lum_y * bs:lum_y * bs + bs]
                
                # Concatenate chrominance and authentication bits
                cr_bits = Wcr[cr_indices[list_idx]]
                cb_bits = Wcb[cb_indices[list_idx]]
                auth_bits = Wa[list_idx]
                W2_block = encode_chrominance_and_authentication(cr_bits, cb_bits, auth_bits)
                
                # Store blocks for embedding
                embeddings[x:x+bs, y:y+bs, channel] = ((W2_block << 1) | lum_block)

    watermarked = (img & 0b11111000) | embeddings

    # Now utilize correction equation
    v = watermarked - img
    lsb3 = (img & 0b00000100) >> 2
    for x in range(size_x):
        for y in range(size_y):
            for c in range(3):
                if (v[x, y, c] == 3) and (lsb3[x, y, c] == 1):
                    watermarked[x, y, c] = img[x, y, c] - 1
                elif (v[x, y, c] == -3) and (lsb3[x, y, c] == 0):
                    watermarked[x, y, c] = img[x, y, c] + 1
    
    return watermarked

def tamper_detection(img, key_r, key_g, key_b):
    bs = 4
    size_x, size_y = img.shape[:2]
    # Extract embedded watermark
    rec_y_bits, rec_cr_bits, rec_cb_bits, rec_auth_bits = watermark_extraction(img, key_r, key_g, key_b, bs=bs)

    # Do tamper detectio
    # Start by extracting authentication watermarks again
    exp_auth_bits = extract_authentication_watermark(img=img, bs=4)
    
    # Iterate over image, flag each tampered block
    x_blocks = size_x // bs
    y_blocks = size_y // bs
    tamper_mask_3d = np.zeros((x_blocks, y_blocks, 3), dtype=np.uint8)

    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            list_idx = (block_x * y_blocks + block_y)
            for channel in range(3):
                if exp_auth_bits[list_idx, channel] != rec_auth_bits[list_idx, channel]:
                    tamper_mask_3d[block_x, block_y, channel] = 1
    
    # Aggregate tamper masks
    tamper_mask_2d = np.clip(np.sum(tamper_mask_3d, axis=2), a_min=0, a_max=1)

     # There is some tamper mask cleanup, but this is too vague to implement
    tamper_mask = tamper_mask_2d

    # Set tamper mask to full_size
    tamper_mask_fs = blocks_to_pixels(tamper_mask, bs=bs)
    return tamper_mask_fs, tamper_mask, tamper_mask_3d, rec_y_bits, rec_cr_bits, rec_cb_bits

def recovery(img, key_r, key_g, key_b, tamper_mask, tamper_mask_3d, rec_y_bits, rec_cr_bits, rec_cb_bits):
    if np.sum(tamper_mask) == 0:
        return img

    bs = 4
    size_x, size_y = img.shape[:2]
    x_blocks = size_x // bs
    y_blocks = size_y // bs

    # For recovery, shuffle the tamper mask in accordance with the keys to find tampered recovery
    # pixels
    keys = [key_r, key_g, key_b]
    rec_y_auth = np.zeros((tamper_mask_3d.shape), dtype=tamper_mask_3d.dtype)
    rec_cb_auth = np.zeros((size_x * size_y // 16, 3), dtype=tamper_mask_3d.dtype)
    rec_cr_auth = np.zeros((size_x * size_y // 16, 3), dtype=tamper_mask_3d.dtype)
    for channel in range(3):

        # Initialize rng
        rng = np.random.default_rng(seed=keys[channel])
        
        # Randomly shuffle luminance block indices:
        lum_indices = [(x, y) for x in range(size_x // bs) for y in range(size_y // bs)]
        rng.shuffle(lum_indices)

        # Randomly shuffle chrominance bits
        cr_indices = np.arange(x_blocks * y_blocks)
        cb_indices = np.arange(x_blocks * y_blocks)
        rng.shuffle(cr_indices)
        rng.shuffle(cb_indices)
    	
        # Now iterate over all blocks
        for i in range(x_blocks):
            for j in range(y_blocks):
                list_idx = (i * y_blocks + j)
                lum_x, lum_y = lum_indices[list_idx]
                tamper_state = tamper_mask[i, j]
                rec_y_auth[lum_x, lum_y, channel] = tamper_state
                rec_cr_auth[cr_indices[list_idx], channel] = tamper_state
                rec_cb_auth[cb_indices[list_idx], channel] = tamper_state

    # Reshape chrominance authentication to images
    rec_cr_auth = np.reshape(rec_cr_auth, shape=(x_blocks, y_blocks, 3))
    rec_cb_auth = np.reshape(rec_cb_auth, shape=(x_blocks, y_blocks, 3))

    # Transform the chromance bit images back to full scale
    rec_cr_bits = blocks_to_pixels(rec_cr_bits, bs=bs)
    rec_cb_bits = blocks_to_pixels(rec_cb_bits, bs=bs)

    # Now we can try to reconstruct the recovery watermarks
    aggregate_y_bits, tcp_y_bits = recover_tampered_pixels(rec_bits=rec_y_bits, auth_bits=rec_y_auth, size_x=size_x, size_y=size_y)
    aggregate_cr_bits, tcp_cr_bits = recover_tampered_pixels(rec_bits=rec_cr_bits, auth_bits=rec_cr_auth, size_x=size_x, size_y=size_y)
    aggregate_cb_bits, tcp_cb_bits = recover_tampered_pixels(rec_bits=rec_cb_bits, auth_bits=rec_cb_auth, size_x=size_x, size_y=size_y)

    # Now we must do post_processing
    # Turn halftoned Y-channel back into original Y-channel
    aggregate_y_bits = undo_halftoning(aggregate_y_bits)
    
    # Do inpainting of affected pixels
    inpainted_y_bits, rem_tcp_y_bits = do_tcp_inpainting(pixels=aggregate_y_bits, tcp=tcp_y_bits)
    inpainted_cr_bits, rem_tcp_cr_bits = do_tcp_inpainting(pixels=aggregate_cr_bits, tcp=tcp_cr_bits)
    inpainted_cb_bits, rem_tcp_cb_bits = do_tcp_inpainting(pixels=aggregate_cb_bits, tcp=tcp_cb_bits)

    # Do bilateral filtering
    y_bits = cv2.bilateralFilter(src=inpainted_y_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    cr_bits = cv2.bilateralFilter(src=inpainted_cr_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    cb_bits = cv2.bilateralFilter(src=inpainted_cb_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    
    # Transform back to RGB colorspace
    imgYCrCb = np.zeros((size_x, size_y, 3), dtype=np.uint8)
    imgYCrCb[..., 0] = y_bits
    imgYCrCb[..., 1] = cr_bits
    imgYCrCb[..., 2] = cb_bits
    imgRGB = cv2.cvtColor(imgYCrCb, cv2.COLOR_YCrCb2RGB)

    # Set tamper mask to full_size
    tamper_mask = blocks_to_pixels(tamper_mask, bs=bs)
    
    # Duplicate tamper mask across channels
    tamper_mask_3d = np.repeat(tamper_mask[..., None], repeats=3, axis=2)

    # Compute recovered image
    recovered = img * (1 - tamper_mask_3d) + imgRGB * tamper_mask_3d

    return recovered

def do_recovery(img, key_r, key_g, key_b, include_tamper_mask=False):
    bs = 4
    size_x, size_y = img.shape[:2]
    # Extract embedded watermark
    rec_y_bits, rec_cr_bits, rec_cb_bits, rec_auth_bits = watermark_extraction(img, key_r, key_g, key_b, bs=bs)

    # Do tamper detectio
    # Start by extracting authentication watermarks again
    exp_auth_bits = extract_authentication_watermark(img=img, bs=4)
    
    # Iterate over image, flag each tampered block
    x_blocks = size_x // bs
    y_blocks = size_y // bs
    tamper_mask_3d = np.zeros((x_blocks, y_blocks, 3), dtype=np.uint8)

    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            list_idx = (block_x * y_blocks + block_y)
            for channel in range(3):
                if exp_auth_bits[list_idx, channel] != rec_auth_bits[list_idx, channel]:
                    tamper_mask_3d[block_x, block_y, channel] = 1
    
    # Aggregate tamper masks
    tamper_mask_2d = np.clip(np.sum(tamper_mask_3d, axis=2), a_min=0, a_max=1)

    # There is some tamper mask cleanup, but this is too vague to implement
    tamper_mask = tamper_mask_2d

    if np.sum(tamper_mask) == 0:
        # Return img, as it is untampered
        if include_tamper_mask:
            return img, np.zeros(img.shape[:2], dtype=np.uint8)
        return img

    # For recovery, shuffle the tamper mask in accordance with the keys to find tampered recovery
    # pixels
    keys = [key_r, key_g, key_b]
    rec_y_auth = np.zeros((tamper_mask_3d.shape), dtype=tamper_mask_3d.dtype)
    rec_cb_auth = np.zeros((size_x * size_y // 16, 3), dtype=tamper_mask_3d.dtype)
    rec_cr_auth = np.zeros((size_x * size_y // 16, 3), dtype=tamper_mask_3d.dtype)
    for channel in range(3):

        # Initialize rng
        rng = np.random.default_rng(seed=keys[channel])
        
        # Randomly shuffle luminance block indices:
        lum_indices = [(x, y) for x in range(size_x // bs) for y in range(size_y // bs)]
        rng.shuffle(lum_indices)

        # Randomly shuffle chrominance bits
        cr_indices = np.arange(x_blocks * y_blocks)
        cb_indices = np.arange(x_blocks * y_blocks)
        rng.shuffle(cr_indices)
        rng.shuffle(cb_indices)
    	
        # Now iterate over all blocks
        for i in range(x_blocks):
            for j in range(y_blocks):
                list_idx = (i * y_blocks + j)
                lum_x, lum_y = lum_indices[list_idx]
                tamper_state = tamper_mask[i, j]
                rec_y_auth[lum_x, lum_y, channel] = tamper_state
                rec_cr_auth[cr_indices[list_idx], channel] = tamper_state
                rec_cb_auth[cb_indices[list_idx], channel] = tamper_state

    # Reshape chrominance authentication to images
    rec_cr_auth = np.reshape(rec_cr_auth, shape=(x_blocks, y_blocks, 3))
    rec_cb_auth = np.reshape(rec_cb_auth, shape=(x_blocks, y_blocks, 3))

    # Transform the chromance bit images back to full scale
    rec_cr_bits = blocks_to_pixels(rec_cr_bits, bs=bs)
    rec_cb_bits = blocks_to_pixels(rec_cb_bits, bs=bs)

    # Now we can try to reconstruct the recovery watermarks
    aggregate_y_bits, tcp_y_bits = recover_tampered_pixels(rec_bits=rec_y_bits, auth_bits=rec_y_auth, size_x=size_x, size_y=size_y)
    aggregate_cr_bits, tcp_cr_bits = recover_tampered_pixels(rec_bits=rec_cr_bits, auth_bits=rec_cr_auth, size_x=size_x, size_y=size_y)
    aggregate_cb_bits, tcp_cb_bits = recover_tampered_pixels(rec_bits=rec_cb_bits, auth_bits=rec_cb_auth, size_x=size_x, size_y=size_y)

    # Now we must do post_processing
    # Turn halftoned Y-channel back into original Y-channel
    aggregate_y_bits = undo_halftoning(aggregate_y_bits)
    # Do inpainting of affected pixels
    
    inpainted_y_bits, rem_tcp_y_bits = do_tcp_inpainting(pixels=aggregate_y_bits, tcp=tcp_y_bits)
    inpainted_cr_bits, rem_tcp_cr_bits = do_tcp_inpainting(pixels=aggregate_cr_bits, tcp=tcp_cr_bits)
    inpainted_cb_bits, rem_tcp_cb_bits = do_tcp_inpainting(pixels=aggregate_cb_bits, tcp=tcp_cb_bits)

    # Do bilateral filtering
    y_bits = cv2.bilateralFilter(src=inpainted_y_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    cr_bits = cv2.bilateralFilter(src=inpainted_cr_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    cb_bits = cv2.bilateralFilter(src=inpainted_cb_bits, d=5, sigmaSpace=2.5, sigmaColor=7.0)
    
    # Transform back to RGB colorspace
    imgYCrCb = np.zeros((size_x, size_y, 3), dtype=np.uint8)
    imgYCrCb[..., 0] = y_bits
    imgYCrCb[..., 1] = cr_bits
    imgYCrCb[..., 2] = cb_bits
    imgRGB = cv2.cvtColor(imgYCrCb, cv2.COLOR_YCrCb2RGB)

    # Set tamper mask to full_size
    tamper_mask = blocks_to_pixels(tamper_mask, bs=bs)
    
    # Duplicate tamper mask across channels
    tamper_mask_3d = np.repeat(tamper_mask[..., None], repeats=3, axis=2)

    # Compute recovered image
    recovered = img * (1 - tamper_mask_3d) + imgRGB * tamper_mask_3d

    if include_tamper_mask:
        return recovered, tamper_mask

    return recovered

def undo_halftoning(img):
    
    # According to "N. Damera-Venkata, T. D. Kite, M. Venkataraman and B. L. Evans, 
    # "Fast blind inverse halftoning" 
    # 10.1109/ICIP.1998.723318
    g_sigma = np.sqrt(2.5)
    g_kernel = (9, 9)
    m1_kernel = 3
    blurred = cv2.GaussianBlur(img * 255, g_kernel, g_sigma).astype(np.uint8)
    S = cv2.medianBlur(blurred, ksize=m1_kernel)

    # Construct bandpass filter
    def bandpass_filter(S):
        S_lopass= cv2.GaussianBlur(img * 255, (13, 13), g_sigma).astype(np.uint8)
        S_hipass = S - S_lopass
        S_bandrej = S_lopass - S_hipass
        S_bandpass = np.absolute(img - S_bandrej)
        return S_bandpass
    B = bandpass_filter(S).astype(np.uint8)

    T_val = 20
    _, T = cv2.threshold(B, T_val, 255, cv2.THRESH_BINARY_INV)
    
    B_med = cv2.medianBlur(B, ksize=5).astype(np.uint8)

    E = B_med & T
    G = 1
    Y = S
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if E[x, y] == 1:
                Y[x, y] += G * B[x, y]

    return Y.astype(np.uint8)

def blocks_to_pixels(image, bs=4):
    x_blocks, y_blocks = image.shape[:2]
    size_x = x_blocks * bs
    size_y = y_blocks * bs

    if len(image.shape) == 3:
        pixels_image = np.zeros((size_x, size_y, 3), dtype=np.uint8)
    else:
        pixels_image = np.zeros((size_x, size_y), dtype=np.uint8)
    
    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            x = block_x * bs
            y = block_y * bs
            pixels_image[x:x+bs, y:y+bs] = image[block_x, block_y]
    
    return pixels_image

def do_tcp_inpainting(pixels, tcp):
    change = float("inf")
    its = 0
    start_tcp = np.sum(tcp)
    while change > 0:
        tcp_old = tcp.copy()
        pixels, tcp = __inpaint_tcp_pixels(pixels=pixels, tcp=tcp)
        change = np.sum(tcp_old) - np.sum(tcp)
        its += 1
    end_tcp = np.sum(tcp)

    return pixels, tcp

def __inpaint_tcp_pixels(pixels, tcp):
    size_x, size_y = pixels.shape[:2]
    
    pixels_out = pixels.copy()
    tcp_out = tcp.copy()

    for x in range(1, size_x - 1):
        for y in range(1, size_y -1):
            w_tcp = tcp[x-1:x+2, y-1:y+2]
            if w_tcp[1, 1] == 1:
                tot = 0
                for i in range(3):
                    for j in range(3):
                        tot += (1 - w_tcp[i, j])
                if tot > 1:
                    w_pixels = pixels[x-1:x+2, y-1:y+2]
                    pixels_maux = (w_pixels * (1 - w_tcp)).astype(float)
                    pixels_out[x, y] = (np.sum(pixels_maux) / tot.astype(float)).astype(np.uint8)
                    tcp_out[x, y] = 0

    return pixels_out, tcp_out

def recover_tampered_pixels(rec_bits, auth_bits, size_x, size_y, bs=4):
    aggregate_bits = rec_bits[..., 0]
    tcp_bits = np.zeros((size_x, size_y), dtype=np.uint8)

    x_blocks = size_x // bs
    y_blocks = size_y // bs
    for block_x in range(x_blocks):
        for block_y in range(y_blocks):
            x = block_x * bs
            y = block_y * bs
            # If R-channel has tampered block
            if auth_bits[block_x, block_y, 0] == 1:
                if auth_bits[block_x, block_y, 1] == 1:
                    if auth_bits[block_x, block_y, 2] == 0:
                        aggregate_bits[x:x+bs,y:y+bs] = rec_bits[x:x+bs,y:y+bs, 2]
                    else:
                        tcp_bits[x:x+bs,y:y+bs] = 1
                else:
                     aggregate_bits[x:x+bs,y:y+bs] = rec_bits[x:x+bs,y:y+bs, 1]

    return aggregate_bits, tcp_bits
    

def watermark_extraction(img, key_r, key_g, key_b, bs=4):
    size_x, size_y = img.shape[:2]

    # Received watermark bits
    rec_luminance_bits = img & 1
    rec_other_bits = (img & 2) >> 1
    
    # Extracted auth bits
    rec_auth_bits = np.zeros((size_x * size_y // bs, 3))

    # We retrieve 3 solutions for each of the various channels
    rec_y_bits = np.zeros((size_x, size_y, 3))
    rec_cr_bits = np.zeros((size_x * size_y // 16, 3))
    rec_cb_bits = np.zeros((size_x * size_y // 16, 3))
    keys = [key_r, key_g, key_b]
    for channel in range(3):
        rng = np.random.default_rng(seed=keys[channel])

        # Randomly shuffle luminance block indices:
        lum_indices = [(x, y) for x in range(size_x // bs) for y in range(size_y // bs)]
        rng.shuffle(lum_indices)

        # Randomly shuffle chrominance bits
        cr_indices = np.arange(rec_cr_bits.shape[0])
        cb_indices = np.arange(rec_cb_bits.shape[0])
        rng.shuffle(cr_indices)
        rng.shuffle(cb_indices)

        # Now iterate over each block, and extract the retained information
        x_blocks = size_x // bs
        y_blocks = size_y // bs
        for i in range(x_blocks):
            for j in range(y_blocks):
                x = i * bs
                y = j * bs
                list_idx = (i * y_blocks + j)

                # Recover luminance information
                lum_x, lum_y = lum_indices[list_idx]
                rec_y_bits[lum_x * bs:lum_x * bs + bs, lum_y * bs:lum_y * bs + bs, channel] = rec_luminance_bits[x:x+bs, y:y+bs, channel]
                
                # Recover chrominance and authentication info
                cr_bits, cb_bits, auth_bits = decode_chrominance_and_authentication(rec_other_bits[x:x+bs, y:y+bs, channel])
                rec_cr_bits[cr_indices[list_idx], channel] = cr_bits
                rec_cb_bits[cb_indices[list_idx], channel] = cb_bits
                rec_auth_bits[list_idx, channel] = auth_bits

    # Transform chrominance vectors back to images
    rec_cr_bits = np.reshape(rec_cr_bits, shape=(x_blocks, y_blocks, 3))
    rec_cb_bits = np.reshape(rec_cb_bits, shape=(x_blocks, y_blocks, 3))

    return rec_y_bits, rec_cr_bits, rec_cb_bits, rec_auth_bits

def encode_chrominance_and_authentication(cr_bits, cb_bits, auth_bits):
    block = np.zeros((4, 4), dtype=np.uint8)
    # Encode Cr
    block[0, 0] = (cr_bits & 128) >> 7
    block[0, 1] = (cr_bits & 64) >> 6
    block[0, 2] = (cr_bits & 32) >> 5
    block[0, 3] = (cr_bits & 16) >> 4
    block[1, 0] = (cr_bits & 8) >> 3
    block[1, 1] = (cr_bits & 4) >> 2
    # Encode Cb
    block[1, 2] = (cb_bits & 128) >> 7
    block[1, 3] = (cb_bits & 64) >> 6
    block[2, 0] = (cb_bits & 32) >> 5
    block[2, 1] = (cb_bits & 16) >> 4
    block[2, 2] = (cb_bits & 8) >> 3
    block[2, 3] = (cb_bits & 4) >> 2
    # Encode auth
    block[3, 0] = (auth_bits & 8) >> 3
    block[3, 1] = (auth_bits & 4) >> 2
    block[3, 2] = (auth_bits & 2) >> 1
    block[3, 3] = (auth_bits & 1) >> 0

    return block

def decode_chrominance_and_authentication(block):
    cr_bits = 0
    cb_bits = 0
    auth_bits = 0
    
    cr_bits = (block[0, 0] << 7) | (block[0, 1] << 6) | (block[0, 2] << 5) | (block[0, 3] << 4) | (block[1, 0] << 3) | (block[1, 1] << 2)
    cb_bits = (block[1, 2] << 7) | (block[1, 3] << 6) | (block[2, 0] << 5) | (block[2, 1] << 4) | (block[2, 2] << 3) | (block[2, 3] << 2)
    auth_bits = (block[3, 0] << 3) | (block[3, 1] << 2) | (block[3, 2] << 1) | block[3, 3]

    return cr_bits, cb_bits, auth_bits

def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.
    """
    return np.round(old_val * (nc - 1)) / (nc - 1)

def floyd_steinberg_dithering(img, new_bitrange=1):
    size_x, size_y = img.shape[:2]
    processing_array = np.array(img, dtype=float) / 255
    # print(processing_array[:10, :10])
    for y in range(size_y):
        for x in range(size_x):
            old_pixel = processing_array[x, y]
            new_pixel = get_new_val(old_val=old_pixel, nc=(2**new_bitrange))
            quant_error = old_pixel - new_pixel
            processing_array[x, y] = new_pixel
            
            # Now spread error over neighbours
            x_not_edge = x < (size_x - 1)
            y_not_edge = y < (size_y - 1)
            if x_not_edge:
                processing_array[x + 1][y] = processing_array[x + 1][y] + quant_error * 7 / 16
            if y_not_edge:
                processing_array[x][y + 1] = processing_array[x][y + 1] + quant_error * 5 / 16
                processing_array[x - 1][y + 1] = processing_array[x - 1][y + 1] + quant_error * 3 / 16
            if x_not_edge and y_not_edge:
                processing_array[x + 1][y + 1] = processing_array[x + 1][y + 1] + quant_error * 1 / 16

    return (processing_array * 255).astype(np.uint8)

def compute_block_average(block, mask=0b11111100):
    # Zero 3 LSBs
    block = block & 0b11111000

    # Compute average
    avg = np.average(block, axis=(0, 1)).astype(np.uint8)

    # Keep 6 MSB
    avg = avg & mask
    
    return avg

def get_block_auth_bits(block):
    avg = compute_block_average(block, mask=0b11111000)

    # Now xor neighbour bits
    b4 = (avg & 128) >> 7
    b3 = (avg & 64) >> 6
    b2 = (avg & 32) >> 5
    b1 = (avg & 16) >> 4
    b0 = (avg & 8) >> 3

    x3 = b4 ^ b3
    x2 = b3 ^ b2
    x1 = b2 ^ b1
    x0 = b1 ^ b0

    return (x3 << 3) | (x2 << 2) | (x1 << 1) | x0

def extract_authentication_watermark(img, bs=4):
    size_x, size_y = img.shape[:2]
    rgb_auth_bits = np.zeros((size_x // bs, size_y // bs, 3), dtype=img.dtype)

    for x in range(0, size_x, bs):
        for y in range(0, size_y, bs):
            rgb_auth_bits[x // bs, y // bs] = get_block_auth_bits(img[x:x+bs,y:y+bs])

    # Authentication bits need to be presented as vectors
    rgb_auth_bits = rgb_auth_bits.reshape((-1, 3))
    
    return rgb_auth_bits
    # return rgb_auth_bits[..., 0], rgb_auth_bits[..., 1], rgb_auth_bits[..., 2]

def extract_recovery_watermark(img, bs=4):
    size_x, size_y = img.shape[:2]
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    Y = imgYCC[..., 0]
    Cr = imgYCC[..., 1]
    Cb = imgYCC[..., 2]

    # Do halftoning on Y
    Y_dithered = floyd_steinberg_dithering(Y)
    # display_images([Y_dithered], "result of dithering")
    
    # currently outputs 255 instead 1, so shift to right
    Y_dithered = (Y_dithered >> 7).astype(dtype=np.uint8)

    # Obtain average chrominance for 4x4 blocks
    avg_chrominance = np.zeros((size_x // bs, size_y // bs, 2), dtype=img.dtype)

    for x in range(0, size_x, bs):
        for y in range(0, size_y, bs):
            # Block average
            avg_chrominance[x // bs, y // bs, 0] = compute_block_average(Cr[x:x+bs,y:y+bs])
            avg_chrominance[x // bs, y // bs, 1] = compute_block_average(Cb[x:x+bs,y:y+bs])

    # These averages need to be presented as a vector
    Cr_vector = np.reshape(avg_chrominance[..., 0], shape=(-1))
    Cb_vector = np.reshape(avg_chrominance[..., 1], shape=(-1))

    return Y_dithered, Cr_vector, Cb_vector
