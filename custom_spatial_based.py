import numpy as np
from help_functions import grow_tamper_pixels_single_channel, fix_voids_using_interpolation, reduce_img_size, maxpool2d, do_inpainting


def fix_hi_using_lo(size_x_hi, size_y_hi, size_x_lo, void_hi, void_lo, rec_hi, rec_lo):
    hi_lo_scale = size_x_hi / size_x_lo
    for x_hi in range(size_x_hi):
        for y_hi in range(size_y_hi):
            if void_hi[x_hi, y_hi] == 1:
                x_lo = int(x_hi // hi_lo_scale)
                y_lo = int(y_hi // hi_lo_scale)

                # Only replace values if the low resolution watermark is not void at this location 
                if void_lo[x_lo, y_lo] == 0:
                    rec_hi[x_hi, y_hi] = rec_lo[x_lo, y_lo]
                    void_hi[x_hi, y_hi] = 0
    rec_watermark = rec_hi
    void_watermark = void_hi

    return rec_watermark, void_watermark


def vectorized_random_auth_data_locations(rng, size_x, size_y):
    """
    Vectorized function to randomize authentication data locations
    """

    # We know that the number of length and width pixels will always be even,
    # so we can take the max values of 2x2 areas without having to worry about padding
    a = rng.random(size=(size_x * size_y // 4, 4))
    
    # Set max values to one
    # Use [:,None] to turn the 1D max array into a 2D array,
    # which is needed for appropriate boolean matching
    b = (a == a.max(axis=1)[:,None]).astype(int)

    # Reshape into desired shape:
    auth_loc_mask = np.empty((size_x, size_y), dtype=int)
    auth_loc_mask[0::2, :] = b[:, :2].reshape(-1, size_y)
    auth_loc_mask[1::2, :] = b[:, 2:].reshape(-1, size_y)

    data_loc_mask = auth_loc_mask * (-1) + 1

    return auth_loc_mask.astype(bool), data_loc_mask.astype(bool)



def __void_index_to_data_indices(auth_loc, auth_array, voids_array, base_x, base_y):
    # Number of block rows (= number of rows containing blocks)
    nbr = base_x // 2
    # Number of block columns (= number of blocks in a row)
    nbc = base_y // 2

    # We work on a row-by-row basis:
    offset = 0
    for block_row in range(nbr):
        auth_i_start = block_row * nbc

        blocks = auth_loc[block_row*2: block_row*2 + 2]

        # First, for each block, we determine the corresponding void array indices:
        void_indices = [[] for _ in range(nbc)]
        
        for block_x in range(2):
            for block_y in range(nbc):
                for y_offset in range(2):
                    
                    if blocks[block_x, 2*block_y + y_offset] == False:
                        void_indices[block_y].append(offset)
                        offset += 1

        # Then, we can iterate over the authentication values to determine which blocks are tampered with
        # For these blocks we can then look up the corresponding void indices

        # For every block, we check if it has been tampered with
        for auth_i in range(auth_i_start, auth_i_start + nbc):
            if auth_array[auth_i] == 1:
                # Tampering has been detected
                # Determine which block this is
                block_idx = auth_i - auth_i_start
                voids_array[void_indices[block_idx]] = 1


def embed_watermark_no_vectorization(input_img, seed):
    """
    "watermark_high_low_quality_secure_v2" without vectorization (e.g. using for-loops)
    """
    # Generate recovery watermark
    base_x, base_y = input_img.shape[:2]
    n_channels = 3 if len(input_img.shape) == 3 else 1
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4
    size_x_hi=base_x // 2
    size_y_hi=base_y // 2
    # Low resolution and high resolution recovery watermarks
    rec_w_lo = reduce_img_size(input_image=input_img, new_size_x=size_x_lo, new_size_y=size_y_lo)
    rec_w_hi = reduce_img_size(input_image=input_img, new_size_x=size_x_hi, new_size_y=size_y_hi)

    # Generate authentication watermark
    auth_w_size_x = base_x // 2
    auth_w_size_y = base_y // 2
    rng = np.random.default_rng(seed=seed)

    auth_w = rng.integers(low=0, high=4, size=(auth_w_size_x * auth_w_size_y, n_channels), dtype=np.uint8)
    auth_locations = rng.integers(low=0, high=4, size=(base_x * base_y // 4))

    # Put all reconstruction watermarks in one big list
    base_array = np.zeros((base_x * base_y, n_channels), np.uint8).reshape((-1, n_channels))
    
    # High resolution subarrays
    hi_numel = size_x_hi * size_y_hi
    rec_w_hi = rec_w_hi.reshape((hi_numel, n_channels))
    hi0 = (rec_w_hi >> 6) & 3
    hi1 = (rec_w_hi >> 4) & 3
    hi_arr = np.concat([hi0, hi1])
    
    # Low resolution subarrays
    lo_numel = size_x_lo * size_y_lo
    rec_w_lo = rec_w_lo.reshape((lo_numel, n_channels))
    lo0 = (rec_w_lo >> 6) & 3
    lo1 = (rec_w_lo >> 4) & 3
    lo_arr = np.concat([lo0, lo1, lo0, lo1])

    # Concatenate all data
    data_array = np.concat([hi_arr, lo_arr])
    
    # Shuffle the array
    shuffle_indices = np.arange(2* hi_numel + 4*lo_numel)
    rng.shuffle(shuffle_indices)

    data_array = data_array[shuffle_indices]

    # Repeat each element once

    # Now we need to interlace the reconstruction data with authentication data
    # Expected result:
    # [a r a r a r]
    # [r r r r r r]
    # [a r a r a r]
    # [r r r r r r]
    # [a r a r a r]
    # [r r r r r r]
    # Where a=authentication pixel, and r=reconstruction pixel
    # For each group of 4 pixels, a indicates whether their data is reliable
    # Note: Position of a in 2x2 square is random
    
    base_array = np.empty((base_x, base_y, n_channels), dtype=np.uint8)

    a_idx = 0
    data_idx = 0
    for row in range(0, base_x, 2):
        for col in range(0, base_y, 2):
            a_loc = auth_locations[a_idx]
            a_val = auth_w[a_idx]
            # a locations based on a_val
            # [0 1]
            # [2 3]
            if a_loc == 0:
                base_array[row, col] = a_val
                base_array[row, col + 1] = data_array[data_idx]
                base_array[row+1, col:col+2] = data_array[data_idx+1:data_idx+3]
            elif a_loc == 1:
                base_array[row, col] = data_array[data_idx]
                base_array[row, col + 1] = a_val
                base_array[row+1, col:col+2] = data_array[data_idx+1:data_idx+3]
            elif a_loc == 2:
                base_array[row, col:col+2] = data_array[data_idx:data_idx+2]
                base_array[row+1, col] = a_val
                base_array[row+1, col+1] = data_array[data_idx+2]
            elif a_loc == 3:
                base_array[row, col:col+2] = data_array[data_idx:data_idx+2]
                base_array[row+1, col] = data_array[data_idx+2]
                base_array[row+1, col+1] = a_val
            a_idx +=1 
            data_idx += 3

    return (input_img & 0b11111100) | base_array


def tamper_detection_no_vectorization(input_img, seed, include_tamper_mask=False, sharpen=True):
    """
    Tamper detection function that goes alongside the unvectorized embedding function
    """
    
    base_x, base_y = input_img.shape[:2]
    auth_w_size_x = size_x_hi = base_x // 2
    auth_w_size_y = size_y_hi = base_y // 2
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4

    # Generate authentication watermark
    auth_w_size_x = base_x // 2
    auth_w_size_y = base_y // 2
    rng = np.random.default_rng(seed=seed)
    expected_auth_w = rng.integers(low=0, high=4, size=(auth_w_size_x * auth_w_size_y, 3), dtype=np.uint8)
    auth_locations = rng.integers(low=0, high=4, size=(base_x * base_y // 4))
    received_auth_w = np.empty((auth_w_size_x * auth_w_size_y, 3), dtype=np.uint8)
    # Extract the watermark data
    watermarks = input_img & 3
    
    hi_numel = size_x_hi * size_y_hi
    lo_numel = size_x_lo * size_y_lo
    received_data_array = np.zeros((2*hi_numel + 4*lo_numel, 3), np.uint8)

    a_idx = 0
    data_idx = 0
    for row in range(0, base_x, 2):
        for col in range(0, base_y, 2):
            a_loc = auth_locations[a_idx]
            
            if a_loc == 0:
                received_auth_w[a_idx] = watermarks[row, col]
                received_data_array[data_idx] = watermarks[row, col+1]
                received_data_array[data_idx+1:data_idx+3] = watermarks[row+1, col:col+2]
            elif a_loc == 1:
                received_auth_w[a_idx] = watermarks[row, col+1]
                received_data_array[data_idx] = watermarks[row, col]
                received_data_array[data_idx+1:data_idx+3] = watermarks[row+1, col:col+2]
            
            elif a_loc == 1:
                received_auth_w[a_idx] = watermarks[row+1, col]
                received_data_array[data_idx:data_idx+2] = watermarks[row, col:col+2]
                received_data_array[data_idx+3] = watermarks[row+1, col]
            
            elif a_loc == 1:
                received_auth_w[a_idx] = watermarks[row+1, col+1]
                received_data_array[data_idx:data_idx+2] = watermarks[row, col:col+2]
                received_data_array[data_idx+3] = watermarks[row+1, col+1]
            
            a_idx += 1
            data_idx += 3
    
    # Compare received and expected authentication watermarks
    auth_diff = np.clip(np.abs((expected_auth_w.astype(int) - received_auth_w.astype(int))), a_min=0, a_max=1).astype(np.uint8)
    # Normalize the three channels into one channel:
    auth_diff = np.max(auth_diff, axis=1)
    print("Number of tampered authentication pixels:", np.sum(auth_diff))

    # Reshape auth_diff to an image, and identify the affected areas by using the grow function
    auth_diff = auth_diff.reshape((auth_w_size_x, auth_w_size_y))
    tamper_mask = grow_tamper_pixels_single_channel(auth_diff, step=1)
    
    # display_images([tamper_mask, auth_diff], "Tamper localization")
    auth_diff = tamper_mask.reshape(-1)

    tamper_mask = grow_tamper_pixels_single_channel(auth_diff, step=1)

    # Get full-scale tamper mask:
    tamper_mask_fs = np.zeros((base_x, base_y), dtype=np.uint8)
    for block_x in range(auth_w_size_x):
        for block_y in range(auth_w_size_y):
            x = block_x * 2
            y = block_y * 2
            tamper_mask_fs[x:x+2,y:y+2] = tamper_mask[block_x, block_y]
    
    return tamper_mask_fs, tamper_mask, None, None, rng, received_data_array


def embed_watermark(input_img, seed):
    """
    Watermark embedding
    """    
    # Generate recovery watermark
    base_x, base_y = input_img.shape[:2]
    # n_channels = 3 if len(input_img.shape) == 3 else 1
    n_channels = input_img.shape[2]
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4
    size_x_hi=base_x // 2
    size_y_hi=base_y // 2
    # Low resolution and high resolution recovery watermarks
    rec_w_lo = reduce_img_size(input_image=input_img, new_size_x=size_x_lo, new_size_y=size_y_lo)
    rec_w_hi = reduce_img_size(input_image=input_img, new_size_x=size_x_hi, new_size_y=size_y_hi)
    # display_images([input_img, rec_w_hi, rec_w_lo], "Original, High quality, Low quality")

    # Generate authentication watermark
    auth_w_size_x = base_x // 2
    auth_w_size_y = base_y // 2
    rng = np.random.default_rng(seed=seed)

    auth_w = rng.integers(low=0, high=4, size=(auth_w_size_x * auth_w_size_y, n_channels), dtype=int)

    auth_loc, data_loc = vectorized_random_auth_data_locations(rng=rng, size_x=base_x, size_y=base_y)
    
    # Put all reconstruction watermarks in one big list
    # High resolution subarrays
    hi_numel = size_x_hi * size_y_hi
    rec_w_hi = rec_w_hi.reshape((hi_numel, n_channels))
    hi0 = (rec_w_hi >> 6) & 3
    hi1 = (rec_w_hi >> 4) & 3
    hi_arr = np.concat([hi0, hi1])
    
    # Low resolution subarrays
    lo_numel = size_x_lo * size_y_lo
    rec_w_lo = rec_w_lo.reshape((lo_numel, n_channels))
    lo0 = (rec_w_lo >> 6) & 3
    lo1 = (rec_w_lo >> 4) & 3
    lo_arr = np.concat([lo0, lo1, lo0, lo1])

    # Concatenate all data
    data_array = np.concat([hi_arr, lo_arr])
    
    # Shuffle the array
    shuffle_indices = np.arange(2* hi_numel + 4*lo_numel)
    rng.shuffle(shuffle_indices)

    data_array = data_array[shuffle_indices]
    
    base_array = np.empty((base_x, base_y, n_channels), dtype=np.uint8)

    base_array[auth_loc] = auth_w
    base_array[data_loc] = data_array
    # print(input_img.shape, base_array.shape)
    
    return (input_img & 0b11111100) | base_array


def tamper_detection(input_img, seed):
    base_x, base_y = input_img.shape[:2]
    auth_w_size_x = size_x_hi = base_x // 2
    auth_w_size_y = size_y_hi = base_y // 2
    auth_vals = auth_w_size_x * auth_w_size_y
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4
    n_channels = input_img.shape[2]

    # Generate authentication watermark
    auth_w_size_x = base_x // 2
    auth_w_size_y = base_y // 2
    rng = np.random.default_rng(seed=seed)
    expected_auth_w = rng.integers(low=0, high=4, size=(auth_vals, n_channels), dtype=int)

    auth_loc, data_loc = vectorized_random_auth_data_locations(rng=rng, size_x=base_x, size_y=base_y)

    received_auth_w = np.zeros((auth_vals, n_channels), dtype=int)
    
    # Extract the watermark data
    watermarks = input_img & 3
    
    hi_numel = size_x_hi * size_y_hi
    lo_numel = size_x_lo * size_y_lo
    received_data_array = np.zeros((2*hi_numel + 4*lo_numel, n_channels), np.uint8)

    received_auth_w = watermarks[auth_loc]
    received_data_array = watermarks[data_loc]

    # Also arrange order:
    # The problem we are trying to solve is this:
    # While we work in 2x2 blocks, the indexer in cover[auth_loc] does not know that
    # As such, the placement of the authentication pixels can be unpredictably skewed
    # For example, take the auth values [1, 2, 3, 4]
    # And the corresponding auth_loc:
    # [ 0 0 0 1 0 0 0 1 ]
    # [ 0 1 0 0 1 0 0 0 ]
    # We would want our auth values to be:
    # [ 0 0 0 2 0 0 0 4]
    # [ 0 1 0 0 3 0 0 0]
    # However, due to the nature of indexing in numpy, we instead get
    # [ 0 0 0 1 0 0 0 2]
    # [ 0 3 0 0 4 0 0 0]
    # As previously stated, this behaviour depends on the random initialization of the auth_loc, 
    # and is as such unpredictable. Therefor, to counteract its effects, we perform a similar 
    # transformation on an "order" matrix that allows for correcting the extracted authentication values
    # The order_input will provide the values that can serve as indices for transformation later on 
    order_input = np.arange(auth_vals)
    # We will use a max pooling operation, so all other values should be zero
    order = np.zeros(shape=(base_x, base_y), dtype=int)
    # Place the inputs according to the random locations
    order[auth_loc] = order_input
    # Isolate 2x2 blocks and take their maximum = maxpooling
    order = maxpool2d(order, size_x=base_x, size_y=base_y, k=2).reshape(-1)

    # Now correct received authentication values with the order transformation
    # We need to transform both the actually embedded values, as well as expected values,
    # to ensure they remain on the same domain

    # Compare received and expected authentication watermarks
    auth_diff = np.clip(np.abs((expected_auth_w.astype(int) - received_auth_w.astype(int))), a_min=0, a_max=1).astype(np.uint8)

    # Normalize the three channels into one channel:
    auth_diff = np.max(auth_diff, axis=1)
    
    # Transform auth_diff to account for the order
    # We do this here instead of at received_auth_w and expected_auth_w because here we only need to transform once instead of twice
    auth_diff = auth_diff[order]


    # Reshape auth_diff to an image, and identify the affected areas by using the grow function
    auth_diff = auth_diff.reshape((auth_w_size_x, auth_w_size_y))
    tamper_mask = grow_tamper_pixels_single_channel(auth_diff, step=1)

    # Get full-scale tamper mask:
    tamper_mask_fs = np.zeros((base_x, base_y), dtype=np.uint8)
    for block_x in range(auth_w_size_x):
        for block_y in range(auth_w_size_y):
            x = block_x * 2
            y = block_y * 2
            tamper_mask_fs[x:x+2,y:y+2] = tamper_mask[block_x, block_y]
    
    return tamper_mask_fs, tamper_mask, auth_loc, data_loc, rng, received_data_array


def __tamper_detection_for_test(input_img, seed):
    base_x, base_y = input_img.shape[:2]
    auth_w_size_x = size_x_hi = base_x // 2
    auth_w_size_y = size_y_hi = base_y // 2
    auth_vals = auth_w_size_x * auth_w_size_y
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4
    n_channels = input_img.shape[2]

    # Generate authentication watermark
    auth_w_size_x = base_x // 2
    auth_w_size_y = base_y // 2
    rng = np.random.default_rng(seed=seed)
    expected_auth_w = rng.integers(low=0, high=4, size=(auth_vals, n_channels), dtype=int)

    auth_loc, data_loc = vectorized_random_auth_data_locations(rng=rng, size_x=base_x, size_y=base_y)

    received_auth_w = np.zeros((auth_vals, n_channels), dtype=int)
    
    # Extract the watermark data
    watermarks = input_img & 3
    
    hi_numel = size_x_hi * size_y_hi
    lo_numel = size_x_lo * size_y_lo
    received_data_array = np.zeros((2*hi_numel + 4*lo_numel, n_channels), np.uint8)

    received_auth_w = watermarks[auth_loc]
    received_data_array = watermarks[data_loc]

    # Also arrange order:
    # The problem we are trying to solve is this:
    # While we work in 2x2 blocks, the indexer in cover[auth_loc] does not know that
    # As such, the placement of the authentication pixels can be unpredictably skewed
    # For example, take the auth values [1, 2, 3, 4]
    # And the corresponding auth_loc:
    # [ 0 0 0 1 0 0 0 1 ]
    # [ 0 1 0 0 1 0 0 0 ]
    # We would want our auth values to be:
    # [ 0 0 0 2 0 0 0 4]
    # [ 0 1 0 0 3 0 0 0]
    # However, due to the nature of indexing in numpy, we instead get
    # [ 0 0 0 1 0 0 0 2]
    # [ 0 3 0 0 4 0 0 0]
    # As previously stated, this behaviour depends on the random initialization of the auth_loc, 
    # and is as such unpredictable. Therefor, to counteract its effects, we perform a similar 
    # transformation on an "order" matrix that allows for correcting the extracted authentication values
    # The order_input will provide the values that can serve as indices for transformation later on 
    order_input = np.arange(auth_vals)
    # We will use a max pooling operation, so all other values should be zero
    order = np.zeros(shape=(base_x, base_y), dtype=int)
    # Place the inputs according to the random locations
    order[auth_loc] = order_input
    # Isolate 2x2 blocks and take their maximum = maxpooling
    order = maxpool2d(order, size_x=base_x, size_y=base_y, k=2).reshape(-1)

    # Now correct received authentication values with the order transformation
    # We need to transform both the actually embedded values, as well as expected values,
    # to ensure they remain on the same domain

    # Compare received and expected authentication watermarks
    auth_diff = np.clip(np.abs((expected_auth_w.astype(int) - received_auth_w.astype(int))), a_min=0, a_max=1).astype(np.uint8)

    # Normalize the three channels into one channel:
    auth_diff = np.max(auth_diff, axis=1)

    # Transform auth_diff to account for the order
    # We do this here instead of at received_auth_w and expected_auth_w because here we only need to transform once instead of twice
    auth_diff = auth_diff[order]


    # Reshape auth_diff to an image, and identify the affected areas by using the grow function
    auth_diff = auth_diff.reshape((auth_w_size_x, auth_w_size_y))
    tamper_mask = grow_tamper_pixels_single_channel(auth_diff, step=1)

    # Get full-scale tamper mask:
    tamper_mask_fs = np.zeros((base_x, base_y), dtype=np.uint8)
    auth_diff_fs = np.zeros((base_x, base_y), dtype=np.uint8)
    for block_x in range(auth_w_size_x):
        for block_y in range(auth_w_size_y):
            x = block_x * 2
            y = block_y * 2
            tamper_mask_fs[x:x+2,y:y+2] = tamper_mask[block_x, block_y]
            auth_diff_fs[x:x+2,y:y+2] = auth_diff[block_x, block_y]
    
    return tamper_mask_fs, auth_diff_fs


def recovery(input_img, tamper_mask, auth_loc, data_loc, rng, received_data_array, sharpen=True):
    base_x, base_y = input_img.shape[:2]
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4
    size_x_hi=base_x // 2
    size_y_hi=base_y // 2
    size_x_lo=base_x // 4
    size_y_lo=base_y // 4

    hi_numel = size_x_hi * size_y_hi
    lo_numel = size_x_lo * size_y_lo

    auth_diff = tamper_mask.reshape(-1)
    print("number of tampered pixels", np.sum(tamper_mask))
    # Use the tampered authentication pixels to extract void information
    # This void information will tell us which reconstruction pixels are invalid
    void_array = np.zeros((2*hi_numel + 4*lo_numel), np.uint8)

    __void_index_to_data_indices(auth_loc=auth_loc, auth_array=auth_diff, voids_array=void_array,
                                 base_x=base_x, base_y=base_y)
    
    # Visualize void marking
    void_map = np.zeros((base_x, base_y), dtype=np.uint8)
    void_map[data_loc] = void_array

    # Undo the shuffling
    shuffle_indices = np.arange(2* hi_numel + 4*lo_numel)
    rng.shuffle(shuffle_indices)

    received_data_array[shuffle_indices] = received_data_array
    void_array[shuffle_indices] = void_array

    # Extract original watermarks
    hi_watermark_bits = received_data_array[:2*hi_numel]
    lo_watermark_bits = received_data_array[2*hi_numel:]
    
    hi_voids = void_array[:2*hi_numel]
    lo_voids = void_array[2*hi_numel:]

    # Reconstruct high resolution watermark
    #  Extract MSB and LSB portions
    hi_msb = (hi_watermark_bits[:hi_numel] & 3) << 6
    hi_lsb = (hi_watermark_bits[hi_numel:] & 3) << 4
    hi_msb_voids = hi_voids[hi_numel:]
    hi_lsb_voids = hi_voids[:hi_numel]
    #  Recombine MSB and LSB
    rec_w_hi = (hi_msb | hi_lsb)
    rec_w_hi = rec_w_hi.reshape((size_x_hi, size_y_hi, 3))
    hi_voids = hi_msb_voids | hi_lsb_voids
    hi_voids = hi_voids.reshape((size_x_hi, size_y_hi))

    # Reconstruct low resolution watermark
    lo0_msb = lo_watermark_bits[0*lo_numel:1*lo_numel]
    lo0_lsb = lo_watermark_bits[1*lo_numel:2*lo_numel]
    lo1_msb = lo_watermark_bits[2*lo_numel:3*lo_numel]
    lo1_lsb = lo_watermark_bits[3*lo_numel:4*lo_numel]

    lo0_msb_voids = lo_voids[0*lo_numel:1*lo_numel]
    lo0_lsb_voids = lo_voids[1*lo_numel:2*lo_numel]
    lo1_msb_voids = lo_voids[2*lo_numel:3*lo_numel]
    lo1_lsb_voids = lo_voids[3*lo_numel:4*lo_numel]

    lo0 = (lo0_msb << 6) | (lo0_lsb << 4)
    lo1 = (lo1_msb << 6) | (lo1_lsb << 4)
    lo0_voids = lo0_msb_voids | lo0_lsb_voids
    lo1_voids = lo1_msb_voids | lo1_lsb_voids

    lo0 = lo0.reshape((size_x_lo, size_y_lo, 3))
    lo1 = lo1.reshape((size_x_lo, size_y_lo, 3))
    lo0_voids = lo0_voids.reshape((size_x_lo, size_y_lo))
    lo1_voids = lo1_voids.reshape((size_x_lo, size_y_lo))

    # Recombine low resolution watermarks to exclude voids
    rec_w_lo = np.zeros((size_x_lo, size_y_lo, 3), dtype=np.uint8)
    lo_voids = np.zeros((size_x_lo, size_y_lo), dtype=np.uint8)
    for x in range(size_x_lo):
        for y in range(size_y_lo):
            if not lo0_voids[x,y]:
                rec_w_lo[x,y] = lo0[x,y]
                lo_voids[x,y] = 0
            elif not lo1_voids[x,y]:
                rec_w_lo[x,y] = lo1[x,y]
                lo_voids[x,y] = 0
            else:
                lo_voids[x,y] = 1

    rec_watermark, void_watermark = fix_hi_using_lo(size_x_hi=size_x_hi, size_y_hi=size_y_hi, size_x_lo=size_x_lo, 
                                                    void_hi=hi_voids, void_lo=lo_voids, rec_hi=rec_w_hi, rec_lo=rec_w_lo)

    # Fix the final holes using interpolation
    rec_watermark, void_watermark = fix_voids_using_interpolation(void_watermark=void_watermark, rec_watermark=rec_watermark, size_x=size_x_hi, size_y=size_y_hi)
    
    # After fixing the reconstruction watermark, we can finally proceed with inpainting
    tamper_mask_3d = np.repeat(tamper_mask[:, :, np.newaxis], 3, axis=2)

    inpainted, fs_tamper_mask = do_inpainting(attacked_image=input_img, tamper_mask=tamper_mask_3d, recovery_watermark=rec_watermark, do_sharpening=sharpen)

    return inpainted

