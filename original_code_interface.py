from own_functions_without_config import embedding_DWT_watermark as embedding_DWT_watermark_fast
from own_functions_without_config import extraction_DWT_watermark as extraction_DWT_watermark_fast
from own_functions_without_config import Tamper_detection as Tamper_detection_fast
from own_functions_without_config import Tamper_localization as Tamper_localization_fast
from own_functions_without_config import recovery_process as recovery_process_fast
from own_functions_without_config import self_embedding as self_embedding_fast

from functions_without_config import embedding_DWT_watermark as embedding_DWT_watermark_slow
from functions_without_config import self_embedding as self_embedding_slow
from functions_without_config import extraction_DWT_watermark as extraction_DWT_watermark_slow
from functions_without_config import recovery_process as recover_process_slow

from help_functions import flip_binary_array
import numpy as np


def do_fast_embedding(input_image):
    img_size_x = input_image.shape[0]
    img_size_y = input_image.shape[1]
    bloc_size = 2
    wat_size_x = int(img_size_x/bloc_size/2)
    wat_size_y = int(img_size_y/bloc_size/2)
    kwargs = {
        "self_embed": True,
        "Auth_encryption": True,
        "Rec_scrambling": True,
        "auth_key": (1.5, 2.3),
        "scramble_key1": 5,
        "scramble_key2": 10,
        "BPP": 2,
        "bloc_size": bloc_size,
        "img_size_x": img_size_x,
        "img_size_y": img_size_y,
        "wat_size_x": wat_size_x,
        "wat_size_y": wat_size_y,
        "embedding_type": "DWT"
    }
    return embedding_DWT_watermark_fast(cover=input_image, org_watermark=None, **kwargs)
    


def do_slow_embedding(input_image):
    img_size_x = input_image.shape[0]
    img_size_y = input_image.shape[1]
    bloc_size = 2
    wat_size_x = int(img_size_x/bloc_size/2)
    wat_size_y = int(img_size_y/bloc_size/2)
    kwargs = {
        "self_embed": True,
        "Auth_encryption": True,
        "Rec_scrambling": True,
        "auth_key": (1.5, 2.3),
        "scramble_key1": 5,
        "scramble_key2": 10,
        "BPP": 2,
        "bloc_size": bloc_size,
        "img_size_x": img_size_x,
        "img_size_y": img_size_y,
        "wat_size_x": wat_size_x,
        "wat_size_y": wat_size_y,
        "embedding_type": "DWT"
    }

    return embedding_DWT_watermark_slow(cover=input_image, org_watermark=None, img_size_x=img_size_x, img_size_y=img_size_y, wat_size_x=wat_size_x, 
                                        wat_size_y=wat_size_y, key=kwargs["auth_key"])


def do_slow_recovery(input_image):
    img_size_x = input_image.shape[0]
    img_size_y = input_image.shape[1]
    bloc_size = 2
    wat_size_x = int(img_size_x/bloc_size/2)
    wat_size_y = int(img_size_y/bloc_size/2)
    kwargs = {
        "self_embed": True,
        "Auth_encryption": True,
        "Rec_scrambling": True,
        "auth_key": (1.5, 2.3),
        "scramble_key1": 5,
        "scramble_key2": 10,
        "BPP": 2,
        "bloc_size": bloc_size,
        "img_size_x": img_size_x,
        "img_size_y": img_size_y,
        "wat_size_x": int(img_size_x/bloc_size/2),
        "wat_size_y": int(img_size_y/bloc_size/2),
        "embedding_type": "DWT"
    }

    ext_watermark,Rec_watermark1,Rec_watermark2 = extraction_DWT_watermark_slow(input_image, wat_size_x=wat_size_x, wat_size_y=wat_size_y, key=kwargs["auth_key"])
    org_water=self_embedding_slow(input_image, embedding_type=kwargs["embedding_type"], wat_size_x=wat_size_x, wat_size_y=wat_size_y)
    tamper=Tamper_detection_fast(org_water, ext_watermark, **kwargs)

    return recover_process_slow(input_image, tamper, Rec_watermark1, Rec_watermark2, img_size_x=img_size_x, img_size_y=img_size_y, 
                                wat_size_x=wat_size_x, wat_size_y=wat_size_y)

def do_fast_tamper_detection(input_image):
    img_size_x = input_image.shape[0]
    img_size_y = input_image.shape[1]
    bloc_size = 2
    kwargs = {
        "self_embed": True,
        "Auth_encryption": True,
        "Rec_scrambling": True,
        "auth_key": (1.5, 2.3),
        "scramble_key1": 5,
        "scramble_key2": 10,
        "BPP": 2,
        "bloc_size": bloc_size,
        "img_size_x": img_size_x,
        "img_size_y": img_size_y,
        "wat_size_x": int(img_size_x/bloc_size/2),
        "wat_size_y": int(img_size_y/bloc_size/2),
        "embedding_type": "DWT"
    }

    ext_watermark,Rec_watermark1,Rec_watermark2 = extraction_DWT_watermark_fast(input_image, **kwargs)
    org_water=self_embedding_fast(input_image, **kwargs)
    
    tamper = Tamper_detection_fast(org_water, ext_watermark, **kwargs)
    tamper = Tamper_localization_fast(tamper=tamper, **kwargs)

    # Transform tamper mask to appropriate format:
    tamper_inverted = flip_binary_array(tamper)
    tamper_fs = np.zeros((img_size_x, img_size_y), dtype=np.uint8)

    for i in range(int(img_size_x/bloc_size/2)):
        for j in range(int(img_size_y/bloc_size/2)):
            x = i * bloc_size * 2
            y = j * bloc_size * 2
            tamper_fs[x:(x + bloc_size * 2), y:(y + bloc_size * 2)] = tamper_inverted[i, j]
    
    return tamper_fs, tamper, Rec_watermark1, Rec_watermark2, kwargs

def do_fast_recovery(input_image, tamper, r1, r2, kwargs):
    Rec_watermark1 = r1
    Rec_watermark2 = r2

    return recovery_process_fast(input_image, tamper, Rec_watermark1, Rec_watermark2, **kwargs)

def do_fast_recovery_old(input_image, include_tamper_mask=False):
    img_size_x = input_image.shape[0]
    img_size_y = input_image.shape[1]
    bloc_size = 2
    kwargs = {
        "self_embed": True,
        "Auth_encryption": True,
        "Rec_scrambling": True,
        "auth_key": (1.5, 2.3),
        "scramble_key1": 5,
        "scramble_key2": 10,
        "BPP": 2,
        "bloc_size": bloc_size,
        "img_size_x": img_size_x,
        "img_size_y": img_size_y,
        "wat_size_x": int(img_size_x/bloc_size/2),
        "wat_size_y": int(img_size_y/bloc_size/2),
        "embedding_type": "DWT"
    }

    ext_watermark,Rec_watermark1,Rec_watermark2 = extraction_DWT_watermark_fast(input_image, **kwargs)
    org_water=self_embedding_fast(input_image, **kwargs)
    
    tamper = Tamper_detection_fast(org_water, ext_watermark, **kwargs)
    tamper = Tamper_localization_fast(tamper=tamper, **kwargs)

    if include_tamper_mask:
        # Transform tamper mask to appropriate format:
        tamper_inverted = flip_binary_array(tamper)
        tamper_fs = np.zeros((img_size_x, img_size_y), dtype=np.uint8)

        for i in range(int(img_size_x/bloc_size/2)):
            for j in range(int(img_size_y/bloc_size/2)):
                x = i * bloc_size * 2
                y = j * bloc_size * 2
                tamper_fs[x:(x + bloc_size * 2), y:(y + bloc_size * 2)] = tamper_inverted[i, j]
        
        return recovery_process_fast(input_image, tamper, Rec_watermark1, Rec_watermark2, **kwargs), tamper_fs

    return recovery_process_fast(input_image, tamper, Rec_watermark1, Rec_watermark2, **kwargs), tamper