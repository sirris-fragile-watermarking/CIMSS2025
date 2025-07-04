import cv2
import numpy as np 
import imageio.v2 as imageio
import os


current_directory = os.path.dirname(__file__)
img_path = os.path.join(current_directory, 'Images/Saliboat.tiff')
wat_path = os.path.join(current_directory, 'Images/Peppers.tiff')

img_size=512
bloc_size=2                   #Choose bloc_size=2, BPP=2 OR bloc_size=3, BPP=1
BPP=2                         #Embedding capacity

Color_image=True              #False mean that images will be converted to grayscale
self_embed=True               #Meaning that the watermark is generated from the cover image
Auth_encryption=True          #Encrypt the authentication watermark using Henon map
key = (1.5, 2.3)
Rec_scrambling=True           #Scramble the recovery watermarks using Arnold transformation. This is mendatory to perform recovery
key1=5
key2=10
embedding_type="DWT"          #The method used during the authentication watermark generation

wat_size=int(img_size/bloc_size/2)
# Load the cover image
img = imageio.imread(img_path)
if np.max(img)>255 or np.min(img)<0:
    raise ValueError("Please use 8-bit images with pixel values ranging between 0 and 255")
"""print(img[0][0:3])"""

original_img=cv2.resize(img,(img_size,img_size))
if Color_image==False and len((np.asarray(original_img)).shape)==3:
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)


# Load the watermark
watermark=imageio.imread(wat_path)
if np.max(watermark)>255 or np.min(watermark)<0:
    raise ValueError("Please use 8-bit images with pixel values ranging between 0 and 255")
if len((np.asarray(original_img)).shape)!=3:
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
org_watermark=cv2.resize(watermark,(wat_size,wat_size)) 
