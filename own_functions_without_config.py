"""
Majorly modified version of functions.py in https://github.com/Riadh-Bouarroudj/Fragile-image-watermarking-with-recovery/tree/main?tab=readme-ov-file
Particularly, all binary operations have changed from string operations to binary arithmatic, resulting in major speedups


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [2024] [Riadh Bouarroudj]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


"""

import numpy as np
import pywt

#Arnold transform
def scramble_image(image, iterations):
    #image=np.array(image)
    n, m = image.shape
    transformed_image = np.zeros((n, m),dtype="uint8")
    for x in range(n):
        for y in range(m):
            transformed_image[(2*x + y) % n][(x + y) % m] = image[x][y]
    if iterations > 1:
        return scramble_image(transformed_image, iterations - 1)
    else:
        return transformed_image

def unscramble_image(image, iterations):
    #image=np.array(image)
    n, m = image.shape
    transformed_image = np.zeros((n, m),dtype="uint8")
    for x in range(n):
        for y in range(m):
            transformed_image[x][y]= image[(2*x + y) % n][(x + y) % m]
    if iterations > 1:
        return unscramble_image(transformed_image, iterations - 1)
    else:
        return transformed_image

#Henon map
def encrypt_image(image, key):
    encrypted_image = np.copy(image)
    height, width = image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Encrypt the pixel value using XOR operation
            xor = np.astype(np.asarray(x * 255), np.uint8)
            encrypted_image[i, j] = image[i, j] ^ xor
    
    return encrypted_image

def decrypt_image(encrypted_image, key):
    decrypted_image = np.copy(encrypted_image)
    height, width = encrypted_image.shape
    
    # Initialize the Hanon map with the given key
    x, y = key[0], key[1]
    
    for i in range(height):
        for j in range(width):
            # Generate pseudo-random numbers using the Hanon map
            x = 1 - key[0] * x**2 +  y
            y = key[1] *x
            
            # Decrypt the pixel value using XOR operation
            xor = np.astype(np.asarray(x * 255), np.uint8)
            decrypted_image[i, j] = encrypted_image[i, j] ^ xor
    
    return decrypted_image


def dec_to_bin(n):
    x=bin(n).replace("0b", "")
    while len(x)<8 :
      x="0"+x 
    return (x)
    
def bin_to_dec(n):
    return int(n, 2)


def watermark_to_digit(wat):
   #Transform wat to 1D array
   wat = np.array(wat)
   wat = wat.flatten()
   return wat


def embedding_DWT_watermark(cover, org_watermark, **kwargs):
   self_embed=kwargs.get("self_embed", True)
   Auth_encryption=kwargs.get("Auth_encryption", True)
   Rec_scrambling=kwargs.get("Rec_scrambling", True)
   key = kwargs.get("auth_key", None)
   key1 = kwargs.get("scramble_key1", None)
   key2 = kwargs.get("scramble_key2", None)
   BPP = kwargs.get("BPP", 2)
   bloc_size = kwargs.get("bloc_size", 2)

   if len((np.asarray(cover)).shape) == 3:
      long = 3
   else:
      long = 1

   # Normalize the cover image to the range of [4,251] to avoid overfloaw and underflow problems
   cover = np.clip(cover, a_min=4, a_max=251)

   # Load the watermarks
   if self_embed == True:
      Auth_wat = self_embedding(cover, **kwargs)
   else:
      Auth_wat = org_watermark

   Rec_wat = Recovery_watermark_construction(cover, long, **kwargs)
   Auth_arr = []
   Rec_arr1 = []
   Rec_arr2 = []
   # Prepare the watermarks and transform them to long binary digits
   for channel in range(long):
      if long == 3:
         Auth = Auth_wat[:, :, channel]
         Rec = Rec_wat[:, :, channel]
      else:
         Auth = Auth_wat
         Rec = Rec_wat
      if Auth_encryption == True:
         Auth = encrypt_image(Auth, key)
      if Rec_scrambling == True:
         Rec1 = scramble_image(Rec, key1)
         Rec2 = scramble_image(Rec, key2)
      else:
         Rec1 = Rec
         Rec2 = Rec
      Auth = watermark_to_digit(Auth)
      Rec1 = watermark_to_digit(Rec1)
      Rec2 = watermark_to_digit(Rec2)
      Auth_arr.append(Auth)
      Rec_arr1.append(Rec1)
      Rec_arr2.append(Rec2)

   w_comp_arr = []
   # Loop on the RGB channels of the cover image
   for channel in range(long):
      if long == 3:
         watermarked_img = cover[:, :, channel]
      else:
         watermarked_img = np.copy(cover)
      Auth_watermark = Auth_arr[channel]
      Rec_watermark1 = Rec_arr1[channel]
      Rec_watermark2 = Rec_arr2[channel]

      # Apply Discrete wavelet transform to the cover image
      # print("Voor dwt2 berekening")
      coeffs = pywt.dwt2(watermarked_img, "haar")
      LL, (LH, HL, HH) = coeffs
      lis = ["LH", "HL", "HH"]

      # Loop on the frequency subbands of the image
      for subb in lis:
         if subb == "LH":
               subband = LH
               watermark = Rec_watermark1
         elif subb == "HL":
               subband = HL
               watermark = Rec_watermark2
         elif subb == "HH":
               subband = HH
               watermark = Auth_watermark
         a = 0
         # Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT
         subband = np.round(subband, 5)
         # Loop on each subband
         wat_Byte = 0
         wat_bit = 0
         BPP_mask = 0b11000000 if BPP == 2 else 0b10000000
         BPP_pixel_mask = 0b11111100 if BPP == 2 else 0b11111110
         QU_mask = 0b00000100 if BPP == 2 else 0b00000010
         while a + bloc_size <= subband.shape[0]:
            b = 0
            while b + bloc_size <= subband.shape[1]:
               v = 0
               # t1 = time()
               # print("Subband {},{}".format(a, b), end="\r")
               # Loop on each bloc
               for j in range(bloc_size):
                  for k in range(bloc_size):
                        if v * BPP == 8:
                           # Ensure that 8 bits are embedded in each block to provide tamper localization
                           break
                        neg = False
                        # Given that the coefficients values can be negative, transform them to positive values for binary transformation
                        if subband[a + j][b + k] < 0:
                           subband[a + j][b + k] = (subband[a + j][b + k] * -1)
                           neg = True
                        dec_part = subband[a + j][b + k] % 1
                        int_part = int(subband[a + j][b + k])
                        # pixel = dec_to_bin(int_part)
                        pixel_b = int_part

                        # Watermark bits embedding
                        # bits = str(watermark[1][0:BPP])
                        
                        bits_b = ((watermark[wat_Byte] << wat_bit) & BPP_mask) >> (8 - BPP)

                        if wat_bit == 6:
                           wat_bit = 0
                           wat_Byte += 1
                        else:
                           wat_bit += BPP
                        
                        # if int(bits, 2) != bits_b:
                        #    raise ValueError()

                        # watermark = watermark[0], watermark[1][BPP : len(watermark[1])]
                        # pixel = pixel[0 : len(pixel) - BPP] + bits
                        pixel_b = int((pixel_b & BPP_pixel_mask) | bits_b)
                        # print(pixel, pixel_b)

                        # Pixel quality adjustement fo better watermarked image quality
                        if BPP > 1:

                           qu_b = (pixel_b & QU_mask) >> BPP
                           # qu_b = (~qu_b) & 1 # Complement turns the entire thing wrong otherwise
                           qu_b = (0b11111111 - qu_b) & 1
                           # print(type(qu_b), type(pixel_b), ~QU_mask, type(~QU_mask), type(QU_mask), type(np.astype(~QU_mask, np.uint8)))
                           pixel_qu_b = (pixel_b & ~QU_mask) | (qu_b << BPP)
                           # print("A: {:08b} <-> {:08b}, QU = {} -> {}".format(pixel_b, pixel_qu_b, (pixel_b & QU_mask) >> (BPP), qu_b))
                           if abs(int_part - pixel_qu_b) < abs(int_part - pixel_b):
                              pixel_b = pixel_qu_b

                        else:
                           pixel_b = bin_to_dec(pixel_b)

                        # Update the watermarked coefficient
                        subband[a + j][b + k] = pixel_b + dec_part
                        if neg == True:
                           subband[a + j][b + k] = (subband[a + j][b + k] * -1)
                        v = v + 1
               b = b + bloc_size
               # t2 = time()
               # times.append(t2 - t1)
            a = a + bloc_size
         # print("Average t-step cost = {}".format(np.round(np.mean(times),decimals=6)))
         # print("Na bits berekening")
         if subb == "LL":
               LL = subband
         elif subb == "LH":
               LH = subband
         elif subb == "HL":
               HL = subband
         elif subb == "HH":
               HH = subband
      # Apply inverse DWT to the watermarked subbands
      watermarked_coeffs = LL, (LH, HL, HH)
      watermarked = pywt.idwt2(watermarked_coeffs, "haar")

      # print("Voor conversie naar integers")
      # Convert the watermarked channel to integer values
      x_size, y_size = watermarked.shape
      # print(watermarked_img.shape, watermarked.shape)
      for i in range(x_size):
         for j in range(y_size):
               p = watermarked[i][j] % 1
               if p > 0.6:
                  watermarked_img[i][j] = int(watermarked[i][j]) + 1
               else:
                  watermarked_img[i][j] = int(watermarked[i][j])
      w_comp_arr.append(watermarked_img)
      # print(np.min(watermarked_img),np.max(watermarked_img))

   if long == 3:
      watermarked_img = np.stack([w_comp_arr[0], w_comp_arr[1], w_comp_arr[2]], axis=2)
   else:
      watermarked_img = w_comp_arr[0]

   return watermarked_img

def extraction_DWT_watermark(imagex, **kwargs):
    # self_embed=kwargs.get("self_embed", True)
    Auth_encryption=kwargs.get("Auth_encryption", True)
    Rec_scrambling=kwargs.get("Rec_scrambling", True)
    key = kwargs.get("auth_key", None)
    key1 = kwargs.get("scramble_key1", None)
    key2 = kwargs.get("scramble_key2", None)
    img_size_x = kwargs.get("img_size_x", -1)
    img_size_y = kwargs.get("img_size_y", -1)
    BPP = kwargs.get("BPP", 2)
    bloc_size = kwargs.get("bloc_size", 2)
    wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
    wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))

    if len((np.asarray(imagex)).shape)==3:
       long=3
    else:
       long=1
    image=np.copy(imagex)

    FAuth_watermark=[]
    FRec_watermark1=[]
    FRec_watermark2=[]

    #Loop on the watermarked image channels
    for channel in range (long): 
     if long==3:
         image=imagex[:, :, channel]
     else:
        image=imagex
     Auth_watermark=[]
     Rec_watermark1=[]
     Rec_watermark2=[]
     # Apply Discrete wavelet transform to the channel

     coeffs = pywt.dwt2(image, 'haar')             
     LL, (LH, HL, HH) = coeffs
     lis=["LH","HL","HH"]

     #Loop on the image subbands
     for subb in lis:                  
      if subb=="LL" : subband=LL
      elif subb=="LH" : subband=LH
      elif subb=="HL" : subband=HL
      elif subb=="HH" : subband=HH
      #Round the coefficient values to 5 numbers after the decimal point to avoid problems caused by DWT-IDWT 
      subband=np.round(subband,5)
      #Loop on each frequency subband
      a=0
      # while a+bloc_size<=len(subband):
      while a+bloc_size<=subband.shape[0]:       
        b=0
        # while b+bloc_size<=len(subband):
        while b+bloc_size<=subband.shape[1]:
            wat=""
            v=0
            #Loop on each block
            for j in range(bloc_size):
               for k in range(bloc_size):
                  #Stop if 8 bits are extracted from the current block or if all the watermark bits have been extracted
                  # if v*BPP==8 or len(Auth_watermark)==wat_size*wat_size:  break             
                  if v*BPP==8 or len(Auth_watermark)==wat_size_x*wat_size_y:  break
                  # Given that the coefficients values can be negative, transform them to positive values for binary transformation 
                  if subband[a+j][b+k]<0 :                                     
                        subband[a+j][b+k]=subband[a+j][b+k]*-1                                                
                  #Watermark bits extraction
                  int_part =int(subband[a+j][b+k])
                  pixel =dec_to_bin(int_part)  
                  wat=wat+pixel[len(pixel)-BPP:len(pixel)]
 
                  # if 8 bits have been extracted from the current block, append them to their corresponding subband
                  if len(wat)==8:          
                     wat=bin_to_dec(wat)  
                     if subb=="LH":
                        Rec_watermark1.append(wat)
                     elif subb=="HL":
                        Rec_watermark2.append(wat)
                     elif subb=="HH":
                        Auth_watermark.append(wat)
                     wat=""
                  v=v+1        
            b=b+bloc_size
        a=a+bloc_size

     #Reconstruct and decrypt the extracted watermarks 
     Auth_watermark=np.array(Auth_watermark)
     # Auth_watermark=Auth_watermark.reshape(wat_size,wat_size)
     Auth_watermark=Auth_watermark.reshape(wat_size_x,wat_size_y)
     Auth_watermark = np.array(Auth_watermark.astype("uint8"))
     Rec_watermark1=np.array(Rec_watermark1)
     # Rec_watermark1=Rec_watermark1.reshape(wat_size,wat_size)
     Rec_watermark1=Rec_watermark1.reshape(wat_size_x,wat_size_y)
     Rec_watermark1 = Rec_watermark1.astype("uint8")
     Rec_watermark2=np.array(Rec_watermark2)
     # Rec_watermark2=Rec_watermark2.reshape(wat_size,wat_size)
     Rec_watermark2=Rec_watermark2.reshape(wat_size_x,wat_size_y)
     Rec_watermark2 = Rec_watermark2.astype("uint8")
     if Auth_encryption==True:
        Auth_watermark = decrypt_image(Auth_watermark, key)
     if Rec_scrambling==True:
        Rec_watermark1 = unscramble_image(Rec_watermark1, key1)
        Rec_watermark2 = unscramble_image(Rec_watermark2, key2)
     FAuth_watermark.append(Auth_watermark)
     FRec_watermark1.append(Rec_watermark1)
     FRec_watermark2.append(Rec_watermark2)

    if long==3:
      Auth_watermark =  np.stack([FAuth_watermark[0], FAuth_watermark[1], FAuth_watermark[2]], axis=2)
      Rec_watermark1 =  np.stack([ FRec_watermark1[0],  FRec_watermark1[1], FRec_watermark1[2]], axis=2)
      Rec_watermark2 =  np.stack([ FRec_watermark2[0],  FRec_watermark2[1], FRec_watermark2[2]], axis=2)
    else:
      Auth_watermark =  FAuth_watermark[0]
      Rec_watermark1 =  FRec_watermark1[0]
      Rec_watermark2 =  FRec_watermark2[0]
    return(Auth_watermark,Rec_watermark1,Rec_watermark2)


def self_embedding(imagex, **kwargs):
   embedding_type = kwargs.get("embedding_type", "DWT")
   img_size_x = kwargs.get("img_size_x", -1)
   img_size_y = kwargs.get("img_size_y", -1)
   bloc_size = kwargs.get("bloc_size", 2)
   wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
   wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))
   # print("Received info: {}".format(kwargs))
   # print("Extracted info: {} {}".format(img_size_x, wat_size_x))
   img=np.copy(imagex)
   if len((np.asarray(img)).shape)==3:
     long=3
   else:
     long=1
   ww_arr=[]
   for channel in range (long): 
      if long==3:
         image=img[:, :, channel]
      else:
         image=img
      #In case of 12bit or 16bit image, normalize the image to to an 8-bit image
      if np.max(np.abs(image))>256:
         if np.max(np.abs(image))<4096:
            maaax=4095
         else: 
            maaax=65535
         img_norm = (image/ maaax) * 255
         image=np.round(img_norm,0)   
         if np.min(image)<0:
            image = (image + 255) /2 
      # print("before DWT", image.shape)
      if embedding_type=='DWT':
         coeffs = pywt.dwt2(image, 'haar')
         LL, (LH, HL, HH) = coeffs
         LL = (LL /600) * 255
         image=LL
      
      # Laurens: Changes to np.zeros for readability and optimization
      # Added support for non-squares
      watermark = np.zeros((wat_size_x, wat_size_y), dtype=np.uint8)
      # print(img_size_x, wat_size_x)
      down_size_x=int(img_size_x/wat_size_x)
      down_size_y=int(img_size_y/wat_size_y)
      # print("Self embedding: down_size ({}x{})".format(down_size_x, down_size_y))
      
      ii=0
      i=0
      # print(image.shape, down_size_x, down_size_y)
      while i+down_size_x < image.shape[0]:
         j=0
         jj=0
         while j+down_size_y < image.shape[1]:
             s=0.0
             # print(i, j, down_size_x, down_size_y)
             for k in range(down_size_x):
              for m in range(down_size_y):
                 s=s+image[i+k][j+m]
             sum=s/(down_size_x*down_size_y)
             watermark[ii][jj]=int(round(sum,0))
             jj=jj+1
             j=j+down_size_y
         i=i+down_size_x
         ii=ii+1
      ww_arr.append(watermark)

   if long==3:
      watermark =  np.stack([ww_arr[0], ww_arr[1], ww_arr[2]], axis=2)
   else:
      watermark =  ww_arr[0]
   return(watermark)
   

def Recovery_watermark_construction(img, long, **kwargs):
    img_size_x = kwargs.get("img_size_x", -1)
    img_size_y = kwargs.get("img_size_y", -1)
    bloc_size = kwargs.get("bloc_size", 2)
    wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
    wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))
    # THIS CODE WAS CHANGED   
    comp_arr=[]
    for channel in range (long):
        if long==3:
            image=img[:, :, channel]
        else:
            image=img
        # down_s=int(len(image[0])/wat_size)
        down_s_x=int(img_size_x/wat_size_x)
        down_s_y=int(img_size_y/wat_size_y)
        bloc = np.zeros((wat_size_x, wat_size_y), dtype=np.uint8)

        ii=0
        i=0
        # print(img.shape)
        # print(down_s, down_s_x, down_s_y)
        while i+down_s_x < img_size_x:
            j=0
            jj=0
            while j+down_s_y < img_size_y:
                s=0.0
                # print(i, j, down_s_x, down_s_y)
                for k in range(down_s_x):
                    for m in range(down_s_y):
                        s=s+image[i+k][j+m]
                        # s = int(s) + int(image[i + k][j + m])
                sum=s/(down_s_x*down_s_y)
                bloc[ii][jj]=int(round(sum,0))
                jj=jj+1
                j=j+down_s_y
            i=i+down_s_x
            ii=ii+1 
        # print(bloc)
        comp_arr.append(bloc)
    
    if long==3:
       watermarked =  np.stack([comp_arr[0], comp_arr[1], comp_arr[2]], axis=2)
    else:
       watermarked=comp_arr[0]
    return(watermarked)


def Tamper_detection(org_watermar, ext_watermar, **kwargs):
   self_embed = kwargs.get("self_embed", True)
   img_size_x = kwargs.get("img_size_x", -1)
   img_size_y = kwargs.get("img_size_y", -1)
   bloc_size = kwargs.get("bloc_size", 2)
   wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
   wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))
   
   #Given that the embedding is done bit-by-bit, val represents the number of different bits we tolerate between the binary reprentation of two pixels
   val=0  
   total=0
   if len((np.asarray(org_watermar)).shape)==3:
     long=3
   else:
     long=1

   t_arr=[]
   # 0 for altered pixels and 1 for unaltered ones
   # tamper = np.array([[0 for j in range(wat_size)] for i in range(wat_size)], dtype='uint8')
   tamper = np.zeros((wat_size_x, wat_size_y), dtype=np.uint8)
   for channel in range (long): 
      if long==3:
         og_watermark=org_watermar[:, :, channel]
         ex_watermark=ext_watermar[:, :, channel]
      else:
         og_watermark=org_watermar
         ex_watermark=ext_watermar
      # for i in range(wat_size):
      for i in range(wat_size_x):
         # for j in range(wat_size):
         for j in range(wat_size_y):
            if og_watermark[i][j]>ex_watermark[i][j]:
               diff=og_watermark[i][j]-ex_watermark[i][j]
            else:
               diff=ex_watermark[i][j]-og_watermark[i][j]
            #We use this thereshold only when the authentication watermark is genrated from the cover image, otherwise no need to use a threshold of 3
            if (diff<3) and (self_embed==True):  
                  tamper[i][j]=1 
            else : 
                  pixel=dec_to_bin(int(og_watermark[i][j]))
                  pixel2=dec_to_bin(int(ex_watermark[i][j]))
                  sum=0
                  for k in range(len(pixel)):
                     if pixel[k]!=pixel2[k]:
                        sum=sum+1
                  if sum>val:
                     tamper[i][j]=0
                     total=total+sum
                  else:
                     tamper[i][j]=1
      t_arr.append(tamper)
   
   final_tamper = np.copy(tamper)
   if long==3:
      for i in range (wat_size_x):
      # for i in range (wat_size):
         for j in range (wat_size_y):
         # for j in range (wat_size):
            if t_arr[0][i][j]==1 and t_arr[1][i][j]==1 and t_arr[2][i][j]==1:
               final_tamper[i][j]=1
            else:
               final_tamper[i][j]=0
   else:
      final_tamper=t_arr[0]

   # BER=total/(wat_size_x*wat_size_y*8)/long*100
   # print("Bit error rate BER: ",BER,"%")
   return(final_tamper)


def Tamper_localization(tamper, **kwargs):
   img_size_x = kwargs.get("img_size_x", -1)
   img_size_y = kwargs.get("img_size_y", -1)
   bloc_size = kwargs.get("bloc_size", 2)
   wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
   wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))
   #Perform a mojority vote between neighboords, if a pixel is unaltered by 4 or more of its neighboords are altered, the pixel is considered altered
   tamperx=np.copy(tamper)
   for i in range(wat_size_x):
   # for i in range(wat_size):
      for j in range(wat_size_y):
      # for j in range(wat_size):
         #Ensure that the pixel is not an edge pixel to perform majority vote
         if tamperx[i][j]==1 and i>0 and i<wat_size_x-1 and j>0 and j<wat_size_y-1:
         # if tamperx[i][j]==1 and i>0 and i<wat_size-1 and j>0 and j<wat_size-1:
            som=0
            ii=-1
            while ii<=1:
               jj=-1
               while jj<=1:
                  if tamperx[i+ii][j+jj]==0:
                     som=som+1
                  jj=jj+1  
               ii=ii+1
            if som>=4:
               tamper[i][j]=0

   #Can use dilatation and erosion operations, but the accuracy is not optimal
   return(tamper)
 

def recovery_process(imagex,tamper,Rec_watermark1,Rec_watermark2, **kwargs):
   key1 = kwargs.get("scramble_key1", None)
   key2 = kwargs.get("scramble_key2", None)
   img_size_x = kwargs.get("img_size_x", -1)
   img_size_y = kwargs.get("img_size_y", -1)
   bloc_size = kwargs.get("bloc_size", 2)
   wat_size_x = kwargs.get("wat_size_x", int(img_size_x/bloc_size/2))
   wat_size_y = kwargs.get("wat_size_y", int(img_size_y/bloc_size/2))
   if not any(0 in row for row in tamper):
      raise ValueError("No Tampering detected, Recovery impossible")
   
   image=np.copy(imagex)
   posttamp=np.copy(tamper)
   if len((np.asarray(image)).shape)==3:
      long=3
   else :
      long=1
   
   det1 = unscramble_image(posttamp, key1)
   det2 = unscramble_image(posttamp, key2)
   
   numb_x = int(img_size_x/wat_size_x)
   numb_y = int(img_size_y/wat_size_y)

   for i in range(wat_size_x):
      for j in range(wat_size_y):
         if posttamp[i][j]==0:
            if det1[i][j]==1 or det2[i][j]==1:
               if det1[i][j]==1:
                  pixel=Rec_watermark1[i][j]
               elif det2[i][j]==1:
                  pixel=Rec_watermark2[i][j]
               for a in range (numb_x):
                  for b in range (numb_y):      
                     image[i*numb_x+a][j*numb_y+b]=pixel
               posttamp[i][j]=1
   rec_img=np.copy(image)
   
   

   #Impainting method
   posttamp_without_edges = posttamp[1:-1, 1:-1]
   # while any(0 in row for row in posttamp):
   while any(0 in row for row in posttamp_without_edges):
    image=np.copy(rec_img)
    # print(image.shape, wat_size_x, wat_size_y)
    for i in range(1, wat_size_x -1):
      for j in range(1, wat_size_y - 1):
         if posttamp[i][j]==0:
            som1=0.0; som2=0.0; som3=0.0; n=0
            a=-1 
            while a<2:
               b=-1
               while b<2:
                  if a==0 and b==0: b=1     #Skip the current pixel
                  # if (i+a >= wat_size_x) or (j+b >= wat_size_y):
                  #    continue
                  # print(i, a, j, b)
                  if posttamp[i+a][j+b]==1:
                     sum11=0.0; sum22=0.0; sum33=0.0
                     if long==3:
                        for aa in range(numb_x):
                           for bb in range(numb_y):
                              sum11=sum11+image[(i+a)*numb_x+aa][(j+b)*numb_y+bb][0]
                              sum22=sum22+image[(i+a)*numb_x+aa][(j+b)*numb_y+bb][1]
                              sum33=sum33+image[(i+a)*numb_x+aa][(j+b)*numb_y+bb][2]
                        som1=som1+sum11/(numb_x*numb_y)
                        som2=som2+sum22/(numb_x*numb_y)
                        som3=som3+sum33/(numb_x*numb_y)
                     else:
                        for aa in range(numb_x):
                           for bb in range(numb_y):
                              sum11=sum11+image[(i+a)*numb_x+aa][(j+b)*numb_y+bb]
                        som1=som1+sum11/(numb_x*numb_y)
                     n=n+1
                  b=b+1
               a=a+1  
            if (n>=1):
               posttamp[i][j]=1
               for d in range (numb_x):
                  for f in range (numb_y): 
                     if long==3:    
                        image[i*numb_x+d][j*numb_y+f][0]=som1/n
                        image[i*numb_x+d][j*numb_y+f][1]=som2/n
                        image[i*numb_x+d][j*numb_y+f][2]=som3/n
                     else:
                        image[i*numb_x+d][j*numb_y+f]=som1/n

    rec_img=np.copy(image)  
   return(rec_img)