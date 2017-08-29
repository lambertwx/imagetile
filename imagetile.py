# -*- coding: utf-8 -*-
"""
Functions that take a bunch of arbitrary-sized rectangular images and lays them out into a single image.
"""
# MIT License

# Copyright Lambert Wixson(c) 2017.  Downloadable from https://github.com/lambertwx/imagetile

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from PIL import Image
from rectpack import newPacker
from skimage.measure import label, regionprops

#%%
def makebool3mask(mask1 : np.ndarray):
    """
    Given a boolean 2D mask, makes an equivalent mask with 3 boolean bands.
    """
    mask3 = np.zeros([mask1.shape[0], mask1.shape[1], 3],dtype='bool')
    for i in range(3):
        mask3[:,:,i] = mask1.copy()
    return mask3

#%%
def extracttiles(img : np.ndarray, mask : np.ndarray, pad : int):
    """
    Extract a set of tiles, one for each connected component in the mask image, from a single image.
    
    @param img: An r x c x 3 image.  E.g. an RGB image from scikit-image or ndimage.
    
    @param mask: A boolean r x c image. 
    
    @param pad: Border to add, in pixels.
  
    @return: A list containing the image tiles.  Each image tile is an nparray containing the region specified by the bounding box of its connected component in the mask, padded.
    """
    assert(img.shape[0] == mask.shape[0])
    assert(img.shape[1] == mask.shape[1])
    # Create one tile for each separate region in the mask
    labeled = label(mask.astype('uint8'), connectivity=2)
    regions = regionprops(labeled)
    
    tiles = []
    for reg in regions:   
        rtop = reg.bbox[0]
        rleft = reg.bbox[1]
        rbot = reg.bbox[2]
        rright = reg.bbox[3]
        
        ttop = max(0, rtop - pad)
        tleft = max(0, rleft - pad)
        tright = min(img.shape[1], rright + pad)
        tbot = min(img.shape[0], rbot + pad)
    
        # Now we want to copy the region bounded by the tile coords into a new image.
        # However that bounding box might also include another region in which the mask was on.
        # So we first need to be sure that we have an image in which only this region is present.
        regmask = makebool3mask(labeled == reg.label)
        c = np.zeros(img.shape, dtype=img.dtype)
        np.copyto(c, img, where=regmask)
        tiles.append( c[ttop:tbot,tleft:tright,:].copy() )
        
    return tiles

#%%
def layouttiles(images):
    """
    Takes a bunch of images (each may have a unique size) and lays them out in a single image.
    
    @param images: list of numpy.ndarray objects, each representing one image
    """
    sumw = 0
    sumh = 0
    sumarea = 0
    for img in images:
        sumh += img.shape[0]
        sumw += img.shape[1]
        sumarea += (img.shape[0] * img.shape[1])
    
    trialw = int(np.ceil(np.sqrt(sumarea)))
    trialh = int(np.ceil(np.sqrt(sumarea)))
    
    packer = None
    
    while True:
        packer = newPacker()
        packer.add_bin(trialw, trialh)
        
        for i, img in enumerate(images):
            packer.add_rect(img.shape[1], img.shape[0], i)
        
        packer.pack()
        if (len(packer) == 1) and (len(packer[0]) == len(images)):
            # If we get here, all the images fit successfully.  Now generate the final image.
            break
        
        # If we get here, then not all the images fit into the bounding bin.  So boost the 
        # area of the bin by 50% and try again
        trialw = int(np.ceil(trialw * np.sqrt(1.5)))
        trialh = int(np.ceil(trialh * np.sqrt(1.5)))     
        
    # Now generate the final image
    out = Image.new('RGB', (trialh, trialw), color=(0,0,0))
    all_rects = packer.rect_list()
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        print(rid)
        pim = Image.fromarray(images[rid])
        out.paste(pim, (y, x))
    
    return np.array(out)
    
        
       
        