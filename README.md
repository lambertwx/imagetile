# imagetile
Takes a set of images and variably-shaped ROI masks, extracts the masked regions into rectangular tiles, and assembles into one image.  Useful for visualizing training examples that come from lots of images.

## Requirements
Python 3.5+

pillow - Python Imaging Library

scikit-image

rectpack - 2D-bin packing library.  Obtain via ```pip install rectpack```, or from https://github.com/secnot/rectpack/ . 

## Basic Usage

Extract the tiles from each of your images and masks.
```
tiles = extracttiles(img1, mask1, 5)
tiles.extend(extracttiles(img2, mask2, 5)
```

Then call ```layouttiles``` to arrange the tiles into a nice mosaic using 2D bin packing. 
mosaic = layouttiles(tiles)
