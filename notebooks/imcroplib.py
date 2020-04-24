from PIL import Image
import PIL.ImageOps
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def im2square(path):
    im = Image.open(path)
    arr = rgb2gray(255 - np.array(im))
    h = np.sum(arr, axis=0)
    hi = h[::-1]
    v = np.sum(arr, axis=1)
    vi = v[::-1]
    left = np.argmax(h > 0)
    right = len(h) - np.argmax(hi > 0)
    top = np.argmax(v > 0)
    bottom = len(v) - np.argmax(vi > 0)
    width = right-left
    height = bottom-top
    if width < height:
        diff = height-width
        half = int(diff/2)
        left-=half
        right+=diff-half
    elif height < width:
        diff = width-height
        half = int(diff/2)
        top-=half
        bottom+=diff-half
    return im.crop((left, top, right, bottom))

def originalim(path):
    return Image.open(path)

def iminvert(path):
    im = Image.open(path)
    PIL.ImageOps.invert(im).save(path)