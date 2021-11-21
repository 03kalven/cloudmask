import numpy as np
from PIL import Image
from netCDF4 import Dataset
import pandas as pd
import os
import scipy.signal

def allThreeOnlyPixelTest(r, g, b):
    '''
    Simple check for a bad pixel. Will be labeled -999 in the final cloud mask. Returns (0,0,0) if the pixel is filtered by this function, returns the input pixel if not.

    Parameters:
    r,g,b -- 8-bit RGB pixel values
    '''
    if (g < 20 and b < 20) or (b < 30) or (r*g*b == 0):
        (r, g, b) = (0, 0, 0)

    return (r, g, b)


def allThreeOnly(inImg):
    '''
    Returns an a filtered version of an input PIL image. See allThreeOnlyPixelTest for the filter. 
    '''
    img = inImg.copy()
    for x in range(img.width):
        for y in range(img.height):
            rgb = img.getpixel((x, y))
            img.putpixel((x,y), allThreeOnlyPixelTest(*rgb))
    return img


def checkUNETNum(num):
    '''
    Returns the output dimenstion for a UNET if the input number is a valid UNET input, or None if the input is not possible.
    '''
    i = int(num)
    for k in range(4):
        i -= 4
        i /= 2
        if i != int(i):
            return 0
    for k in range(4):
        i -= 4
        i *= 2
    i -= 4
    return i


def checkUNETNumBack(num):
    '''
    Returns the input dimenstion for a UNET if the input number is a valid UNET output, or None if the output is not possible.
    '''
    i = int(num)
    for k in range(4):
        i += 4
        i /= 2
        if i != int(i):
            return None
    for k in range(4):
        i += 4
        i *= 2
    i += 4
    return i


def getPolarBounds(ls, c=0):
    '''
    Returns the southern and northen polar extents for a given solar longitude. Formulas provided by https://doi.org/10.1016/j.icarus.2019.05.041.

    Parameters:
    ls -- solar longitude (0 to 360)
    c -- degree offset. Moves the polar extents c degrees towards the center 
    '''
    latN = 59.0
    latS = -57.0

    if ls <= 90:
        latN = 59.0 + 0.214*ls-c
    elif ls <= 180:
        latN = 59.0 + 0.214*(180-ls)-c
    elif ls <= 270:
        latS = -12.0 - 0.247*ls+c
    elif ls <= 360:
        latS = -12.0 - 0.247*(540-ls)+c

    return (latN, latS)


def getPolarBoundsNew(ls, c=0):
    '''
    Returns the southern and northen polar extents for a given solar longitude. Formulas derived from https://doi.org/10.1016/j.icarus.2019.05.041 and my additional independent analysis.

    Parameters:
    ls -- solar longitude (0 to 360)
    c -- degree offset. Moves the polar extents c degrees towards the center
    '''
    if ls <= 180:
        latN = 59.0 + 0.214*(90-abs(90-ls))-c
        latS = -57.0 + 0.247*(90-abs(90-ls))+c
    else:
        latN = 59.0 - 0.214*(90-abs(270-ls))-c
        latS = -57.0 - 0.247*(90-abs(270-ls))+c

    return (latN, latS)


def boundToPixel(bound, size=1801, ax='lat'):
    '''
    Converts the latitude/longitude degree value to a pixel value, 0 is top and size is bottom.

    Parameters:
    bound -- latitude or longitude value, in degrees,  
    size --  pixels along the goven axis (default 1801)
    ax -- latitude ('lat') or longitude ('lon')
    '''
    if ax == 'lat':
        deg = 180
    elif ax == 'lon':
        deg = 360

    res = deg / size
    mid = size // 2
    pix = round(mid - bound / res)

    return pix

def pixelToBound(pix, size=1801, ax='lat'):
    '''
    Converts the pixel value to a latitude/longitude degree value.

    Parameters:
    bound -- latitude or longitude value, in degrees,  
    size --  pixels along the goven axis (default 1801)
    ax -- latitude ('lat') or longitude ('lon')
    '''
    if ax == 'lat':
        deg = 180
    elif ax == 'lon':
        deg = 360

    res = deg / size
    mid = size // 2
    bound = (pix - mid) * -1 * res

    return bound

def getCloudMaskBounds(clouds):
    '''
    Returns the upper and lower bounds of the cloud mask in pixels.

    Parameters:
    clouds -- numpy array of a given cloud mask
    '''
    (fullY, fullX) = clouds.shape
    y = 0
    top = 0
    bottom = 0

    while y < fullY:
        if clouds[y, :].max() >= 0:
            top = y
            break
        y += 1

    y = fullY - 1
    while y >= 0:
        if clouds[y, :].max() >= 0:
            bottom = y
            break
        y -= 1
    return (top, bottom)

def getCloudMask(maskPath):
    '''
    Takes a .ncdf cloudMask path and returns its values in a 2D numpy array
    '''
    cloudMask_ncdf = Dataset(maskPath)
    cloudMask = cloudMask_ncdf.variables['cloudmask'][:,:]
    cloudMask = np.flip(cloudMask, axis=0)
    return cloudMask

def binaryCloudMask(clouds):
    '''
    Takes an array of cloud mask probabilities and returns a binary classification
    '''
    cloudMask = clouds.copy()
    cloudMask[cloudMask < 0.5] = 0
    cloudMask[cloudMask >= 0.5] = 1
    return cloudMask

def blackBoundsPixelTest(r, g, b):
    '''
    Simple check for a black pixel. Uses 30 as a threshold for vales instead of 0 in case of JPEG compression. Returns (0,0,0) if the pixel is filtered by this function, returns the input pixel if not.

    Parameters:
    r,g,b -- 8-bit RGB pixel values
    '''
    if r < 30 or g < 30 or b < 30:
        (r, g, b) = (0, 0, 0)

    return (r, g, b)


def getBlackBounds(img, sides='', yTrim = None):
    '''
    Returns the first left and right bounds of an MDGM that are NOT entirely black. Uses blackBoundsPixelTest to check for black pixel.

    Parameters:
    img -- PIL image of the RGB MDGM
    sides -- 'l' if want to crop left bound, 'r' if want to crop right bound, 'lr' if want to crop both, '' if want to crop neither (default '')
    yTrim -- an optional parameter to bound the y axis in the search. Must be in the form of a tuple: (min, max) (default None)
    '''
    xsize, ysize = img.size
    
    minY = 0
    maxY = ysize
    minX = 0
    maxX = xsize

    if yTrim is not None:
        minY, maxY = yTrim
        maxY += 1

    if 'l' in sides:
        # crop left
        for x in range(xsize):
            for y in range(minY, maxY):
                rgb = img.getpixel((x, y))
                if max(rgb) > 0 and blackBoundsPixelTest(*rgb) == rgb:
                    minX = x
                    break
            if minX != 0:
                break
    if 'r' in sides:
        # crop right
        for x in reversed(range(xsize)):
            for y in range(minY, maxY):
                rgb = img.getpixel((x, y))
                if max(rgb) > 0 and blackBoundsPixelTest(*rgb) == rgb:
                    maxX = x
                    break
            if maxX != xsize:
                break
    return (minX, maxX)

def getInfo_train(imgPath):
    '''
    Retrieves the solar longitude and martian year for a given photo located in the directory structure below. Cloudmasks retrieved from https://doi.org/10.7910/DVN/WU6VZ8 should satisfy this structure:

    B01
    │   list_ls.txt    
    │
    └───cloudmask
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │   
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    '''
    path = os.path.abspath(imgPath)
    mdgmSplit = os.path.split(path)
    subsetPath = os.path.split(mdgmSplit[0])[0]
    lsPath = os.path.join(subsetPath,'list_ls.txt') 
    subsetAndDay = mdgmSplit[1][:8]

    allLS = pd.read_table(lsPath, delim_whitespace=True, names=['name', 'my', 'ls'], engine='python')
    ls = float(allLS[allLS['name'] == '{}'.format(subsetAndDay)]['ls'])
    my = float(allLS[allLS['name'] == '{}'.format(subsetAndDay)]['ls'])
    return ls, my

def getInfo_compute(imgPath):
    '''
    Retrieves the solar longitude and martian year for a given photo located in the directory structure below. Data retrieved from https://doi.org/10.7910/DVN/U3766S should satisfy this structure:

    B
    │
    └───B01
    │   │   B01_ls.txt    
    │   │
    │   │   B01_day01_zequat.jpg
    │   │   B01_day02_zequat.jpg
    │   │   B01_day32_zequat.jpg
    │   │   ...
    │   │
    │   └───list
    │       │   B01_day01.list
    │       │   B01_day02.list
    │       │   B01_day03.list
    │       │   ...
    │
    └───B02
    │
    └───B03
    ...
    '''
    path = os.path.abspath(imgPath)
    subsetSplit = os.path.split(path)
    subsetAndDay = subsetSplit[1][:9]
    lsPath = os.path.join(subsetSplit[0],'{}_ls.txt'.format(subsetAndDay[:3]))
    allLS = pd.read_table(lsPath, delim_whitespace=True, names=['name', 'my', 'ls'], engine='python')
    ls = float(allLS[allLS['name'] == subsetAndDay]['ls'])
    my = float(allLS[allLS['name'] == subsetAndDay]['ls'])
    return ls, my

def getInfo_model(imgPath):
    '''
    Retrieves the solar longitude and martian year for a given photo/cloudmask pair located in the directory structure below. Cloudmasks generated by the model should satisfy this structure:

    B01
    │   B01_ls.txt    
    │
    └───cloudmask
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │   
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    '''
    path = os.path.abspath(imgPath)
    mdgmSplit = os.path.split(path)
    subsetPath = os.path.split(mdgmSplit[0])[0]
    lsPath = os.path.join(subsetPath,'{}_ls.txt'.format(mdgmSplit[1][:3]))
    subsetAndDay = mdgmSplit[1][:9]

    allLS = pd.read_table(lsPath, delim_whitespace=True, names=['name', 'my', 'ls'], engine='python')
    ls = float(allLS[allLS['name'] == subsetAndDay]['ls'])
    my = float(allLS[allLS['name'] == subsetAndDay]['my'])
    return ls, my

def getCloudMask_train(imgPath):
    '''
    Retrieves the cloudmask for a given photo located in the directory structure below. Cloudmasks retrieved from https://doi.org/10.7910/DVN/WU6VZ8 should satisfy this structure:

    B01
    │   list_ls.txt    
    │
    └───cloudmask
    │   │   cloudmask_B01day01.ncdf
    │   │   cloudmask_B01day02.ncdf
    │   │   cloudmask_B01day03.ncdf
    │   │   ...
    │   
    └───mdgms
        │   B01day01.jpeg
        │   B01day02.jpeg
        │   B01day03.jpeg
        │   ...
    '''
    path = os.path.abspath(imgPath)
    mdgmSplit = os.path.split(path)
    subsetPath = os.path.split(mdgmSplit[0])[0]
    subsetAndDay = mdgmSplit[1][:8]

    cloudMaskPath = os.path.join(subsetPath, 'cloudmask', 'cloudmask_{}.ncdf'.format(subsetAndDay))
    return cloudMaskPath

def getCloudMask_predicted(imgPath):
    '''
    Retrieves the cloudmask for a given photo located in the directory structure below. Cloudmasks from this ML model should satisfy this structure:

    B01
    │   list_ls.txt    
    │
    └───cloudmasks
    │   │   cloudmask_B01day_01.ncdf
    │   │   cloudmask_B01day_02.ncdf
    │   │   cloudmask_B01day_03.ncdf
    │   │   ...
    │   
    └───mdgms
        │   B01day_01.jpeg
        │   B01day_02.jpeg
        │   B01day_03.jpeg
        │   ...
    '''
    path = os.path.abspath(imgPath)
    mdgmSplit = os.path.split(path)
    subsetPath = os.path.split(mdgmSplit[0])[0]
    subsetAndDay = mdgmSplit[1][:9]

    cloudMaskPath = os.path.join(subsetPath, 'cloudmasks', 'cloudmask_{}.ncdf'.format(subsetAndDay))
    return cloudMaskPath

def padMDGM(mdgm, highX, lowX, highY, lowY):
    '''
    Prepares an inputted mdgm for subdividing. Crops the image's black borders and polar extents out, and adds 92 pixels of each end of the mdgm to the opposite ends.

    Parameters:
    mdgm -- a PIL image
    highX & lowX -- the x coordinate for cropping (highX > lowX, 0 is left)
    highY & lowY -- the y coordinates for cropping (highY > lowY, 0 is top)
    '''
    # for cropping, use highX + 1 and highY + 1 because that index is the first completely black column
    croppedMDGM = mdgm.crop((lowX,lowY-92,highX+1,highY+93))
    croppedW, croppedH = croppedMDGM.size
    paddedMDGM = Image.new('RGB', (croppedW + 2*92, croppedH))

    (fullW,fullH) = paddedMDGM.size
    paddedMDGM.paste(croppedMDGM.crop((croppedW-92,0,croppedW,fullH)), (0,0,92,fullH))
    paddedMDGM.paste(croppedMDGM, (92,0,croppedW+92,fullH))
    paddedMDGM.paste(croppedMDGM.crop((0,0,92,fullH)), (fullW-92,0,fullW,fullH))
    
    return paddedMDGM

def _spline_window(window_size, power=2):
    '''
    {Credit: Vooban, https://github.com/Vooban/Smoothly-Blend-Image-Patches}
    Returns a spline function:
    
    Parameters:
    window_size -- the domain of the spline function [0, window_size]
    power -- power of the spline function (default 2)
    '''
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

def window_2D(window_size, power=2):
    '''
    {Credit: Vooban, https://github.com/Vooban/Smoothly-Blend-Image-Patches}
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.

    Parameters:
    window_size -- the domain of the spline function [0, window_size]
    power -- power of the spline function (default 2)
    '''
    wind = _spline_window(window_size, power)
    wind = np.expand_dims(wind, 1)
    wind = wind * wind.transpose(1, 0)
    return np.expand_dims(wind, axis=0)